from . import transformer
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array, Key
from nnjax import pytree_dataclass
import dataclasses
from functools import partial


def right_align(tokens, input_mask):
    def _right_align(tokens, input_mask):
        first_invalid = input_mask.argmin()
        tokens = jnp.roll(tokens, tokens.shape[-1] - first_invalid)
        input_mask = jnp.roll(input_mask, tokens.shape[-1] - first_invalid)
        return tokens, input_mask

    return jax.vmap(_right_align)(tokens, input_mask)


@pytree_dataclass
class SampleState:
    kv_cache: transformer.LayerKVCache
    input_mask: Int[Array, "b cache_size"]
    last_logits: Int[Array, "b vocab_size"]
    last_positions: Int[Array, "b 1"]
    token_buffer: Int[Array, "b max_length"]
    done: Int[Array, "b 1"]
    idx: Int[Array, ""]
    key: Key


@partial(
    jax.jit,
    static_argnames=("max_length", "cache_dtype"),
)
def sample(
    model: transformer.Gemma,
    initial_tokens: Int[Array, "b l"],
    input_mask: Int[Array, "b l"],
    max_length: int,
    eos_token: int,
    key: Key,
    cache_dtype="bfloat16",
) -> Int[Array, "b max_length"]:
    bs, prefill_length = initial_tokens.shape
    print(bs, prefill_length)
    cache_size = max_length + prefill_length
    kv_cache = model.init_cache(bs, cache_size, cache_dtype)
    initial_tokens, input_mask = right_align(initial_tokens, input_mask)

    positions = jnp.cumsum(input_mask, axis=-1)

    padded_input_mask = jnp.pad(input_mask, [(0, 0), (0, max_length)]).astype(jnp.bool_)
    mask = jnp.logical_and(
        (jnp.arange(prefill_length)[:, None] >= jnp.arange(cache_size)[None, :]),
        padded_input_mask[:, None, None, :],
    )

    init_logits, kv_cache = model(
        initial_tokens, positions=positions, mask=mask, kv_cache=kv_cache, cache_idx=0
    )

    def sample_tokens(logits, key):
        return jax.random.categorical(key, logits)

    def sample_step(state: SampleState):
        cache_idx = state.idx + prefill_length

        key, new_key = jax.random.split(state.key)
        tokens = sample_tokens(state.last_logits, key)
        new_token_buffer = jax.lax.dynamic_update_slice_in_dim(
            state.token_buffer, tokens, state.idx, axis=1
        )
        done = jnp.logical_or(tokens == eos_token, state.done)
        not_done = jnp.logical_not(done)

        new_input_mask = jax.lax.dynamic_update_slice_in_dim(
            state.input_mask, not_done, cache_idx, axis=1
        )
        positions = state.last_positions + not_done.astype(jnp.int32)
        mask = jnp.logical_and(
            (cache_idx >= jnp.arange(cache_size)[None, :]),
            new_input_mask[:, None, None, :],
        )
        new_logits, kv_cache = model(
            tokens,
            positions=positions,
            mask=mask,
            kv_cache=state.kv_cache,
            cache_idx=cache_idx,
        )

        return dataclasses.replace(
            state,
            kv_cache=kv_cache,
            input_mask=new_input_mask,
            last_logits=new_logits,
            last_positions=positions,
            token_buffer=new_token_buffer,
            done=done,
            key=new_key,
            idx=state.idx + 1,
        )

    def _continue_sampling(state: SampleState):
        return jnp.logical_and(
            jnp.any(jnp.logical_not(state.done)), state.idx < max_length
        )

    init_state = SampleState(
        kv_cache=kv_cache,
        input_mask=padded_input_mask,
        last_logits=init_logits[:, -1:],
        last_positions=positions[:, -1:],
        token_buffer=jnp.zeros((bs, max_length), dtype=jnp.int32),
        done=jnp.zeros((bs, 1), dtype=jnp.bool_),
        idx=jnp.zeros((), dtype=jnp.int32),
        key=key,
    )
    final_state = jax.lax.while_loop(_continue_sampling, sample_step, init_state)
    return final_state.token_buffer
