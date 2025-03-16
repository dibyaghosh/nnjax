import numpy as np
import flax
from gemma import reference, transformer, sampler  # noqa
import jax
import jax.numpy as jnp
import nnjax

import orbax.checkpoint as ocp
from etils import epath

DEFAULT_PATH = epath.Path("/nfs/nfs2/users/dibya/gemma-2/")


def _load_tokenizer(path: epath.Path = DEFAULT_PATH / "tokenizer.model"):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(path))


def _load_params(path: epath.Path = DEFAULT_PATH / "gemma2-2b/"):
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(path)
    params = flax.traverse_util.flatten_dict(params, sep="/")
    new_params = {k: v for k, v in params.items() if "transformer/layer_" not in k}

    for k in filter(lambda x: "transformer/layer_0/" in x, params.keys()):
        new_params[k.replace("/layer_0/", "/layers/")] = np.stack(
            [params[k.replace("layer_0", f"layer_{i}")] for i in range(26)]
        )
    new_params["transformer/layers/mlp/gating_einsum"] = new_params.pop(
        "transformer/layers/mlp/gating_einsum/w"
    )
    new_params["transformer/layers/mlp/linear"] = new_params.pop(
        "transformer/layers/mlp/linear/w"
    )
    new_params = flax.traverse_util.unflatten_dict(new_params, sep="/")
    return new_params


@jax.jit
def _run_reference(params, tokens):
    reference_model = reference.Model(
        **{**reference.get_config("gemma2_2b").to_dict(), "vocab_size": 256_128}
    )
    return reference_model.apply({"params": params['transformer']}, tokens)


@jax.jit
def _run_ours(params, tokens):
    model = transformer.Gemma.create_from_pretrained(params, transformer.GemmaConfig.gemma2_2b())
    intermediates = {}
    with nnjax.capture_intermediates(intermediates):
        logits, _ = model(tokens)
    return logits, intermediates


def _cossim(a, b, axis=None):
    ab = (a * b).sum(axis=axis)
    a_norm = (a * a).sum(axis=axis)
    b_norm = (b * b).sum(axis=axis)
    return ab * jax.lax.rsqrt(a_norm * b_norm)


def test_equality():
    params = _load_params()
    tokenizer = _load_tokenizer()
    tokens = tokenizer.encode("Continue this sequence: 1 1 2 3 5", add_bos=True)
    tokens = jnp.array(tokens)[None]
    _, reference_extra = _run_reference(params, tokens)
    _, pred_extra = _run_ours(params, tokens)
    print(_cossim(reference_extra["encoded"], pred_extra["encoded"], axis=-1))
    assert jnp.all(_cossim(reference_extra["encoded"], pred_extra["encoded"]) > 0.99)

    sampled_tokens = sample(params, tokens)
    print(sampled_tokens)
    print("Continue this sequence: 1 1 2 3 5")
    print(tokenizer.decode(sampled_tokens[0].tolist()))

@jax.jit
def sample(params, tokens):
    model = transformer.Gemma.create_from_pretrained(params, transformer.GemmaConfig.gemma2_2b())
    return sampler.sample(model, tokens, jnp.ones_like(tokens, dtype=jnp.int32), 32, 1, jax.random.PRNGKey(0))


if __name__ == "__main__":
    test_equality()
