# ruff: noqa: F722
from __future__ import annotations
import jax.nn as nn
from jaxtyping import Float, Array, Int
import jax.numpy as jnp
import jax
from nnjax import pytree_dataclass, static_field
import nnjax
import einops
import dataclasses
from functools import partial
from typing import Literal, Annotated

Dtypelike = jnp.dtype | str

def trunc_norm_init(in_axis, out_axis, batch_axis):
    return nn.initializers.variance_scaling(
        1.0,
        "fan_in",
        "truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
    )


@dataclasses.dataclass  # just a normal dataclass
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    vocab_size: int
    dtype: Dtypelike
    remat_policy: str
    final_logits_softcap: float | None = None
    attn_logits_softcap: float | None = None
    query_pre_attn_norm: Literal["rsqrt_head_dim", "rsqrt_emb_per_head"] = (
        "rsqrt_head_dim"
    )
    post_norms: bool = False

    @classmethod
    def gemma_2b(cls, dtype: Dtypelike = "float32"):
        return cls(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            norm_eps=1e-6,
            vocab_size=256_000,
            dtype=dtype,
            remat_policy="nothing_saveable",
        )

    @classmethod
    def gemma2_2b(cls, dtype: Dtypelike = "float32"):
        return cls(
            width=2304,
            depth=26,
            mlp_dim=9216,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            norm_eps=1e-6,
            vocab_size=256_000,
            dtype=dtype,
            final_logits_softcap=30.0,
            attn_logits_softcap=50.0,
            post_norms=True,
            remat_policy="nothing_saveable",
        )


@pytree_dataclass
class Einsum:
    w: jax.Array
    dtype: Dtypelike = static_field()

    def __call__(self, x, pattern: str):
        return jnp.einsum(pattern, x.astype(self.dtype), self.w.astype(self.dtype)).astype(x.dtype)


@pytree_dataclass
class RMSNorm:
    scale: jax.Array

    def __call__(self, x: Float[Array, "..."]):
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        out = x * jax.lax.rsqrt(var + 1e-6) * (1 + self.scale)
        return out.astype(in_dtype)

    @classmethod
    def create(cls, _, embedding_dim: int):
        scale = jnp.zeros(embedding_dim, dtype=jnp.float32)
        return cls(scale)


@pytree_dataclass
class FeedForward:
    gating_einsum: Float[Array, "2 features hidden_dim"]
    linear: Float[Array, "hidden_dim features"]
    dtype: Dtypelike = static_field()

    def __call__(self, x: Float[Array, "b l d"]):#
        ff_gate, ff1 = jnp.einsum("bld,2dh->2blh", x.astype(self.dtype), self.gating_einsum.astype(self.dtype))
        activations = nn.gelu(ff_gate) * ff1
        out = jnp.einsum("blh,hd->bld", activations, self.linear.astype(self.dtype))
        return out.astype(x.dtype)

    @classmethod
    def create(
        cls,
        key: jax.Array,
        embedding_dim: int,
        mlp_dim: int,
        dtype: Dtypelike,
    ):
        key, key_dense_1, key_dense_2 = jax.random.split(key, 3)
        gating_einsum = trunc_norm_init(in_axis=(1,), out_axis=(0, 2), batch_axis=())(
            key_dense_1, (2, embedding_dim, mlp_dim)
        )
        linear = trunc_norm_init(in_axis=(0,), out_axis=(1,), batch_axis=())(
            key_dense_2, (mlp_dim, embedding_dim)
        )
        return cls(gating_einsum, linear, dtype=dtype)


def _apply_rope(x, *, positions, max_wavelength: float = 10_000):
    positions = jnp.expand_dims(positions, axis=range(positions.ndim, x.ndim))

    freq_exponents = jnp.linspace(0, 1, x.shape[-1] // 2 + 1)[:-1]
    timescale = max_wavelength**freq_exponents
    radians = positions / timescale  # (..., d//2)
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


LayerKVCache = tuple[Float[Array, "b cache_size n d"], Float[Array, "b cache_size n d"]]


@pytree_dataclass
class AttentionBlock:
    q_einsum: Einsum
    kv_einsum: Einsum
    attn_vec_einsum: Einsum

    num_heads: int = static_field()
    num_kv_heads: int = static_field()
    head_dim: int = static_field()
    dtype: Dtypelike = static_field()
    attn_logits_softcap: float | None = static_field()
    query_pre_attn_norm: Literal["rsqrt_head_dim", "rsqrt_emb_per_head"] = (
        static_field()
    )

    def __call__(
        self,
        x: Float[Array, "b nq d"],
        positions: Float[Array, "b nq"],
        mask: Float[Array, "b 1 nq nk"],
        kv_cache: LayerKVCache | None = None,
        cache_idx: int | None = None,
    ):
        dtype = x.dtype
        q = self.q_einsum(x, "bqd,ndh->bqnh")
        k, v = self.kv_einsum(x, "bqd,2ndh->2bqnh")
        q = _apply_rope(q, positions=positions)
        k = _apply_rope(k, positions=positions)

        if self.query_pre_attn_norm == "rsqrt_head_dim":
            q *= self.head_dim**-0.5
        elif self.query_pre_attn_norm == "rsqrt_emb_per_head":
            q *= (self.features // self.num_heads) ** -0.5
        else:
            raise ValueError(f"Unknown query_pre_attn_norm: {self.query_pre_attn_norm}")

        if kv_cache is not None:
            cache_dtype = kv_cache[0].dtype
            k_cache = jax.lax.dynamic_update_slice_in_dim(
                kv_cache[0], k.astype(cache_dtype), cache_idx, axis=1
            )
            v_cache = jax.lax.dynamic_update_slice_in_dim(
                kv_cache[1], v.astype(cache_dtype), cache_idx, axis=1
            )
            kv_cache = (k_cache, v_cache)
            k, v = k_cache.astype(k.dtype), v_cache.astype(v.dtype)
        else:
            kv_cache = (k, v)

        q = einops.rearrange(
            q,
            "b q (n g) h -> b q n g h",
            n=self.num_kv_heads,
            g=self.num_heads // self.num_kv_heads,
        )
        logits = jnp.einsum("bqngh,bknh->bngqk", q, k)
        logits = logits.astype(jnp.float32)
        if self.attn_logits_softcap is not None:
            # bound logits to [-softcap, softcap]
            logits = (
                jnp.tanh(logits / self.attn_logits_softcap) * self.attn_logits_softcap
            )

        logits = jnp.where(mask[:, :, None, :, :], logits, -2.3819763e38)
        probs = nn.softmax(logits, axis=-1).astype(k.dtype)
        encoded = jnp.einsum("bngqk,bknh->bqngh", probs, v)
        encoded = einops.rearrange(
            encoded,
            "b q n g h -> b q (n g) h",
            n=self.num_kv_heads,
            g=self.num_heads // self.num_kv_heads,
        )
        encoded = self.attn_vec_einsum(encoded, "bqnh,nhd->bqd")
        return encoded.astype(dtype), kv_cache

    @classmethod
    def create(
        cls,
        key: jax.Array,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: Dtypelike,
        attn_logits_softcap: float | None,
        query_pre_attn_norm: Literal["rsqrt_head_dim", "rsqrt_emb_per_head"] = (
            "rsqrt_head_dim"
        ),
    ):
        key, key_q_proj, key_k_proj, key_v_proj, key_out_proj = jax.random.split(key, 5)
        query_einsum = Einsum(
            trunc_norm_init(in_axis=(1,), out_axis=(0, 2), batch_axis=())(
                key_q_proj, (num_heads, embedding_dim, head_dim)
            ),
            dtype=dtype,
        )
        kv_einsum = Einsum(
            trunc_norm_init(in_axis=(2,), out_axis=(0, 1, 3), batch_axis=())(
                key_k_proj, (2, num_kv_heads, embedding_dim, head_dim)
            ),
            dtype=dtype,
        )
        attn_vec_einsum = Einsum(
            trunc_norm_init(in_axis=(0, 1), out_axis=(2,), batch_axis=())(
                key_out_proj, (num_heads, head_dim, embedding_dim)
            ),
            dtype=dtype,
        )
        return cls(
            q_einsum=query_einsum,
            kv_einsum=kv_einsum,
            attn_vec_einsum=attn_vec_einsum,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            attn_logits_softcap=attn_logits_softcap,
            query_pre_attn_norm=query_pre_attn_norm,
        )


@pytree_dataclass
class TransformerLayer:
    pre_attention_norm: RMSNorm
    attn: AttentionBlock
    post_attention_norm: RMSNorm | None

    pre_ffw_norm: RMSNorm
    mlp: FeedForward
    post_ffw_norm: RMSNorm | None

    def __call__(
        self,
        x: Float[Array, "b l d"],
        positions: Float[Array, "b l"],
        mask: Float[Array, "b 1 l l"],
        kv_cache: LayerKVCache | None = None,
        cache_idx: int | None = None,
    ) -> Float[Array, "b l d"]:
        out = x
        # Attn
        y = nnjax.capture("pre_attention_norm", self.pre_attention_norm(out))
        y, kv_cache = self.attn(y, positions, mask, kv_cache, cache_idx)
        nnjax.capture("attn", y)
        if self.post_attention_norm is not None:
            y = self.post_attention_norm(y)
            nnjax.capture("post_attention_norm", y)
        out = out + y

        # MLP
        y = nnjax.capture("pre_ffw_norm", self.pre_ffw_norm(out))
        y = nnjax.capture("mlp", self.mlp(y))
        if self.post_ffw_norm is not None:
            y = nnjax.capture("post_ffw_norm", self.post_ffw_norm(y))
        out = out + y
        nnjax.capture("outputs", out)
        return out, kv_cache

    @classmethod
    def create(cls, key: jax.Array, config: GemmaConfig):
        keys = jax.random.split(key, 6)
        pre_attention_norm = RMSNorm.create(keys[0], config.width)
        attn = AttentionBlock.create(
            keys[1],
            embedding_dim=config.width,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            attn_logits_softcap=config.attn_logits_softcap,
            query_pre_attn_norm=config.query_pre_attn_norm,
        )
        pre_ffw_norm = RMSNorm.create(keys[3], config.width)
        mlp = FeedForward.create(
            keys[4],
            embedding_dim=config.width,
            mlp_dim=config.mlp_dim,
            dtype=config.dtype,
        )

        if config.post_norms:
            post_attention_norm = RMSNorm.create(keys[2], config.width)
            post_ffw_norm = RMSNorm.create(keys[5], config.width)
        else:
            post_attention_norm = None
            post_ffw_norm = None

        return cls(
            pre_attention_norm=pre_attention_norm,
            attn=attn,
            post_attention_norm=post_attention_norm,
            pre_ffw_norm=pre_ffw_norm,
            mlp=mlp,
            post_ffw_norm=post_ffw_norm,
        )


@pytree_dataclass
class Transformer:
    layers: TransformerLayer
    final_norm: RMSNorm
    depth: int = static_field()

    def __call__(
        self,
        x: Float[Array, "b l d"],
        positions: Float[Array, "b l"],
        mask: Float[Array, "b 1 l l"],
        kv_cache: LayerKVCache | None = None,
        cache_idx: int | None = None,
    ) -> Float[Array, "b l d"]:
        def body(x, inputs):
            layer, cache = inputs
            return layer(
                x, positions=positions, mask=mask, kv_cache=cache, cache_idx=cache_idx
            )

        x, carry = nnjax.scan(body, x, (self.layers, kv_cache))
        nnjax.capture("encoded", x)
        return self.final_norm(x), carry

    @classmethod
    def create(cls, key: jax.Array, config: GemmaConfig):
        layers = jax.vmap(
            partial(TransformerLayer.create, config=config),
        )(jax.random.split(key, config.depth))
        final_norm = RMSNorm.create(key, config.width)
        return cls(layers, final_norm, config.depth)


@pytree_dataclass
class Embedder:
    input_embedding: Float[Array, "vocab_size embed_dim"]

    def encode(self, x: Int[Array, "..."]) -> Float[Array, "..."]:
        x = self.input_embedding[(x,)]
        x = x * jnp.sqrt(self.input_embedding.shape[1]).astype(x.dtype)
        return x

    def decode(
        self, x: Float[Array, "... embed_dim"]
    ) -> Float[Array, "... vocab_size"]:
        return jnp.dot(x, self.input_embedding.T)

    @classmethod
    def create(cls, key: jax.Array, vocab_size: int, embedding_dim: int):
        init = nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
            in_axis=1,
            out_axis=0,
        )
        return cls(init(key, (vocab_size, embedding_dim)))


@pytree_dataclass
class Gemma:
    embedder: Embedder
    transformer: Transformer
    final_logits_softcap: float | None = static_field()
    dtype: Dtypelike = static_field()
    
    def __call__(
        self,
        tokens: Int[Array, "b l"],
        positions: Float[Array, "b l"] = None,
        mask: Float[Array, "b 1 l cache_size"] = None,
        kv_cache: LayerKVCache | None = None,
        cache_idx: int | None = None,
    ) -> tuple[Float[Array, "b l vocab_size"], LayerKVCache]:
        b, l = tokens.shape

        embeddings = self.embedder.encode(tokens).astype(self.dtype)
        nnjax.capture("embeddings", embeddings)

        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(l).astype(jnp.int32)[None, :], (b, l))

        if mask is None:
            # Add a causal mask to the query indices
            if kv_cache is not None:
                query_indices = jnp.arange(l) + cache_idx
                key_indices = jnp.arange(kv_cache[0].shape[2])
            else:
                query_indices = jnp.arange(l)
                key_indices = jnp.arange(l)

            mask = query_indices[:, None] >= key_indices[None, :]
            mask = jnp.broadcast_to(mask[None, None, :, :], (b, 1, len(query_indices), len(key_indices)))

        output_embeddings, new_kv_cache = self.transformer(
            embeddings, positions, mask=mask, kv_cache=kv_cache, cache_idx=cache_idx
        )
        nnjax.capture("pre_logits", output_embeddings)
        logits = self.embedder.decode(output_embeddings)
        nnjax.capture("logits_pre_norm", logits)
        if self.final_logits_softcap is not None:
            logits = jnp.tanh(logits / self.final_logits_softcap) * self.final_logits_softcap
        nnjax.capture("logits", logits)
        return logits, new_kv_cache

    def init_cache(
        self, batch_size: int, max_length: int, cache_dtype: Dtypelike
    ) -> LayerKVCache:
        num_layers = self.transformer.depth
        num_kv_heads = self.transformer.layers.attn.num_kv_heads
        head_dim = self.transformer.layers.attn.head_dim

        return (
            jnp.zeros(
                (num_layers, batch_size, max_length, num_kv_heads, head_dim),
                dtype=cache_dtype,
            ),
            jnp.zeros(
                (num_layers, batch_size, max_length, num_kv_heads, head_dim),
                dtype=cache_dtype,
            ),
        )

    @classmethod
    def create(cls, key: jax.Array, config: GemmaConfig):
        embedding = Embedder.create(key, config.vocab_size, config.width)
        transformer = Transformer.create(key, config)
        return cls(embedding, transformer, config.final_logits_softcap, config.dtype)

    @classmethod
    def create_from_pretrained(cls, pretrained_params: dict, config: GemmaConfig):
        import flax
        import numpy as np

        model = jax.eval_shape(
            lambda: cls.create(key=jax.random.PRNGKey(0), config=config)
        )
        new_params = jax.tree.map(lambda x: x, pretrained_params) # Copy
        new_params["embedder"] = new_params["transformer"].pop("embedder")
        new_params = jax.tree.map(jnp.asarray, new_params)
        return nnjax.merge(model, new_params)
