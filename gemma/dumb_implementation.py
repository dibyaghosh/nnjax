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
from .transformer import *

@pytree_dataclass
class Transform:
    w: Array
    b: Array

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x @ self.w + self.b

    @classmethod
    def create(cls, key: jax.Array, in_dim: int, out_dim: int):
        w = trunc_norm_init(in_axis=(0,), out_axis=(1,), batch_axis=())(
            key, (in_dim, out_dim)
        )
        b = jax.random.normal(key, (out_dim,)) * 0.01
        return cls(w=w, b=b)

@pytree_dataclass
class NewGemma():  
    embedders: list[Embedder]
    in_transforms: list[Transform]
    out_transforms: list[Transform]
    transformer: Transformer
    final_logits_softcap: float | None = static_field()
    dtype: Dtypelike = static_field()
  
    @classmethod
    def create(cls, key: jax.Array, config: GemmaConfig):
        def get_key():
            nonlocal key
            key, key_new = jax.random.split(key)
            return key_new
        embedders = [Embedder.create(get_key(), config.vocab_size, config.width) for _ in range(2)]
        in_transforms = [Transform.create(get_key(), config.width, config.width) for _ in range(2)]
        out_transforms = [Transform.create(get_key(), config.width, config.width) for _ in range(2)]
        transformer = Transformer.create(get_key(), config)
        return cls(embedders=embedders, in_transforms=in_transforms, out_transforms=out_transforms, transformer=transformer, final_logits_softcap=config.final_logits_softcap, dtype=config.dtype)


    def __call__(
        self,
        tokens: Int[Array, "b 2 l"],
    ) -> tuple[Float[Array, "b vocab_size"], LayerKVCache]:
        b, _, l = tokens.shape

        embeddings = [embedder.encode(tokens[:, 0]) for embedder in self.embedders]
        embeddings = [in_transform(embedding) for in_transform, embedding in zip(self.in_transforms, embeddings, strict=True)]
        embeddings = (embeddings[0] + embeddings[1]) / 2
        embeddings = embeddings.astype(self.dtype)
        
        
        positions = jnp.broadcast_to(jnp.arange(l).astype(jnp.int32)[None, :], (b, l))

        query_indices = jnp.arange(l)
        key_indices = jnp.arange(l)
        mask = query_indices[:, None] >= key_indices[None, :]
        mask = jnp.broadcast_to(mask[None, None, :, :], (b, 1, len(query_indices), len(key_indices)))

        output_embeddings, new_kv_cache = self.transformer(
            embeddings, positions, mask=mask, kv_cache=None, cache_idx=None
        )

        output_embeddings = [out_transform(output_embeddings) for out_transform in self.out_transforms]
        logits = [embedder.decode(output_embedding) for output_embedding, embedder in zip(output_embeddings, self.embedders, strict=True)]
        nnjax.capture("logits_pre_norm", logits)
        if self.final_logits_softcap is not None:
            logits = [
                jnp.tanh(logit / self.final_logits_softcap) * self.final_logits_softcap
                for logit in logits
            ]
        nnjax.capture("logits", logits)
        return jnp.stack(logits, axis=1), None