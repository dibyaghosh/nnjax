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

@pytree_dataclass
class LayerNorm:
    weight: Float[Array, "d"]
    bias: Float[Array, "d"]
    eps: float = static_field(default=1e-6)

    def __call__(self, x: Float[Array, "..."]):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(mean2 - jnp.square(mean), 0)
        mul = jax.lax.rsqrt(var + self.eps) * self.weight
        y = (x - mean) * mul + self.bias

        return y.astype(dtype)

@pytree_dataclass
class Linear:
    weight: Float[Array, "d d2"]
    bias: Float[Array, "d2"] | None

    def __call__(self, x: Float[Array, "... d"]):
        y = x @ self.weight.astype(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y


@pytree_dataclass
class CausalSelfAttention:
    c_attn: Linear
    c_proj: Linear
    
        