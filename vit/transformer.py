# ruff: noqa: F722
from __future__ import annotations
import jax.nn as nn
from jaxtyping import Float, Array, Key, Int
import jax.numpy as jnp
import jax
from nnjax import pytree_dataclass, static_field
import nnjax
import einops
from functools import partial
import dataclasses

Dtypelike = jnp.dtype | str


@dataclasses.dataclass  # just a normal dataclass
class ViTConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    patch_size: int
    dropout: float = 0.0
    image_size: int | None = 224

    @classmethod
    def B16(cls):
        return cls(
            width=768,
            depth=12,
            mlp_dim=3072,
            num_heads=12,
            head_dim=64,
            dropout=0.0,
            patch_size=16,
        )


@pytree_dataclass
class Einsum:
    kernel: jax.Array
    bias: jax.Array | None

    def __call__(self, x, pattern: str):
        x = jnp.einsum(pattern, x, self.kernel)
        if self.bias is not None:
            x = x + self.bias
        return x


def dropout(x, train: bool, rate: float, key: Key | None = None):
    if not train or rate == 0.0:
        return x

    mask = jax.random.bernoulli(key, 1 - rate, x.shape)
    return jnp.where(mask, x / (1 - rate), 0.0)


@pytree_dataclass
class LayerNorm:
    scale: jax.Array
    bias: jax.Array
    eps: float = static_field(default=1e-6)

    def __call__(self, x: Float[Array, "..."]):
        dtype = x.dtype
        x = x.astype(jnp.float32)  # Perform layer norm computations in float32

        mean = jnp.mean(x, axis=-1, keepdims=True)
        y = x - mean
        # Use EX^2 - (EX)^2 to compute variance because it is faster
        # make sure that var >= 0 due to floating point errors
        mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(mean2 - jnp.square(mean), 0)

        mul = jax.lax.rsqrt(var + self.eps) * self.scale
        y = y * mul
        y = y + self.bias
        return y.astype(dtype)

    @classmethod
    def create(cls, _, embedding_dim: int, eps: float = 1e-6):
        scale = jnp.ones(embedding_dim, dtype=jnp.float32)
        bias = jnp.zeros(embedding_dim, dtype=jnp.float32)
        return cls(scale, bias, eps)


@pytree_dataclass
class MlpBlock:
    Dense_0: Einsum
    Dense_1: Einsum
    dropout_rate: float = static_field()

    def __call__(
        self, x: Float[Array, "b l d"], train: bool, key: Key | None = None
    ) -> Float[Array, "b l d"]:
        x = self.Dense_0(x, "bld,dm->blm")
        x = nn.gelu(x)
        x = self.Dense_1(x, "blm,md->bld")
        x = dropout(x, train, self.dropout_rate, key)
        return x

    @classmethod
    def create(
        cls,
        key: jax.Array,
        embedding_dim: int,
        mlp_dim: int,
        dropout_rate: float,
    ):
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.normal(stddev=1e-6)

        key, key_dense_0, key_dense_1 = jax.random.split(key, 3)
        dense_0 = Einsum(
            kernel=kernel_init(key_dense_0, (embedding_dim, mlp_dim)),
            bias=bias_init(key_dense_0, (mlp_dim,)),
        )
        dense_1 = Einsum(
            kernel=kernel_init(key_dense_1, (mlp_dim, embedding_dim)),
            bias=bias_init(key_dense_1, (embedding_dim,)),
        )
        return cls(dense_0, dense_1, dropout_rate)


@pytree_dataclass
class AttentionBlock:
    query: Einsum
    key: Einsum
    value: Einsum
    out: Einsum

    num_heads: int = static_field()
    head_dim: int = static_field()

    def __call__(
        self, x: Float[Array, "b l d"], mask: Float[Array, "b 1 l l"] | None = None
    ) -> Float[Array, "b l d"]:
        dtype = x.dtype
        q = self.query(x, "bld,dnh->blnh")
        k = self.key(x, "bld,dnh->blnh")
        v = self.value(x, "bld,dnh->blnh")

        q = q / jnp.sqrt(self.head_dim).astype(dtype)
        attn_logits = jnp.einsum("bqnh,bknh->bnqk", q, k)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, jnp.finfo(dtype).min)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bnqk, bknh -> bqnh", attn_weights, v)
        out = self.out(out, "bqnh,nhd->bqd")
        return out

    @classmethod
    def create(
        cls, key: jax.Array, *, embedding_dim: int, num_heads: int, head_dim: int
    ):
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.zeros
        key, key_q_proj, key_k_proj, key_v_proj, key_out_proj = jax.random.split(key, 5)
        query = Einsum(
            kernel=kernel_init(key_q_proj, (embedding_dim, num_heads, head_dim)),
            bias=bias_init(key_q_proj, (num_heads, head_dim)),
        )
        key = Einsum(
            kernel=kernel_init(key_k_proj, (embedding_dim, num_heads, head_dim)),
            bias=bias_init(key_k_proj, (num_heads, head_dim)),
        )
        value = Einsum(
            kernel=kernel_init(key_v_proj, (embedding_dim, num_heads, head_dim)),
            bias=bias_init(key_v_proj, (num_heads, head_dim)),
        )
        out = Einsum(
            kernel=kernel_init(key_out_proj, (num_heads, head_dim, embedding_dim)),
            bias=bias_init(key_out_proj, (embedding_dim,)),
        )
        return cls(query, key, value, out, num_heads, head_dim)


@pytree_dataclass
class TransformerLayer:
    ln1: LayerNorm
    attn: AttentionBlock

    ln2: LayerNorm
    mlp: MlpBlock

    dropout_rate: float = static_field()

    def __call__(
        self,
        x: Float[Array, "b l d"],
        mask: Float[Array, "b 1 l l"] | None = None,
        train: bool = False,
        key: Key | None = None,
    ) -> Float[Array, "b l d"]:
        if train:
            keys = jax.random.split(key, 3)
        else:
            keys = [None] * 3

        out_dict = {}
        out = x
        # Attn
        y = self.ln1(out)
        y = out_dict["sa"] = self.attn(y, mask)
        y = dropout(y, train, self.dropout_rate, keys[0])
        out = out_dict["+sa"] = out + y

        # MLP
        y = self.ln2(out)
        y = out_dict["mlp"] = self.mlp(y, train, keys[1])
        y = dropout(y, train, self.dropout_rate, keys[2])
        out = out_dict["+mlp"] = out + y
        return out, out_dict

    @classmethod
    def create(cls, key: jax.Array, config: ViTConfig):
        keys = jax.random.split(key, 4)
        ln1 = LayerNorm.create(keys[0], config.width)
        attn = AttentionBlock.create(
            keys[1],
            embedding_dim=config.width,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
        )
        ln2 = LayerNorm.create(keys[2], config.width)
        mlp = MlpBlock.create(
            keys[3],
            embedding_dim=config.width,
            mlp_dim=config.mlp_dim,
            dropout_rate=config.dropout,
        )
        return cls(ln1, attn, ln2, mlp, dropout_rate=config.dropout)


@pytree_dataclass
class Transformer:
    layers: TransformerLayer
    final_ln: LayerNorm
    num_layers: int = static_field()

    def __call__(
        self,
        x: Float[Array, "b l d"],
        mask: Float[Array, "b 1 l l"] | None = None,
        train: bool = False,
        key: Key | None = None,
    ) -> Float[Array, "b l d"]:
        def body(x, layer_key):
            layer, sub_key = layer_key
            if key is None:
                sub_key = None
            return layer(x, mask, train, key=sub_key)

        if key is None:
            split_keys = jax.random.split(jax.random.PRNGKey(0), self.num_layers)
        else:
            split_keys = jax.random.split(key, self.num_layers)
        x, carry = nnjax.scan(body, x, (self.layers, split_keys))
        return self.final_ln(x), carry

    @classmethod
    def create(cls, key: jax.Array, config: ViTConfig):
        layers = jax.vmap(
            partial(TransformerLayer.create, config=config),
        )(jax.random.split(key, config.depth))
        final_ln = LayerNorm.create(key, config.width)
        return cls(layers, final_ln, config.depth)


@pytree_dataclass
class ImageEmbedding:
    embedding: Einsum
    pos_embedding: jax.Array

    patch_size: int = static_field()

    def __call__(self, images: Int[Array, "b h w c"]) -> Float[Array, "b n l d"]:
        images = images.astype(jnp.float32) / 255.0
        images = images * 2.0 - 1.0

        flattened_patches = einops.rearrange(
            images,
            "b (hh ph) (ww pw) c -> b (hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return self.embedding(flattened_patches, "blp,pd->bld") + self.pos_embedding

    @classmethod
    def create(
        cls,
        key: jax.Array,
        patch_size: int,
        embedding_dim: int,
        image_size: int | None,
    ):
        key_embedding_kernel, key_embedding_bias, key_pos_embedding = jax.random.split(
            key, 3
        )
        embedding = Einsum(
            kernel=nn.initializers.normal(stddev=0.02)(
                key_embedding_kernel, (patch_size * patch_size * 3, embedding_dim)
            ),
            bias=nn.initializers.zeros(key_embedding_bias, (embedding_dim,)),
        )
        pos_embedding = nn.initializers.normal(stddev=0.01)(
            key_pos_embedding,
            (1, (image_size // patch_size) ** 2, embedding_dim),
        )
        return ImageEmbedding(embedding, pos_embedding, patch_size=patch_size)


@pytree_dataclass
class ViT:
    embedding: ImageEmbedding
    transformer: Transformer

    def __call__(
        self,
        images: Int[Array, "b h w c"],
        train: bool = False,
        key: Key | None = None,
    ) -> Float[Array, "b l d"]:
        out = {}
        embeddings = out["with_posemb"] = self.embedding(images)
        x, out["encoder"] = self.transformer(embeddings, train=train, key=key)
        out["encoded"] = x
        return x, out

    @classmethod
    def create(cls, key: jax.Array, config: ViTConfig):
        embedding = ImageEmbedding.create(
            key, config.patch_size, config.width, config.image_size
        )
        transformer = Transformer.create(key, config)
        return cls(embedding, transformer)

    @classmethod
    def create_from_pretrained(cls, pretrained_params: dict, config: ViTConfig):
        embedding_params = pretrained_params["embedding"]
        encoder_params = pretrained_params["Transformer"]["encoderblock"]

        param_dict = {
            "embedding": {
                "embedding": {
                    "kernel": embedding_params["kernel"].reshape(-1, 768),
                    "bias": embedding_params["bias"],
                },
                "pos_embedding": pretrained_params["pos_embedding"],
            },
            "transformer": {
                "layers": {
                    "ln1": encoder_params["LayerNorm_0"],
                    "attn": encoder_params["MultiHeadDotProductAttention_0"],
                    "ln2": encoder_params["LayerNorm_1"],
                    "mlp": encoder_params["MlpBlock_0"],
                },
                "final_ln": pretrained_params["Transformer"]["encoder_norm"],
            },
        }

        abstract_model = jax.eval_shape(
            partial(ViT.create, config=config, key=jax.random.PRNGKey(0))
        )
        return nnjax.merge(abstract_model, param_dict)
