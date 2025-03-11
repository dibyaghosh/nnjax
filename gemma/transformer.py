# ruff: noqa: F722
from __future__ import annotations
import jax.nn as nn
from jaxtyping import Float, Array, Key, Int
import jax.numpy as jnp
import jax
from nnjax import dataclass, static_field
import nnjax
import einops
import dataclasses
from functools import partial

Dtypelike = jnp.dtype | str


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


@dataclass
class Einsum:
    kernel: jax.Array

    mm_dtype: Dtypelike = static_field()
    param_dtype: Dtypelike = static_field()

    def __call__(self, x, pattern: str):
        assert x.dtype == 
        x = jnp.einsum(pattern, x, self.kernel)
        if self.bias is not None:
            x = x + self.bias
        return x


@dataclass
class Dropout:
    rate: float = static_field()

    def __call__(self, x, train: bool, key: Key | None = None):
        if not train or self.rate == 0.0:
            return x

        mask = jax.random.bernoulli(key, 1 - self.rate, x.shape)
        return jnp.where(mask, x / (1 - self.rate), 0.0)


@dataclass
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


@dataclass
class MlpBlock:
    Dense_1: Einsum
    Dense_2: Einsum
    dropout: Dropout

    @classmethod
    def create(
        cls,
        key: jax.Array,
        embedding_dim: int,
        mlp_dim: int,
        dropout: float,
        kernel_init: callable,
        bias_init: callable,
    ):
        key, key_dense_1, key_dense_2 = jax.random.split(key, 3)
        dense_1 = Einsum(
            kernel=kernel_init(key_dense_1, (embedding_dim, mlp_dim)),
            bias=bias_init(key_dense_1, (mlp_dim,)),
        )
        dense_2 = Einsum(
            kernel=kernel_init(key_dense_2, (mlp_dim, embedding_dim)),
            bias=bias_init(key_dense_2, (embedding_dim,)),
        )
        dropout = Dropout(rate=dropout)
        return cls(dense_1, dense_2, dropout)

    def __call__(
        self, x: Float[Array, "b l d"], train: bool, key: Key | None = None
    ) -> Float[Array, "b l d"]:
        x = self.Dense_1(x, "bld,dm->blm")
        x = nn.gelu(x)
        x = self.Dense_2(x, "blm,md->bld")
        x = self.dropout(x, train, key)
        return x


@dataclass
class AttentionBlock:
    query: Einsum
    key: Einsum
    value: Einsum
    out: Einsum

    num_heads: int = static_field()
    head_dim: int = static_field()

    @classmethod
    def create(
        cls,
        key: jax.Array,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        kernel_init: callable,
        bias_init: callable,
    ):
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
        nnjax.capture(("attn_logits",), attn_logits)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        nnjax.capture(("attn_weights",), attn_weights)
        out = jnp.einsum("bnqk, bknh -> bqnh", attn_weights, v)
        nnjax.capture(("post_attn",), out)

        out = self.out(out, "bqnh,nhd->bqd")
        return out


@dataclass
class TransformerLayer:
    ln1: LayerNorm
    attn: AttentionBlock
    drop1: Dropout

    ln2: LayerNorm
    mlp: MlpBlock
    drop2: Dropout

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
        y = self.drop1(y, train, keys[0])
        out = out_dict["+sa"] = out + y

        # MLP
        y = self.ln2(out)
        y = out_dict["mlp"] = self.mlp(y, train, keys[1])
        y = self.drop2(y, train, keys[2])
        out = out_dict["+mlp"] = out + y
        return out, out_dict

    @classmethod
    def create(cls, key: jax.Array, config: ViTConfig):
        keys = jax.random.split(key, 4)
        ln1 = LayerNorm.create(keys[0], config.width)
        attn = AttentionBlock.create(
            keys[1],
            config.width,
            config.num_heads,
            config.head_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        drop1 = Dropout(config.dropout)
        ln2 = LayerNorm.create(keys[2], config.width)
        mlp = MlpBlock.create(
            keys[3],
            config.width,
            config.mlp_dim,
            config.dropout,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        drop2 = Dropout(config.dropout)
        return cls(ln1, attn, drop1, ln2, mlp, drop2)


@dataclass
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


@dataclass
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

    def create(
        self,
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


@dataclass
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
        embedding = ImageEmbedding(
            embedding=Einsum(
                kernel=pretrained_params["embedding"]["kernel"].reshape(-1, 768),
                bias=pretrained_params["embedding"]["bias"],
            ),
            pos_embedding=pretrained_params["pos_embedding"],
            patch_size=config.patch_size,
        )
        encoder_params = pretrained_params["Transformer"]["encoderblock"]
        # fmt: off
        ln1 = LayerNorm(
            scale=encoder_params['LayerNorm_0']['scale'],
            bias=encoder_params['LayerNorm_0']['bias'],
        )
        attn_params = encoder_params["MultiHeadDotProductAttention_0"]
        query = Einsum(
            kernel=attn_params["query"]["kernel"],
            bias=attn_params["query"]["bias"]
        )
        key = Einsum(
            kernel=attn_params["key"]["kernel"],
            bias=attn_params["key"]["bias"]
        )
        value = Einsum(
            kernel=attn_params["value"]["kernel"],
            bias=attn_params["value"]["bias"]
        )
        out = Einsum(
            kernel=attn_params["out"]["kernel"],
            bias=attn_params["out"]["bias"],
        )
        attn = AttentionBlock(
            query=query,
            key=key,
            value=value,
            out=out,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
        )
        drop1 = Dropout(config.dropout)

        ln2 = LayerNorm(
            scale=encoder_params['LayerNorm_1']['scale'],
            bias=encoder_params['LayerNorm_1']['bias'],
            eps=1e-6,
        )

        mlp_params = encoder_params["MlpBlock_0"]
        mlp = MlpBlock(
            Dense_1=Einsum(
                kernel=mlp_params["Dense_0"]["kernel"],
                bias=mlp_params["Dense_0"]["bias"]
            ),
            Dense_2=Einsum(
                kernel=mlp_params["Dense_1"]["kernel"],
                bias=mlp_params["Dense_1"]["bias"]
            ),
            dropout=Dropout(config.dropout)
        )
        drop2 = Dropout(config.dropout)
        # fmt: on

        transformer = Transformer(
            layers=TransformerLayer(ln1, attn, drop1, ln2, mlp, drop2),
            final_ln=LayerNorm(
                scale=pretrained_params["Transformer"]["encoder_norm"]["scale"],
                bias=pretrained_params["Transformer"]["encoder_norm"]["bias"],
            ),
            num_layers=config.depth,
        )
        return cls(embedding, transformer)
