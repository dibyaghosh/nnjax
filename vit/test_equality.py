import numpy as np
import flax
import vit.reference as reference_vit
import vit.transformer as new_vit
import jax


def _load_params(path):
    data = np.load(path)
    params = flax.traverse_util.unflatten_dict(
        {
            k.removeprefix("params/img/"): data[k]
            for k in data.keys()
            if k.startswith("params/img/")
        },
        sep="/",
    )
    return params


@jax.jit
def _run_reference(params, images):
    images = (images / 255.0) * 2 - 1
    model = reference_vit.Model(variant="B/16", pool_type="map", scan=True)
    return model.apply({"params": params}, images, train=False)


@jax.jit
def _run_ours(params, images):
    model = new_vit.ViT.create_from_pretrained(params, new_vit.ViTConfig.B16())
    return model(images, train=False)


def _cossim(a, b, axis=None):
    ab = (a * b).sum(axis=axis)
    a_norm = (a * a).sum(axis=axis)
    b_norm = (b * b).sum(axis=axis)
    return ab * jax.lax.rsqrt(a_norm * b_norm + 1e-8)


def test_equality():
    params = _load_params("/home/dibya/nfs/siglip2_b16_224.npz")
    from PIL import Image

    image = np.asarray(Image.open("cat.jpg").resize((224, 224)))
    images = image[None]
    reference_output = _run_reference(params, images)
    reference_output = reference_output[1]["encoded"]
    our_output, _ = _run_ours(params, images)
    print(_cossim(reference_output, our_output))
    assert _cossim(reference_output, our_output) > 0.99


if __name__ == "__main__":
    test_equality()
