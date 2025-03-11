from __future__ import annotations

import yaml
from functools import partial
import jax
import orbax.checkpoint as ocp
from etils import epath
import dataclasses
from typing import Any
import contextlib

static_field = partial(dataclasses.field, metadata=dict(static=True))


def dataclass(cls, **kwargs):
    """Equivalent to
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class cls:
    """
    return jax.tree_util.register_dataclass(dataclasses.dataclass(cls, **kwargs))


def asdict(obj, pytree=True):
    if not pytree:
        return dataclasses.asdict(obj)

    if isinstance(obj, list | tuple):
        return obj.__class__(asdict(o, pytree=pytree) for o in obj)
    elif isinstance(obj, dict):
        return obj.__class__({k: asdict(v, pytree=pytree) for k, v in obj.items()})
    elif dataclasses.is_dataclass(obj):
        return {
            field.name: asdict(getattr(obj, field.name), pytree=pytree)
            for field in dataclasses.fields(obj)
            if not field.metadata.get("static", False)
        }
    return obj


def merge(target: Any, updates: Any):
    if isinstance(target, dict):
        return {
            k: merge(target[k], updates[k]) if k in updates else target[k]
            for k in target
        }
    elif isinstance(target, list | tuple):
        return target.__class__(
            merge(t, u) for t, u in zip(target, updates, strict=True)
        )
    elif dataclasses.is_dataclass(target):
        updates = {k: merge(getattr(target, k), updates[k]) for k in updates}
        return dataclasses.replace(target, **updates)
    else:
        return updates


def _path(path: str | epath.Path):
    return epath.Path(path).resolve()


def save(model, path: str | epath.Path):
    path = _path(path)
    path.rmtree(missing_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, model)
    checkpointer.wait_until_finished()
    with (path / "model.yaml").open("w") as f:
        yaml.dump(model, f)


def load_abstract(path: str | epath.Path):
    path = _path(path)
    with (path / "model.yaml").open("r") as f:
        abstract_model = yaml.load(f, Loader=yaml.Loader)
    return abstract_model


def load(path: str | epath.Path, sharding_fn=None):
    path = _path(path)
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = load_abstract(path)
    if sharding_fn is not None:
        sharding = sharding_fn(abstract_model)

        def set_sharding(x, sharding):
            x.sharding = sharding

        jax.tree.map(set_sharding, abstract_model, sharding)
    return checkpointer.restore(path, target=abstract_model)


def yaml_array(dumper, array: jax.Array | jax.ShapeDtypeStruct):
    tag = "tag:yaml.org,2002:python/object/apply:jax.ShapeDtypeStruct"
    mapping = {
        "kwds": {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "sharding": None,
        }
    }
    return dumper.represent_mapping(tag, mapping)


yaml.add_multi_representer(jax.ShapeDtypeStruct, yaml_array)
yaml.add_multi_representer(jax.Array, yaml_array)


_INTERMEDIATES = None


@contextlib.contextmanager
def capture_intermediates(save_to: dict):
    global _INTERMEDIATES
    old_intermediates = _INTERMEDIATES
    try:
        _INTERMEDIATES = save_to
        yield
    finally:
        _INTERMEDIATES = old_intermediates


def scan(fn, *args, **kwargs):
    if _INTERMEDIATES is None:
        return jax.lax.scan(fn, *args, **kwargs)

    def wrapper(carry, xs):
        intermediates = {}
        with capture_intermediates(intermediates):
            carry, out = fn(carry, xs)
        return carry, (out, intermediates)

    carry, (out, intermediates) = jax.lax.scan(wrapper, *args, **kwargs)
    _INTERMEDIATES["scan"] = intermediates
    return carry, out


scan.__doc__ = jax.lax.scan.__doc__


def capture(name, value):
    if _INTERMEDIATES is not None:
        if name not in _INTERMEDIATES:
            _INTERMEDIATES[name] = []
        _INTERMEDIATES[name].append(value)


def repo_path():
    return epath.Path(__file__).parent
