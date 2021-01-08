import base64
import random
import typing

import mesh_tensorflow as mtf

random.seed(65537)


def random_name() -> str:
    return base64.b64encode(random.getrandbits(256).to_bytes(length=32, byteorder='little')
                            ).decode().replace("+", "").replace("/", "").replace("=", "")


def dim_name(dim: typing.Union[mtf.Dimension, str]) -> str:
    return dim.name if isinstance(dim, mtf.Dimension) else dim


def check_for_dim(inp: mtf.Tensor, dim: typing.Union[mtf.Dimension, str]) -> bool:
    return any(dim_name(dim) == d.name for d in inp.shape)


def anonymize(inp: mtf.Tensor, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    dim = unanonymize_dim(dim)
    if not check_for_dim(inp, dim):
        return inp
    return mtf.rename_dimension(inp, dim, anonymize_dim(dim))


def unanonymize(inp: mtf.Tensor, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    dim = anonymize_dim(dim)
    if not check_for_dim(inp, dim):
        return inp
    return mtf.rename_dimension(inp, dim, unanonymize_dim(dim))


def new_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None,
            new_name: typing.Optional[str] = None):
    name = default(new_name, dim_name(dim))
    if isinstance(dim, mtf.Dimension):
        return mtf.Dimension(name, default(new_size, dim.size))
    if new_size is None:
        return name
    return mtf.Dimension(name, new_size)


def unanonymize_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None):
    name = dim_name(dim)
    if name.startswith('_'):
        name = name[1:]
    return new_dim(dim, new_size, name)


def anonymize_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None):
    name = dim_name(dim)
    if not name.startswith('_'):
        name = '_' + name
    return new_dim(dim, new_size, name)


def activate(block_input: mtf.Tensor) -> mtf.Tensor:
    return mtf.tanh(block_input) * block_input


def slice(tensor: mtf.Tensor, start: int, end: int, dim: typing.Union[mtf.Dimension, str]):
    dim = dim_name(dim)
    if not start and get_dim(tensor, dim).size == end:
        return tensor
    return unanonymize(mtf.slice(anonymize(tensor, dim), start, end - start, anonymize_dim(dim)), dim)


def get_dim(shape: typing.Union[mtf.Tensor, mtf.Shape, typing.List[mtf.Dimension]],
            dim: typing.Union[mtf.Dimension, str],
            index=False) -> typing.Union[int, mtf.Dimension]:
    name = dim_name(dim)
    for idx, d in enumerate(shape.shape if isinstance(shape, mtf.Tensor) else shape):
        if d.name == name:
            return idx if index else d
    raise ValueError(f"Dim {dim} with name {name} not found in shape {shape}")


def concat(tensor_list: typing.List[mtf.Tensor], dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    dim = dim_name(dim)
    return unanonymize(mtf.concat([anonymize(t, dim) for t in tensor_list], anonymize_dim(dim)), dim)


def default(value: typing.Any, default_value: typing.Any) -> typing.Any:
    return default_value if value is None else value
