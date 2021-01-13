import base64
import random
import typing

import mesh_tensorflow as mtf

random.seed(65537)


def random_name() -> str:
    """
    Generates a random name based on the globally set seed using python's random module.
    Each name has 256 bits of entropy and a final length of 44 base64 encoded characters.
    For the sake of convenience, special characters are removed from the final string.
    :return: random string
    """
    return base64.b64encode(random.getrandbits(256).to_bytes(length=32, byteorder='little')
                            ).decode().replace("+", "").replace("/", "").replace("=", "")


def dim_name(dim: typing.Union[mtf.Dimension, str]) -> str:
    """
    :param dim: Mesh TensorFlow dimension or name of dimension
    :return: name of dimension
    """
    return dim.name if isinstance(dim, mtf.Dimension) else dim


def check_for_dim(inp: typing.Union[typing.List[mtf.Dimension], mtf.Shape, mtf.Tensor],
                  dim: typing.Union[mtf.Dimension, str]) -> bool:
    """
    Check if a dimension exists in a Mesh TensorFlow tensor, shape or list of dimensions
    :param inp: input to check in
    :param dim: dimension to check for
    :return: true if dimension is found
    """
    return any(dim_name(dim) == d.name for d in (inp.shape if isinstance(inp, mtf.Tensor) else inp))


def deduplicate(inp: typing.Iterable) -> typing.Iterable:
    """
    Remove duplicates from any iterable while retaining the order of elements.
    :param inp: iterable to deduplicate
    :return: new, unique iterable of same type as input
    """
    return type(inp)(dict.fromkeys(list(inp)))


def anonymize(inp: mtf.Tensor,
              dim: typing.Union[typing.List[typing.Union[mtf.Dimension, str]], typing.Union[mtf.Dimension, str]]
              ) -> mtf.Tensor:
    """
    Add an underscore to the name of a dimension of a tensor. This replicates a given dimension of a tensor on all
    devices.
    :param inp: tensor to replicate
    :param dim: dimension(s) to replicate
    :return: replicated tensor
    """
    if not isinstance(dim, list):
        dim = [dim]
    shape = inp.shape.dims
    for d in dim:
        d = unanonymize_dim(d)
        if not check_for_dim(inp, d):
            continue
        shape = [new_dim(dim_name(anonymize_dim(d)), cdim.size) if d == cdim.name else cdim for cdim in shape]
    if shape != inp.shape:
        return mtf.reshape(inp, shape)
    return inp


def anonymize_shape(inp: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                    dim: typing.Union[mtf.Dimension, str]) -> typing.Union[mtf.Shape, typing.List[mtf.Dimension]]:
    """
    Anonymize one dimension of a given Mesh TensorFlow shape. See anonymize for details on what anonymization does.
    :param inp: shape or list of dimensions
    :param dim: dimension to rename
    :return: new shape/list with renamed dimension
    """
    dim = unanonymize_dim(dim)
    if not check_for_dim(inp, dim):
        return inp
    out = [anonymize_dim(dim) if d == dim else d for d in (inp.dims if isinstance(inp, mtf.Shape) else inp)]
    if isinstance(inp, list):
        return out
    return mtf.Shape(out)


def unanonymize(inp: mtf.Tensor, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    """
    Inverse of anonymize. Un-replicates tensor across axis by removing the underscore from the name of a dimension of
    the tensor. This allows mtf to split the tensor across a given dimension again.
    :param inp: tensor to replicate
    :param dim: dimension of tensor
    :return: un-replicated tensor
    """
    dim = anonymize_dim(dim)
    if not check_for_dim(inp, dim):
        return inp
    return mtf.rename_dimension(inp, dim, dim_name(unanonymize_dim(dim)))


def new_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None,
            new_name: typing.Optional[str] = None):
    """
    Create new mesh tensorflow dimension with optional new size and/or new name to replace the old values with.
    :param dim: Dimension or name of dimension
    :param new_size: Optional new size of mtf dimension
    :param new_name: Optinal new name of dimension
    :return: new mtf.Dimension
    """
    name = default(new_name, dim_name(dim))
    if isinstance(dim, mtf.Dimension):
        return mtf.Dimension(name, default(new_size, dim.size))
    if new_size is None:
        return name
    return mtf.Dimension(name, new_size)


def unanonymize_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None):
    """
    Unanonymize mtf.Dimension by removing a leading underscore, if it exists. Optionally, the size can be changed at
    the same time.
    :param dim: mtf.Dimension to unanonymize
    :param new_size: Optional new size
    :return: mtf.Dimension without leading underscore in name
    """
    name = dim_name(dim)
    if name.startswith('_'):
        name = name[1:]
    return new_dim(dim, new_size, name)


def anonymize_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None):
    """
    Anonymize mtf.Dimension by adding a leading underscore, if it does not exist. Optionally, the size can be changed at
    the same time.
    :param dim: mtf.Dimension to anonymize
    :param new_size: Optional new size
    :return: mtf.Dimension with leading underscore in name
    """
    name = dim_name(dim)
    if not name.startswith('_'):
        name = '_' + name
    return new_dim(dim, new_size, name)


def activate(block_input: mtf.Tensor) -> mtf.Tensor:
    """
    Call activation function on mtf.Tensor.
    :param block_input: mtf.Tensor
    :return: activated mtf.Tensor
    """
    return mtf.tanh(block_input) * block_input


def slice(tensor: mtf.Tensor, start: int, end: int, dim: typing.Union[mtf.Dimension, str]):
    """
    Slice across a given (potentially non-anonymous) dimension in mtf.Tensor. This first anonymizes the dimension to
    allow slicing in the first place, next it slices across the dimension and only then it replicates it on all devices
    again.
    Non-Anonymous shapes are not necessary, as the anonymization can skip itself if it isn't necessary.
    :param tensor: mtf.Tensor to slice
    :param start: start of slice
    :param end: end of slice
    :param dim: dimension or name to slice in
    :return: slice of tensor
    """
    dim = dim_name(dim)
    if not start and get_dim(tensor, dim).size == end:
        return tensor
    return unanonymize(mtf.slice(anonymize(tensor, dim), start, end - start, anonymize_dim(dim)), dim)


def get_dim(shape: typing.Union[mtf.Tensor, mtf.Shape, typing.List[mtf.Dimension]],
            dim: typing.Union[mtf.Dimension, str],
            index=False) -> typing.Union[int, mtf.Dimension]:
    """
    Attempts to get a dimension of a tensor. Raises a ValueError if the dimension does not exist.
    :param shape: shape, tensor or list of dimensions to check in
    :param dim: dimension (or name) to check for
    :param index: whether to return the dimension or its index
    :return: index or dimension
    """
    name = dim_name(dim)
    for idx, d in enumerate(shape.shape if isinstance(shape, mtf.Tensor) else shape):
        if d.name == name:
            return idx if index else d
    raise ValueError(f"Dim {dim} with name {name} not found in shape {shape}")


def concat(tensor_list: typing.List[mtf.Tensor], dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    """
    Concatenate across a given (potentially non-anonymous) dimension in mtf.Tensor. This first anonymizes the dimension
    to concat in the first place, next it concats across the dimension and only then it replicates it on all devices
    again.
    Non-Anonymous shapes are not necessary, as the anonymization can skip itself if it isn't necessary.
    :param tensor_list: mtf.Tensor's to concatenate
    :param dim: dimension or name to concatenate in
    :return: concated tensorlist
    """
    dim = dim_name(dim)
    return unanonymize(mtf.concat([anonymize(t, dim) for t in tensor_list], anonymize_dim(dim)), dim)


def default(value: typing.Any, default_value: typing.Any) -> typing.Any:
    """
    Return a default value if a given value is None.
    This is merely a comfort function to avoid typing out "x if x is None else y" over and over again.
    :param value: value that can be None
    :param default_value: default if value is None
    :return: value or default_value
    """
    return default_value if value is None else value
