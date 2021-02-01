"""
Contains all necessary functions to build a model graph
TODO(Lucas): Write docstrings for all functions
"""

import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from .dataclass import BlockConfig, ModelParameter
from .utils_core import default
from .utils_mtf import activate, anonymize, anonymize_dim, anonymize_shape, concat, deduplicate, random_name, slice


def _get_attention_dim(params: ModelParameter, block_input: typing.Union[mtf.Tensor, mtf.Shape]
                       ) -> typing.Tuple[int, mtf.Dimension]:
    if isinstance(block_input, mtf.Tensor):
        block_input = block_input.shape
    attention_dims = (block_input - params.feature_dims - params.intermediate)[1:]  # Ex: Shape[Sequence, Width, Height]
    idx = params.attention_idx % len(attention_dims)
    dim = attention_dims[idx]
    return idx, dim


def _get_variable(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                  initializer: typing.Callable) -> mtf.Tensor:
    return mtf.get_variable(params.mesh, random_name(), deduplicate(shape), dtype=params.dtype, initializer=initializer)


def _orthogonal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape]) -> mtf.Tensor:
    return _get_variable(params, shape, tf.random_normal_initializer(stddev=0.02))


def _normal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                stddev: float = 0.02, mean: float = 0.) -> mtf.Tensor:
    return _get_variable(params, shape, tf.random_normal_initializer(stddev=stddev, mean=mean))


def _constant_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                  value: float) -> mtf.Tensor:
    return _get_variable(params, shape, tf.constant_initializer(value))


def _scalar(params: ModelParameter, value: float) -> mtf.Tensor:
    return _constant_var(params, [], value)


def _attention(params: ModelParameter, base: mtf.Tensor, key: mtf.Tensor):
    idx, dim = _get_attention_dim(params, base)
    params.attention_idx += 1
    tmp = anonymize_dim(dim)
    qry = _linear_to_features(params, base)
    val = anonymize(_linear_to_features(params, base), dim)

    lgt = mtf.einsum([qry, key], reduced_dims=[params.key_dim])

    if idx in params.masked_attention_dimensions:  # it's auto-regressive
        lgt += mtf.cast(mtf.less(mtf.broadcast(mtf.range(params.mesh, tmp, tf.int32), [tmp, dim]),
                                 mtf.broadcast(mtf.range(params.mesh, dim, tf.int32), [tmp, dim])),
                        params.dtype) * -1e12

    lgt = mtf.exp(lgt - mtf.reduce_max(mtf.stop_gradient(lgt), reduced_dim=tmp))
    out = mtf.einsum([lgt, val], qry.shape) / mtf.reduce_sum(lgt, reduced_dim=tmp)
    return out


def _embed(params: ModelParameter, block_input: typing.Union[mtf.Tensor, mtf.Shape]) -> mtf.Tensor:
    idx, dim = _get_attention_dim(params, block_input)
    return _normal_var(params, [dim] + params.feature_dims, params.embedding_stddev)


def _linear(params: ModelParameter, block_input: mtf.Tensor, old: typing.List[mtf.Dimension],
            new: typing.List[mtf.Dimension]) -> mtf.Tensor:
    with tf.variable_scope(random_name()):
        return mtf.einsum([block_input, _orthogonal_var(params, old + new)],
                          deduplicate((block_input.shape - old).dims + new))


def _linear_to_features(params: ModelParameter, block_input: mtf.Tensor,
                        old: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, default(old, params.intermediate), params.feature_dims)


def _linear_from_features(params: ModelParameter, block_input: mtf.Tensor,
                          new: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, params.feature_dims, default(new, params.intermediate))


def _rezero(params, block_input: mtf.Tensor, init: float = 0.) -> mtf.Tensor:
    with tf.variable_scope(random_name()):
        return block_input * _scalar(params, init)


def _feed_forward(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    return _linear_to_features(params, activate(_linear_from_features(params, block_input)))


def _group_feed_forward(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    intermediate = [params.head_dim, anonymize_dim(params.key_dim, params.key_dim.size * params.group_linear_factor)]
    return _linear_to_features(params, activate(_linear_from_features(params, block_input, intermediate)), intermediate)


def _context_attention(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    _, dim = _get_attention_dim(params, block_input)
    base = activate(_linear_from_features(params, block_input))

    return _attention(params, base, anonymize(_linear_to_features(params, base) * dim.size ** -0.5, dim))


def _positional_attention(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    _, dim = _get_attention_dim(params, block_input)
    base = activate(_linear_from_features(params, block_input))

    return _attention(params, base, _embed(params, anonymize_shape(base.shape, dim)))

  
def _embedded_attention(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    _, dim = _get_attention_dim(params, block_input)
    base = activate(_linear_from_features(params, block_input))

    return _attention(params, base, anonymize(_linear_to_features(params, base) * dim.size ** -0.5 + _embed(params, base.shape), dim))

  

def _group_normalize(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    _ = params
    feat = block_input.shape[-1]
    block_input -= mtf.reduce_mean(block_input, reduced_dim=feat)
    block_input *= mtf.rsqrt(1e-6 + mtf.reduce_mean(mtf.square(block_input), reduced_dim=feat))
    block_input *= _normal_var(params, [feat], mean=1)
    block_input += _normal_var(params, [feat], mean=0)
    return block_input


def _normalize(params: ModelParameter, x: mtf.Tensor) -> mtf.Tensor:
    feat = x.shape[-2:]
    shape = x.shape - feat
    g = mtf.get_variable(x.mesh, "g", feat, initializer=tf.constant_initializer(1), dtype=params.dtype)
    b = mtf.get_variable(x.mesh, "b", feat, initializer=tf.constant_initializer(0), dtype=params.dtype)
    x -= mtf.reduce_mean(x, output_shape=shape)
    return x * mtf.rsqrt(1e-6 + mtf.reduce_mean(mtf.square(x), output_shape=shape)) * g + b


LAYER_FUNCTIONS = {'feed_forward':         _feed_forward,
                   'group_feed_forward':   _group_feed_forward,
                   'context_attention':    _context_attention,
                   'positional_attention': _positional_attention,
                   'embedded_attention':   _embedded_attention,
                   'group_normalize':      _group_normalize,
                   'normalize':            _normalize,
                   'rezero':               _rezero,
                   'embed':                _embed
                   }


def _block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor) -> mtf.Tensor:
    out = block_input
    for layer in block_part_config.layer:
        out = LAYER_FUNCTIONS[layer](params, out)

    if not params.use_revnet:
        if block_part_config.skip:
            out += block_input

    return out


def build(params: ModelParameter,
          vid: typing.Optional[mtf.Tensor],
          txt_src: typing.Optional[mtf.Tensor],
          txt_tgt: typing.Optional[mtf.Tensor],
          vid_msk: typing.Optional[mtf.Tensor],
          tkn_msk: typing.Optional[mtf.Tensor],
          ) -> typing.Tuple[mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor]:
    """
    Build Mesh Tensorflow graph of a model given parameters previously inserted.
    The model slices the video input itself (to save on TPU CPU <--> TPU Core bandwidth), but needs both
    text source and text target.
    :param params: Instance of ModelParameter for which to build the graph
    :param vid: Optional Video to attend over, length=(context+1)
    :param txt_src: Optional tokenized text source, will be embedded
    :param txt_tgt: Optional tokenized text target, required when source is given
    :param vid_msk: Optional mask to remove loss for certain video frames
    :param tkn_msk: Optional mask to remove loss for certain token positions
    :return: (Generated Video, Total Loss, Video Loss, Token Loss)
    """
    video_loss: typing.Union[int, mtf.Tensor] = 0
    token_loss: typing.Union[int, mtf.Tensor] = 0
    frame_out: typing.Union[int, mtf.Tensor] = 0
    token_out: typing.Union[int, mtf.Tensor] = 0

    spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

    # Slice and Normalize the Video input add a zero frame memory token.
    if params.use_video:
        context_dimension = vid.shape[1]
        input_features = vid.shape[-1:]
        tgt = slice(vid, 1, context_dimension.size, context_dimension)
        src = slice(vid, 0, context_dimension.size - 1, context_dimension)
        src = src * vid_msk + _normal_var(params, shape=vid.shape[2:]) * (1 - vid_msk)
        src = _linear_to_features(params, src, input_features)

    # Language embedding and initial feed forward.
    if params.use_language:
        txt_src = _linear_to_features(params, mtf.one_hot(txt_src, params.vocab_dim, dtype=params.dtype),
                                      [params.vocab_dim])
        txt_src = _linear(params, txt_src, [txt_tgt.shape[-1], params.key_dim], [params.key_dim])

    # Connect video and language Input.
    if params.use_video and params.use_language:
        src = concat([src, txt_src], spatial_ctx)

    # If language only mode, set the language input as src.
    elif not params.use_video:
        src: mtf.Tensor = txt_src

    if params.use_revnet:
        out = (src, None, src, None)

        def _layer_builder(block_input: typing.Tuple[mtf.Tensor], block_config: BlockConfig):
            return mtf.layers.reversible_half_residual_and_swap(*block_input,
                                                                lambda x: _block_part_fn(params, block_config, x))
    else:
        out = src

        def _layer_builder(block_input: mtf.Tensor, block_config: BlockConfig):
            return mtf.recompute_grad(lambda x: _block_part_fn(params, block_config, x), [block_input])

    for _ in range(params.n_blocks):
        for block_part in params.block_config:
            out = _layer_builder(out, block_part)

    if params.use_revnet:
        out = out[0] + out[2]

    # Language Loss
    if params.use_language:
        token_out = _linear_from_features(params, slice(out, 0, params.language_token_patch, spatial_ctx),
                                          [txt_tgt.shape[-1], params.vocab_dim])
        cross_entropy = mtf.layers.softmax_cross_entropy_with_logits(token_out, txt_tgt, params.vocab_dim,
                                                                     params.z_loss)
        token_loss = mtf.reduce_mean(tkn_msk * cross_entropy)

    # Video Loss
    if params.use_video:
        out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)
        frame_out = mtf.sigmoid(_linear_from_features(params, out, input_features))
        video_loss: mtf.Tensor = mtf.reduce_mean(mtf.abs(frame_out - tgt) * vid_msk)

    params.layer_idx = 0

    return video_loss, token_loss, frame_out, token_out
