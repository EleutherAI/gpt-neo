"""
Contains a class as a datastore for model parameters as well as the necessary functions to build a model graph
from this class
"""
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .utils import (activate, anonymize_dim, concat, deduplicate, default, random_name, slice)


class ModelParameter(typing.Dict[str, typing.Any]):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__()

        self.use_video = True
        self.use_language = True
        self.time_patch = 1
        self.n_ctx = 32
        self.patch_size = 16
        self.frame_width = 320
        self.frame_height = 176
        self.vocab_size = 256
        self.bucket_name = "text-datasets"
        self.color_channels = 3
        self.three_axes = True
        self.prefix = "datasets/video"
        self.dataset_configs = []
        self.data_seed = 456772
        self.n_head = 8
        self.n_embd = 256
        self.n_layer = 64
        self.buffer_size = 4
        self.shuffle_buffer = 16
        self.interleaved_datasets = 256
        self.token_patch_size = 4
        self.learning_reate = 5e-5
        self.dtype = tf.float32
        self.train_batch_size = 1
        self.mesh_shape = "x:1,y:1,h:32"
        self.layout = "batch:x,heads:y,height:h"
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.language_token_per_frame = 0
        self.warmup_steps = 3000
        self.memory_token = [0, 0, 0]
        self.weight_decay = 0.1
        self.train_steps = 572300
        self.warmup_steps = 3000
        self.iterations = 2500
        self.label_smoothing = 0.2
        self.z_loss = 0.1
        self.gradient_clipping = 1.0
        self.intermediate_feed_forward_multiplier = 1
        self.feed_forward_attention_factor = 4
        self.embedding_stddev = 0.02
        self.model_mode = 'jannet'
        self.aux_loss_factor = 0.01

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.language_token_patch = self.language_token_per_frame // self.token_patch_size

        self.head_dim = mtf.Dimension("heads", self.n_head)
        self.head_dimensions = [self.head_dim]
        self.key_dim = mtf.Dimension("features_per_head", self.n_embd // self.n_head)
        self.anonymous_key_dim = anonymize_dim(self.key_dim)

        self.feature_dims = self.head_dimensions + [self.key_dim]
        self.anonymous_feature_dims = self.head_dimensions + [self.anonymous_key_dim]

        self.intermediate = [mtf.Dimension('_intermediate',
                                           int(np.prod([dim.size for dim in self.feature_dims])
                                               * self.intermediate_feed_forward_multiplier))]
        self.learned_dim = [mtf.Dimension("learned", self.intermediate[0].size * self.feed_forward_attention_factor)]

        self.vocab_dim = mtf.Dimension("vocab", self.vocab_size)
        self.token_patch_count = self.language_token_per_frame // self.token_patch_size
        self.feature_dim_count = len(self.feature_dims)
        self.selected_head_dim = mtf.Dimension("top", (int(self.n_head ** 0.5) + 7) // 8 * 8)

        self._auxiliary_loss = 0
        self.layer_idx = 0

    def __getitem__(self, key: str) -> typing.Any:
        print(f"Getting {key} via deprecated interface")
        return self.key

    def __setitem__(self, key: str, value: typing.Any) -> None:
        print(f"Setting {key} via deprecated interface")
        self.key = value

    def get(self, key: str, default: typing.Any) -> typing.Any:
        """
        Default python get from list
        :param key: key to check for in dictionary
        :param default: default value if key doesn't exist
        :return: whatever value belongs to the key or the default
        """
        print(f"Getting {key} via deprecated interface with default value {default}")
        return self.__dict__.get(key, default)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

    def dict(self) -> typing.Dict[str, typing.Any]:
        """
        :return: dictionary containing parameters
        """
        return self.__dict__


def _get_variable(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                  initializer: typing.Callable) -> mtf.Tensor:
    return mtf.get_variable(params.mesh, random_name(), deduplicate(shape), dtype=params.dtype, initializer=initializer)


def _orthogonal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape]) -> mtf.Tensor:
    return _get_variable(params, shape, tf.orthogonal_initializer())


def _normal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                stddev: float = 0.02) -> mtf.Tensor:
    return _get_variable(params, shape, tf.random_normal_initializer(stddev=stddev))


def _constant_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                  value: float) -> mtf.Tensor:
    return _get_variable(params, shape, tf.constant_initializer(value))


def _scalar(params: ModelParameter, value: float) -> mtf.Tensor:
    return _constant_var(params, [], value)


def _embed(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Dimension]):
    return _normal_var(params, shape, params.embedding_stddev)


def _rezero(params, block_input: mtf.Tensor, init: float) -> mtf.Tensor:
    with tf.variable_scope(random_name()):
        return block_input * _scalar(params, init)


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


def _block_fn(params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    params.layer_idx += 1

    attention_dims = (block_input.shape - params.feature_dims)[1:]  # Ex: Shape[Sequence, Width, Height]
    idx = params.layer_idx % len(attention_dims)
    dim = attention_dims[idx]
    tmp = anonymize_dim(dim)
    attention_scale = (dim.size + params.learned_dim[0].size) ** -0.5

    with tf.variable_scope(random_name()):
        base = activate(_linear_from_features(params, block_input))

        ctx_qry = _linear_to_features(params, base) + _embed(params, [dim] + params.feature_dims)

        ctx_lgt = mtf.einsum([ctx_qry,
                              mtf.einsum([_linear_to_features(params, base),
                                          _orthogonal_var(params, [dim, tmp])],
                                         ctx_qry.shape - dim + tmp)],
                             reduced_dims=[params.key_dim]) * attention_scale
        ftr_lgt = mtf.einsum([_linear_to_features(params, base),
                              _embed(params, params.learned_dim + params.feature_dims) * attention_scale],
                             reduced_dims=[params.key_dim])

        if idx in params.masked_attention_dimensions:  # it's auto-regressive
            ctx_lgt += mtf.cast(mtf.less(mtf.broadcast(mtf.range(params.mesh, tmp, tf.int32), [tmp, dim]),
                                         mtf.broadcast(mtf.range(params.mesh, dim, tf.int32), [tmp, dim])),
                                params.dtype) * -1e12

        max_lgt = mtf.maximum(mtf.reduce_max(mtf.stop_gradient(ctx_lgt), reduced_dim=tmp),
                              mtf.reduce_max(mtf.stop_gradient(ftr_lgt), reduced_dim=params.learned_dim[0]))
        ctx_lgt = mtf.exp(ctx_lgt - max_lgt)
        ftr_lgt = mtf.exp(ftr_lgt - max_lgt)
        output = (mtf.einsum([ctx_lgt, mtf.einsum([_linear_to_features(params, base),
                                                   _orthogonal_var(params, [dim, tmp])],
                                                  ctx_qry.shape - dim + tmp)], ctx_qry.shape) +
                  mtf.einsum([ftr_lgt, _embed(params, params.learned_dim + params.feature_dims)], ctx_qry.shape)
                  ) / (mtf.reduce_sum(ctx_lgt, reduced_dim=tmp) +
                       mtf.reduce_sum(ftr_lgt, reduced_dim=params.learned_dim[0]))

        return _rezero(params, output, 0)


def build(params: ModelParameter,
          vid: typing.Optional[mtf.Tensor],
          txt_src: typing.Optional[mtf.Tensor],
          txt_tgt: typing.Optional[mtf.Tensor],
          vid_msk: typing.Optional[mtf.Tensor],
          tkn_msk: typing.Optional[mtf.Tensor],
          ) -> typing.Tuple[mtf.Tensor, typing.Union[int, mtf.Tensor], mtf.Tensor, mtf.Tensor]:
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
    tkn_msk = mtf.ones(params.mesh, [], tf.float32) if tkn_msk is None else mtf.cast(tkn_msk, tf.float32)
    vid_msk = mtf.ones(params.mesh, [], tf.float32) if vid_msk is None else mtf.cast(vid_msk, tf.float32)

    spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

    if params.use_video:
        context_dimension = vid.shape[1]
        input_features = vid.shape[-1:]
        tgt = slice(vid, 1, context_dimension.size, context_dimension) / 255
        src = slice(vid, 0, context_dimension.size - 1, context_dimension) / _scalar(params, 127.5)
        src += _scalar(params, -1)
        src = _linear_to_features(params, src, input_features)

    if params.use_language:
        txt_src = _linear_to_features(params, mtf.one_hot(txt_src, params.vocab_dim, dtype=params.dtype),
                                      [params.vocab_dim])
        txt_src = _linear(params, txt_src, [txt_tgt.shape[-1], params.key_dim], [params.key_dim])

    if params.use_video and params.use_language:
        src = concat([src, txt_src], spatial_ctx)
    elif not params.use_video:
        src: mtf.Tensor = txt_src

    for pad, (name, _) in zip(params.memory_token, src.shape[1:-params.feature_dim_count]):
        if pad > 0:
            anonymous_dim = anonymize_dim(name, pad)
            memory = mtf.broadcast(_embed(params, [anonymous_dim] + params.feature_dims),
                                   [anonymous_dim if d.name == name else d for d in src.shape])
            src = concat([src, memory], name)

    input_list = (src, None, src, None)

    for _ in range(params.n_layer):
        input_list = mtf.layers.reversible_half_residual_and_swap(*input_list, lambda x: _block_fn(params, x))

    out = _rezero(params, input_list[0], 1) + _rezero(params, input_list[2], 1)

    for pad, (name, size) in zip(params.memory_token, src.shape[1:-params.feature_dim_count]):
        out = slice(out, pad, size, name)

    if params.use_language:
        tkn = _linear_from_features(params, slice(out, 0, params.language_token_patch, spatial_ctx),
                                    [txt_tgt.shape[-1], params.vocab_dim])
        z_loss = mtf.reduce_sum(mtf.square(tkn)) * (params.z_loss / params.vocab_size)
        logsumexp = mtf.reduce_sum(mtf.reduce_logsumexp(tkn, params.vocab_dim) * tkn_msk)
        tkn_loss = mtf.reduce_sum(tkn * tkn_msk
                                  * (params.label_smoothing / params.vocab_size / (1 - params.label_smoothing)
                                     + mtf.one_hot(txt_tgt, params.vocab_dim, dtype=tkn.dtype)))
        tkn_loss *= 1 - params.label_smoothing
        token_loss: mtf.Tensor = mtf.add_n([z_loss, logsumexp, -tkn_loss]) / (tkn.size / params.vocab_size)

    if params.use_video:
        out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)
        src = mtf.sigmoid(_linear_from_features(params, out, input_features))
        video_loss: mtf.Tensor = mtf.reduce_mean(mtf.abs(src - tgt) * vid_msk)

    params.layer_idx = 0

    return (src,
            (video_loss + token_loss),
            video_loss * vid_msk.size / mtf.reduce_sum(vid_msk),
            token_loss * tkn_msk.size / mtf.reduce_sum(tkn_msk))
