import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .utils import activate, anonymize, anonymize_shape, concat, default, new_dim, random_name, slice


class ModelParameter(dict):
    def __init__(self, config=None, **config_kwargs):
        super().__init__()

        if isinstance(config, dict):
            config.update(config_kwargs)
        else:
            config = config_kwargs

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
        self.n_head = 8
        self.n_embd = 256
        self.n_layer = 64
        self.buffer_size = 4
        self.shuffle_buffer = 16
        self.interleaved_datasets = 256
        self.token_patch_size = 4
        self.lr = 5e-5
        self.dtype = tf.float32
        self.train_batch_size = 1
        self.mesh_shape = "x:1,y:1,z:1,h:32"
        self.layout = "batch:x,heads:y,embd:z,height:h"
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

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self._layer_idx = 0

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.head_dim = mtf.Dimension("heads", self.n_head)
        self.key_dim = mtf.Dimension("features_per_head", self.n_embd // self.n_head)
        self.feature_dims = [self.head_dim, self.key_dim]
        self.anonymous_key_dim = mtf.Dimension('_' + self.key_dim.name, self.key_dim.size)
        self.intermediate = [mtf.Dimension('_intermediate',
                                           int(np.prod([dim.size for dim in self.feature_dims[1:]])
                                               * self.intermediate_feed_forward_multiplier))]
        self.learned_dim = [new_dim(self.intermediate[1],
                                    self.intermediate[1].size * self.feed_forward_attention_factor)]
        self.vocab_dim = mtf.Dimension("vocab", self.vocab_size)
        self.token_patch_count = self.language_token_per_frame// self.token_patch_size * self.use_language
        self.feature_dim_count = len(self.feature_dims)

    def __getitem__(self, key):
        print(f"Getting {key} via deprecated interface")
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        print(f"Setting {key} via deprecated interface")
        return self.__setattr__(key, value)

    def get(self, key, default):
        print(f"Getting {key} via deprecated interface with default value {default}")
        return self.__dict__.get(key, default)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)

    def dict(self):
        return self.__dict__

    def _get_variable(self, shape, initializer) -> mtf.Tensor:
        return mtf.get_variable(self.mesh, random_name(), list(dict.fromkeys(shape)),
                                dtype=self.dtype, initializer=initializer)

    def _orthogonal_var(self, shape) -> mtf.Tensor:
        return self._get_variable(shape, tf.orthogonal_initializer())

    def _normal_var(self, shape, stddev=0.02) -> mtf.Tensor:
        return self._get_variable(shape, tf.random_normal_initializer(stddev=stddev))

    def _constant_var(self, shape, value) -> mtf.Tensor:
        return self._get_variable(shape, tf.constant_initializer(value))

    def _scalar(self, value) -> mtf.Tensor:
        return self._constant_var([], value)

    def _embed(self, shape: typing.Union[typing.List[mtf.Dimension], mtf.Dimension]):
        return self._normal_var(shape, self.embedding_stddev)

    def _rezero(self, block_input: mtf.Tensor, init) -> mtf.Tensor:
        with tf.variable_scope(random_name()):
            return block_input * self._scalar(init)

    def _linear(self, block_input: mtf.Tensor, old: typing.List[mtf.Dimension],
                new: typing.List[mtf.Dimension]) -> mtf.Tensor:
        with tf.variable_scope(random_name()):
            return mtf.einsum([block_input, self._orthogonal_var(old + new)],
                              list(dict.fromkeys((block_input.shape - old).dims + new)))

    def _linear_to_features(self, block_input: mtf.Tensor,
                            old: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
        return self._linear(block_input, default(old, self.intermediate), self.feature_dims)

    def _linear_from_features(self, block_input: mtf.Tensor,
                              new: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
        return self._linear(block_input, self.feature_dims, default(new, self.intermediate))

    def _block_fn(self, block_input) -> mtf.Tensor:
        self._layer_idx += 1

        attention_dims = (block_input.shape - self.feature_dims)[1:]  # Ex: Shape[Sequence, Width, Height]
        idx = self._layer_idx % len(attention_dims)
        dim = attention_dims[idx]
        tmp_dim = mtf.Dimension(f'_{dim.name}', dim.size)
        attention_scale = (dim.size + self.learned_dim[0].size) ** -0.5

        with tf.variable_scope(random_name()):
            base = activate(self._linear_from_features(block_input))
            context_qry = (self._linear_to_features(base) + self._embed([dim] + self.feature_dims))
            feature_qry = self._linear_to_features(base)
            context_key = anonymize(self._linear_to_features(base), [dim.name])
            context_val = anonymize(self._linear_to_features(base), [dim.name])

            learned_key = self._embed(self.learned_dim + self.feature_dims)
            learned_val = self._embed(self.learned_dim + self.feature_dims)

            context_logits = mtf.einsum([context_qry, context_key], reduced_dims=[self.key_dim]) * attention_scale
            feature_logits = mtf.einsum([feature_qry, learned_key], reduced_dims=[self.key_dim]) * attention_scale

            if idx in self.masked_attention_dimensions:  # it's auto-regressive
                i = mtf.range(self.mesh, tmp_dim, tf.int32)
                j = mtf.range(self.mesh, dim, tf.int32)
                i = mtf.broadcast(i, [tmp_dim, dim])
                j = mtf.broadcast(j, [tmp_dim, dim])
                context_logits += mtf.cast(mtf.less(i, j), self.dtype) * -1e12

            max_logits = mtf.maximum(mtf.reduce_max(mtf.stop_gradient(context_logits), reduced_dim=tmp_dim),
                                     mtf.reduce_max(mtf.stop_gradient(feature_logits), reduced_dim=self.learned_dim[0]))
            context_logits = mtf.exp(context_logits - max_logits)
            feature_logits = mtf.exp(feature_logits - max_logits)
            sumexp = (mtf.reduce_sum(context_logits, reduced_dim=tmp_dim) +
                      mtf.reduce_sum(feature_logits, reduced_dim=self.learned_dim[0]))
            output = (mtf.einsum([context_logits / sumexp, context_val], context_qry.shape) +
                      mtf.einsum([feature_logits / sumexp, learned_val], context_qry.shape))

            return self._rezero(output, 0)

    def build(self,
              vid: typing.Optional[mtf.Tensor],
              txt_src: typing.Optional[mtf.Tensor],
              txt_tgt: typing.Optional[mtf.Tensor]
              ) -> typing.Tuple[mtf.Tensor, typing.Union[int, mtf.Tensor], mtf.Tensor, mtf.Tensor]:
        video_loss: typing.Union[int, mtf.Tensor] = 0
        token_loss: typing.Union[int, mtf.Tensor] = 0

        spatial_ctx = txt_tgt.shape[-self.feature_dim_count] if self.use_language else vid.shape[2]

        if self.use_video:
            context_dimension = vid.shape[1]
            input_features = vid.shape[-1:]
            tgt = slice(vid, 1, context_dimension.size, context_dimension) / 255
            src = slice(vid, 0, context_dimension.size - 1, context_dimension) / self._scalar(127.5) + self._scalar(-1)
            src = self._linear_to_features(src, input_features)

        if self.use_language:
            txt_src = self._linear_to_features(mtf.one_hot(txt_src, self.vocab_dim, dtype=self.dtype), [self.vocab_dim])
            txt_src = self._linear(txt_src, [txt_tgt.shape[-1], self.key_dim], [self.key_dim])

        if self.use_video and self.use_language:
            src = concat([src, txt_src], spatial_ctx)
        elif not self.use_video:
            src: mtf.Tensor = txt_src

        for pad, (name, _) in zip(self.memory_token, src.shape[1:-self.feature_dim_count]):
            if pad > 0:
                anonymous_name = '_' + name
                memory = mtf.broadcast(self._embed([mtf.Dimension(anonymous_name, pad)] + self.feature_dims),
                                       [mtf.Dimension(anonymous_name, pad) if d.name == name else d for d in src.shape])
                src = concat([src, memory], name)

        input_list = (src, None, src, None)

        for _ in range(self.n_layer):
            input_list = mtf.layers.reversible_half_residual_and_swap(*input_list, self._block_fn)

        out = self._rezero(input_list[0], 1) + self._rezero(input_list[2], 1)

        for pad, (name, size) in zip(self.memory_token, src.shape[1:-self.feature_dim_count]):
            out = slice(out, pad, size, name)

        if self.use_language:
            tkn = self._linear_from_features(slice(out, 0, self.token_patch_count, spatial_ctx),
                                             [txt_tgt.shape[-1], self.vocab_dim])
            token_loss: mtf.Tensor = mtf.add_n([mtf.reduce_sum(mtf.square(tkn) * (self.z_loss / self.vocab_size)),
                                                mtf.reduce_sum(mtf.reduce_logsumexp(tkn, self.vocab_dim)),
                                                -mtf.reduce_sum(tkn
                                                                * (mtf.one_hot(txt_tgt, self.vocab_dim, dtype=tkn.dtype)
                                                                   * (1 - self.label_smoothing)
                                                                   + self.label_smoothing / self.vocab_size ** 1))]
                                               ) / (tkn.shape.size / self.vocab_size)

        if self.use_video:
            out = slice(out, self.token_patch_count, out.shape[2].size, spatial_ctx)
            src = mtf.sigmoid(self._linear_from_features(out, input_features))
            video_loss: mtf.Tensor = mtf.reduce_mean(mtf.abs(src - tgt))

        self._layer_idx = 0

        return src, (video_loss + token_loss), video_loss, token_loss
