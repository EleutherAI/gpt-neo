import base64
import mesh_tensorflow as mtf
import numpy as np
import random
import tensorflow.compat.v1 as tf
import typing


def random_name():
    return base64.b64encode(random.getrandbits(256).to_bytes(length=32, byteorder='little')
                            ).decode().replace("+", "").replace("/", "").replace("=", "")


def anonymize(inp: mtf.Tensor, dim_name: str):
    return mtf.rename_dimension(inp, dim_name, '_' + dim_name)


def unanonymize(inp: mtf.Tensor, dim_name: str):
    return mtf.rename_dimension(inp, '_' + dim_name, dim_name)


def activate(block_input):
    return mtf.tanh(block_input) * block_input


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
        self.frame_width = 2048
        self.frame_height = 1152
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
        self.lr = 5e-5
        self.train_batch_size = 1
        self.mesh_shape = "x:1,y:1,z:1,h:32"
        self.layout = "batch:x,heads:y,embd:z,height:h"
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.language_token_per_frame = 0
        self.warmup_steps = 3000
        self.weight_decay = 0.1
        self.train_steps = 572300
        self.warmup_steps = 3000
        self.iterations = 2500
        self.gradient_clipping = 1.0
        self.feed_forward_per_attention = 2
        self.intermediate_feed_forward_multiplier = 1

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self._layer_idx = 0

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.dim_heads = mtf.Dimension("heads", self.n_head)
        self.key_dim = mtf.Dimension("features_per_head", self.n_embd // self.n_head)
        self.feature_dims = [self.dim_heads, self.key_dim]

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

    def _get_variable(self, shape, initializer):
        return mtf.get_variable(self.mesh, random_name(), shape, dtype=tf.float32, initializer=initializer)

    def _get_scalar(self, value):
        return self._get_variable([], tf.constant_initializer(value))

    def _rezero(self, block_input: mtf.Tensor, init: float = 0.):
        with tf.variable_scope(random_name()):
            return block_input * self._get_scalar(init)

    def _linear(self, block_input: mtf.Tensor, old: typing.List[mtf.Dimension], new: typing.List[mtf.Dimension]):
        with tf.variable_scope(random_name()):
            return mtf.einsum([block_input, self._get_variable(old + new, tf.orthogonal_initializer())],
                              block_input.shape - old + new)

    def _intermediate_dimensions(self, dimensions, intermediate_factor: float = 1.):
        return [mtf.Dimension('_intermediate',
                              int(np.prod([dim.size for dim in dimensions])
                                  * intermediate_factor
                                  * self.intermediate_feed_forward_multiplier))]

    def _generic_feed_forward(self,
                              block_input: mtf.Tensor,
                              reduced: typing.List[mtf.Dimension],
                              new: typing.List[mtf.Dimension],
                              intermediate_factor: float = 1.,
                              experts: typing.List[mtf.Dimension] = tuple()):
        intermediate = self._intermediate_dimensions([dim for dim in new if dim not in experts], intermediate_factor)
        with tf.variable_scope(random_name()):
            block_input = self._linear(block_input, reduced, intermediate)
            block_input = activate(block_input)
            return self._linear(block_input, intermediate, new)

    def _feed_forward(self, x: mtf.Tensor, intermediate_factor: float = 1., grouped=False):
        return self._generic_feed_forward(x,
                                          self.feature_dims,
                                          self.feature_dims,
                                          intermediate_factor / (self.intermediate_feed_forward_multiplier
                                                                 if grouped else 1),
                                          [self.dim_heads] if grouped else [])

    def _block_fn(self, block_input):
        self._layer_idx += 1
        attention_dims = (block_input.shape - self.feature_dims)[1:]  # Ex: Shape[Sequence, Width, Height]

        if self._layer_idx % (self.feed_forward_per_attention + 1) < self.feed_forward_per_attention:
            with tf.variable_scope(random_name()):
                return self._rezero(self._feed_forward(block_input, grouped=True))

        idx = (self._layer_idx // (self.feed_forward_per_attention + 1)) % len(attention_dims)
        dim = attention_dims[idx]
        tmp_dim = mtf.Dimension(f'_{dim.name}', dim.size)
        intermediate = self._intermediate_dimensions(self.feature_dims)

        with tf.variable_scope(random_name()):
            base = activate(self._linear(block_input, self.feature_dims, intermediate))
            q = (self._linear(base, intermediate, self.feature_dims)
                 + self._get_variable([dim] + self.feature_dims, tf.orthogonal_initializer()))
            k = anonymize(self._linear(base, intermediate, self.feature_dims), dim.name)
            v = anonymize(self._linear(base, intermediate, self.feature_dims), dim.name)

            logits = mtf.einsum([q, k], reduced_dims=[self.key_dim]) / dim.size ** 0.5
            if idx in self.masked_attention_dimensions:
                i = mtf.range(self.mesh, tmp_dim, tf.int32)
                j = mtf.range(self.mesh, dim, tf.int32)
                i = mtf.broadcast(i, [tmp_dim, dim])
                j = mtf.broadcast(j, [tmp_dim, dim])
                logits += mtf.cast(mtf.less(i, j), tf.float32) * -1e12
            weights = mtf.softmax(logits, dim)
            output = mtf.einsum([weights, v], q.shape)
            return self._rezero(output)

    def build(self, model_input, tkn_src, tkn_tgt):
        vocab_dim = mtf.Dimension("vocab_size", self.vocab_size)

        x = model_input
        if self.use_video:
            context_dimension = x.shape[1]
            input_features = x.shape[-1:]

            tgt = mtf.slice(x, 1, context_dimension.size - 1, context_dimension.name) / 255
            src = (mtf.slice(x, 0, context_dimension.size - 1, context_dimension.name)
                   / self._get_scalar(127.5)
                   + self._get_scalar(-1))

            src = self._linear(src, input_features, self.feature_dims)

        if self.use_language:
            tkn_src = self._linear(mtf.one_hot(tkn_src, vocab_dim, dtype=tf.float32), [vocab_dim], self.feature_dims)

        if self.use_video and self.use_language:
            spatial_ctx = tkn_tgt.shape[-1]
            src = unanonymize(mtf.concat([anonymize(src, spatial_ctx.name), anonymize(tkn_src, spatial_ctx.name)],
                                         '_' + spatial_ctx.name),
                              spatial_ctx.name)
        elif not self.use_video:
            src = tkn_src

        xs = (src, None, src, None)

        for _ in range(self.n_layer):
            xs = mtf.layers.reversible_half_residual_and_swap(*xs, self._block_fn)

     
        video_loss = 0
        token_loss = 0

      

        tkn = out = self._rezero(xs[0], 1) + self._rezero(xs[2], 1)
        loss = 0

        if self.use_video and self.use_language:
            out = anonymize(out, spatial_ctx.name)
            tkn = unanonymize(mtf.slice(out, tgt.shape[2].size, self.language_token_per_frame, '_' + spatial_ctx.name),
                              spatial_ctx.name)
            out = unanonymize(mtf.slice(out, 0, tgt.shape[2].size, '_' + spatial_ctx.name), spatial_ctx.name)

        if self.use_language:
            tkn = self._generic_feed_forward(tkn, self.feature_dims, [vocab_dim])
            token_loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(logits=tkn,
                                                                                targets=tkn_tgt,
                                                                                vocab_dim=vocab_dim,
                                                                                z_loss=1e-4))




        if self.use_video:
            src = mtf.sigmoid(self._generic_feed_forward(out, self.feature_dims, input_features))
            video_loss += mtf.reduce_mean(mtf.abs(src - tgt))

        self._layer_idx = 0

        return src, (video_loss + token_loss), video_loss, token_loss

