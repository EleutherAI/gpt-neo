import random
import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


class ModelParameter(dict):
    def __init__(self, config=None, **config_kwargs):
        super().__init__()

        self._get_count = {}
        self._set_count = {}

        if isinstance(config, dict):
            config.update(config_kwargs)
        else:
            config = config_kwargs

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
        self.buffer_size = 8
        self.lr = 5e-5
        self.train_batch_size = 1
        self.recompute_grad = True
        self.mesh_shape = "x:1,y:1,z:1,h:32"
        self.layout = "batch:x,heads:y,embd:z,height:h"
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.dataset_ids = ["openwebtext-documents"]
        self.local_attention_radius = 256
        self.n_vocab = 256
        self.language_token_per_frame = 0
        self.embed_dropout = 0
        self.lr_decay = "cosine"
        self.warmup_steps = 3000
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.epsilon = 1e-8
        self.ada_epsilon1 = 1e-30
        self.ada_epsilon2 = 1e-3
        self.opt_name = "adam"
        self.weight_decay = 0.1
        self.attn_dropout = 0
        self.train_steps = 572300
        self.eval_steps = 0
        self.predict_steps = 1
        self.res_dropout = 0
        self.warmup_steps = 3000
        self.eval_batch_size = 64
        self.steps_per_checkpoint = 2500
        self.predict_batch_size = 1
        self.iterations = 2500
        self.datasets = []
        self.shuffle = False
        self.residual = True
        self.scale_by_depth = True
        self.scale_by_in = False
        self.eval = False
        self.activation_function = "gelu"
        self.gradient_clipping = 1.0
        self.dropout_rate = 0.

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self._layer_idx = 0

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.the_batch_size = self.eval_batch_size if self.eval else self.train_batch_size

        if not self.three_axes:
            self.frame_height_patch = self.frame_height_patch * self.frame_width_patch

    @property
    def dim_heads(self):
        return mtf.Dimension("heads", self.n_head)

    @property
    def key_dim(self):
        return mtf.Dimension("features_per_head", self.n_embd // self.n_head)

    def __getitem__(self, key):
        print(f"Getting {key} via deprecated interface")
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        print(f"Setting {key} via deprecated interface")
        return self.__setattr__(key, value)

    def get(self, key, default):
        print(f"Getting {key} via deprecated interface with default value {default}")
        self._get_count[key] = self._get_count.get(key, 0) + 1
        return self.__dict__.get(key, default)

    def __setattr__(self, key, value):
        if value == {} and key in ('_set_count', '_get_count'):
            super().__setattr__(key, value)
            return
        self.__dict__[key] = value
        self._set_count[key] = self._set_count.get(key, 0) + 1

    def __getattr__(self, key):
        self.__getattribute__('_get_count')[key] = self.__getattribute__('_get_count').get(key, 0) + 1
        return self.__getattribute__('__dict__')[key]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)

    def dict(self):
        return self.__dict__

    def _get_variable(self, shape, initializer):
        return mtf.get_variable(self.mesh, f"{random.getrandbits(64):x}", shape, dtype=tf.float32,
                                initializer=initializer)

    def _rezero(self, block_input: tf.Tensor):
        with tf.variable_scope(f'rezero_{random.getrandbits(64):x}'):
            block_input = block_input * self._get_variable([], tf.constant_initializer(0))
        return block_input

    def _generic_feed_forward(self,
                              block_input: mtf.Tensor,
                              reduced_dims: typing.List[mtf.Dimension],
                              new_dimensions: typing.List[mtf.Dimension]):
        intermediate_dimensions = [mtf.Dimension('_' + dim.name, dim.size) for dim in new_dimensions]
        with tf.variable_scope(f'feed_forward_{random.getrandbits(64):x}'):
            weight0 = self._get_variable(reduced_dims + intermediate_dimensions, tf.orthogonal_initializer())
            block_input = mtf.einsum([block_input, weight0],
                                     block_input.shape - reduced_dims + intermediate_dimensions)
            if self.dropout_rate > 0:
                block_input = mtf.dropout(block_input, 1 - self.dropout_rate)
            block_input = block_input * mtf.tanh(block_input)  # LiSHT: https://arxiv.org/abs/1901.05894
            weight1 = self._get_variable(intermediate_dimensions + new_dimensions, tf.orthogonal_initializer())
            block_input = mtf.einsum([block_input, weight1],
                                     block_input.shape - intermediate_dimensions + new_dimensions)
        return block_input

    def _feed_forward(self, x):
        return self._generic_feed_forward(x, [self.dim_heads, self.key_dim], [self.dim_heads, self.key_dim])

    def _block_fn(self, block_input):
        self._layer_idx += 1
        middle_dimensions = block_input.shape[1:-1]  # Ex: Shape[Sequence, Width, Height]
        
        with tf.variable_scope(f"attention_block{self._layer_idx}"):
            outputs = []
            for idx, dim in enumerate(middle_dimensions):
                tmp_dim = mtf.Dimension(f'anonymous_{dim.name}', dim.size)

                q = self._feed_forward(block_input)
                k = self._feed_forward(block_input)
                v = self._feed_forward(block_input)
                k = mtf.rename_dimension(k, dim.name, tmp_dim.name)
                v = mtf.rename_dimension(v, dim.name, tmp_dim.name)

                logits = mtf.einsum([q, k], q.shape - self.key_dim + tmp_dim) / tmp_dim.size ** 0.5
                if idx in self.masked_attention_dimensions:
                    i = mtf.range(self.mesh, tmp_dim, tf.int32) + dim.size - tmp_dim.size
                    j = mtf.range(self.mesh, dim, tf.int32)
                    i = mtf.broadcast(i, [tmp_dim, dim])
                    j = mtf.broadcast(j, [tmp_dim, dim])
                    bias = mtf.cast(mtf.less(i, j), tf.float32) * -1e12
                    logits += mtf.broadcast(bias, logits.shape)
                weights = mtf.softmax(logits, dim)
                a = mtf.einsum([weights, v], q.shape)

                a += self._rezero(self._feed_forward(a))
                outputs.append(a)
            block_output = mtf.add_n([self._rezero(a) for a in outputs])
            block_output += self._rezero(self._feed_forward(mtf.add_n(outputs)))

        return block_output

    def build(self, model_input, token_x_input, token_y_input):
        # TODO: Add support for missing model_input, token_x/y_input
        # TODO: Rename token_x_input
        # TODO: General cleanup
        x = model_input / 255.
        context_dimension = x.shape[1]

        tgt = mtf.slice(x, 1, context_dimension.size - 1, context_dimension.name)
        src = mtf.slice(x, 0, context_dimension.size - 1, context_dimension.name)
        vocab_dim = token_x_input.shape[-1]
        embedding = mtf.get_variable(x.mesh, "embedding", mtf.Shape([vocab_dim, self.dim_heads, self.key_dim]), dtype=tf.float32, initializer=tf.random_normal_initializer())
        token_x_input = einsum([one_hot(token_x_input, vocab_dim, dtype=tf.float32), embedding], reduced_dims=[vocab_dim], output_shape=token_x_input - vocab_dim + [self.dim_heads, self.key_dim])
        
        src_embedding = mtf.add_n([self._get_variable([dim, src.shape[-1]], tf.random_normal_initializer())
                                   for dim in src.shape[1:-1]]  # Ex: Shape[Sequence, Width, Height]
                                  )
        tkn_embedding = mtf.add_n([self._get_variable([dim, token_x_input.shape[-1]], tf.random_normal_initializer())
                                   for dim in src.shape[1:-1]]  # Ex: Shape[Sequence, Width, Height]
                                  )
        src += src_embedding
        token_x_input += tkn_embedding

        src = mtf.concat([src, token_x_input], token_x_input.shape[-2].name)
        
        input_features = [src.shape[-1]]

        xs = (self._generic_feed_forward(src, input_features, [self.dim_heads, self.key_dim]), None,
              self._generic_feed_forward(src, input_features, [self.dim_heads, self.key_dim]), None)

        for layer in range(self.n_layer):
            xs = mtf.layers.reversible_half_residual_and_swap(*xs, self._block_fn)
        output = self._generic_feed_forward(xs[0] + xs[2], [self.dim_heads, self.key_dim], input_features)

        with tf.variable_scope("reduce_mean_final"):
            tkn = mtf.slice(output, tgt.shape[-2].size, token_x_input.shape[-2].size, output.shape[-2].name)
            src = mtf.slice(output, 0, tgt.shape[-2].size, output.shape[-2].name)
            vid_loss = mtf.reduce_mean(mtf.abs(output - tgt))
            tkn_loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(logits=tkn, targets=token_y_input, vocab_dim=token_y_input.shape[-1], z_loss=1e-4))

        self._layer_idx = 0

        return output, vid_loss + tkn_loss

    def attribute_accesses(self):
        return {'GET':    self._get_count,
                'SET':    self._set_count,
                'unread': [k for k, v in {**self.__dict__, **self._set_count}.items() if k not in self._get_count]
                }
