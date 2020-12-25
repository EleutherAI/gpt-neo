import random
import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


class EinsumOperation(mtf.Operation):
    """Einstein summation (matmul, etc).
    The equation follows the dimensions in the input and output shapes.
    Every dimension must occur in at least two of the input/output Tensors.
    i.e. no new dimensions in the output, and no reduction of dimensions that
    occur in only one input.
    """

    def __init__(self, inputs, output_shape, equation, name=None):
        super(EinsumOperation, self).__init__(inputs, name=name or "einsum")
        if not inputs:
            raise ValueError("Einsum needs at least one input")
        for x in inputs:
            if x.dtype != inputs[0].dtype:
                raise ValueError("Input dtypes must be equal got %s"
                                 % ([y.dtype for y in inputs],))
        self._equation = equation

        *equation, output = equation.split(',')
        output = output.split('->')
        self._equation_inputs = list(equation) + [output[0]]
        self._equation_output = output[1]
        self._outputs = [mtf.Tensor(self, output_shape, inputs[0].dtype)]

    def gradient(self, grad_ys):
        dy = grad_ys[0]
        xs = self.inputs
        return [einsum([dy] + [xs[j] for j in range(len(xs)) if j != i],
                       ','.join([self._equation_output] +
                                [self._equation_inputs[j] for j in range(len(self._equation_inputs)) if j != i]) +
                       '->' +
                       self._equation_inputs[i],
                       xs[i].shape)
                for i in range(len(self.inputs))]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        input_shape = mtf.Shape(list(set(sum([x.shape.dims for x in self.inputs], []))))
        output_shape = self.outputs[0].shape

        intersection_shape = mtf.Shape([d for d in output_shape.dims if d in input_shape.dims])

        reduced_mesh_axes = [d for i, d in enumerate(input_shape.dims)
                             if d not in output_shape.dims and mesh_impl.tensor_layout(input_shape)[i] is not None]

        # call tf.einsum
        y = mesh_impl.slicewise(lambda *slices: mesh_impl.einsum(self._equation, *slices),
                                *[lowering.tensors[x] for x in self.inputs])
        if reduced_mesh_axes:
            y = mtf.LazyAllreduceSum(mesh_impl, y, reduced_mesh_axes,
                                     lambda: lowering.add_counter(f"allreduce/{reduced_mesh_axes}/einsum_op",
                                                                  mesh_impl.laid_out_size(intersection_shape)))
        # broadcast from intersection_shape to output_shape
        if intersection_shape != output_shape:
            y = mesh_impl.broadcast_impl(y, intersection_shape, output_shape)

        lowering.set_tensor_lowering(self.outputs[0], y)
        lowering.add_counter("einsum", mesh_impl.laid_out_size(input_shape))
        lowering.add_counter("einsum_unique", input_shape.size)


def einsum(xs, equation, output_shape):
    """Einstein summation.
    einsum(xs, output_shape) is equivalent to broadcasting all inputs
    to the union of all of their shapes, multiplying them componentwise,
    and finally reduce_summing down to output_shape.
    One common case of this is matrix multiplication:
        x has shape [a, b]
        y has shape [b, c]
        matmul(x, y) == einsum([x, y], output_shape=[a, c])
    We provide a few options for specifying the output shape:
    If neither output_shape nor reduced_dims is specified, then the output
    shape is set to the contain all dimensions that appear exactly once in the
    inputs, in order of appearance.
    If output_shape is not specified, then the output shape is set to the contain
    all dimensions that appear in xs but not in reduced_dims, in the order
    that they appear in xs.  If reduced_dims is also not specified, then
    reduced_dims is set to the set of all dimensions that appear at least twice in
    xs.
    If both output_shape and reduced_dims are specified, then we check that
    reduced_dims matches the set of dimensions present in xs but not in
    output_shape, and throw an exception if it does not.  This helps to reduce
    bugs.
    Args:
      xs: a list of Tensors
      equation: einsum equation
      output_shape: resulting einsum shape
    Returns:
      a Tensor
    Raises:
      ValueError: if reduced_dims contradicts output_shape
    """
    return EinsumOperation(xs, mtf.convert_to_shape(output_shape), equation, name=None).outputs[0]


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
        self.token_embedding = False
        self.feed_forward_per_attention = 2

        self.mesh = None

        self.masked_attention_dimensions = [0]

        self._layer_idx = 0

        self._space_dims = f'sx{"y" if self.three_axes else ""}'
        self._base_dims = 'b' + self._space_dims
        self._input_dims = self._base_dims + 'hf'

        self.__dict__.update(config)

        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.the_batch_size = self.eval_batch_size if self.eval else self.train_batch_size

    @property
    def dim_heads(self):
        return mtf.Dimension("heads", self.n_head)

    @property
    def key_dim(self):
        return mtf.Dimension("features_per_head", self.n_embd // self.n_head)

    @property
    def feature_dims(self):
        return [self.dim_heads, self.key_dim]

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

        input_dims = ''.join([chr(ord('i') + i) for i in range(len(reduced_dims))])
        intermediate_dims = ''.join([chr(ord(input_dims[-1]) + i) for i in range(len(new_dimensions))])
        output_dims = ''.join([chr(ord(intermediate_dims[-1]) + i) for i in range(len(new_dimensions))])

        with tf.variable_scope(f'feed_forward_{random.getrandbits(64):x}'):
            weight0 = self._get_variable(reduced_dims + intermediate_dimensions, tf.orthogonal_initializer())
            block_input = einsum([block_input, weight0],
                                 f'{self._base_dims}{input_dims},{input_dims}{intermediate_dims}->{self._base_dims}{intermediate_dims}',
                                 block_input.shape - reduced_dims + intermediate_dimensions)
            if self.dropout_rate > 0:
                block_input = mtf.dropout(block_input, 1 - self.dropout_rate)
            block_input = block_input * mtf.tanh(block_input)  # LiSHT: https://arxiv.org/abs/1901.05894
            weight1 = self._get_variable(intermediate_dimensions + new_dimensions, tf.orthogonal_initializer())
            block_input = einsum([block_input, weight1],
                                 f'{self._base_dims}{intermediate_dims},{intermediate_dims}{output_dims}->{self._base_dims}{output_dims}',
                                 block_input.shape - intermediate_dimensions + new_dimensions)
        return block_input

    def _feed_forward(self, x):
        return self._generic_feed_forward(x, self.feature_dims, self.feature_dims)

    def _block_fn(self, block_input):
        self._layer_idx += 1

        if (self._layer_idx % (self.feed_forward_per_attention + 1)) < self.feed_forward_per_attention:
            with tf.variable_scope(f"feed_forward_block_{self._layer_idx}"):
                output = self._rezero(self._feed_forward(block_input))
            return output

        with tf.variable_scope(f"attention_block_{self._layer_idx}"):
            attention_dims = block_input.shape[1:-2]  # Ex: Shape[Sequence, Width, Height]
            idx = (self._layer_idx // 2) % len(attention_dims)
            dim = attention_dims[idx]

            tmp_dim = mtf.Dimension(f'anonymous_{dim.name}', dim.size)
            attention_dims = self._base_dims + 'ha'
            renamed_input_dims = self._input_dims.replace(self._space_dims[idx], 'a')

            q = self._feed_forward(block_input)
            k = self._feed_forward(block_input)
            v = self._feed_forward(block_input)
            k = mtf.rename_dimension(k, dim.name, tmp_dim.name)
            v = mtf.rename_dimension(v, dim.name, tmp_dim.name)

            logits = einsum([q, k],
                            f"{self._input_dims},{renamed_input_dims}->{attention_dims}",
                            q.shape - self.key_dim + tmp_dim) / tmp_dim.size ** 0.5
            if idx in self.masked_attention_dimensions:
                i = mtf.range(self.mesh, tmp_dim, tf.int32) + dim.size - tmp_dim.size
                j = mtf.range(self.mesh, dim, tf.int32)
                i = mtf.broadcast(i, [tmp_dim, dim])
                j = mtf.broadcast(j, [tmp_dim, dim])
                bias = mtf.cast(mtf.less(i, j), tf.float32) * -1e12
                logits += mtf.broadcast(bias, logits.shape)
            weights = mtf.softmax(logits, dim)
            output = einsum([weights, v],
                            f"{attention_dims},{renamed_input_dims}->{self._input_dims}",
                            q.shape)
            output = self._rezero(output)

        return output

    def build(self, model_input):
        # TODO: Add support for missing model_input, token_x/y_input
        # TODO: General cleanup
        x = model_input / 255.
        context_dimension = x.shape[1]

        tgt = mtf.slice(x, 1, context_dimension.size - 1, context_dimension.name)
        src = mtf.slice(x, 0, context_dimension.size - 1, context_dimension.name)

        input_features = [src.shape[-1]]
        src = self._generic_feed_forward(src, input_features, self.feature_dims)

        src_embedding = mtf.add_n([self._get_variable([dim] + self.feature_dims, tf.random_normal_initializer())
                                   for dim in src.shape[1:-2]]  # Ex: Shape[Sequence, Width, Height]
                                  )
        src += src_embedding

        xs = (self._generic_feed_forward(src, self.feature_dims, self.feature_dims), None,
              self._generic_feed_forward(src, self.feature_dims, self.feature_dims), None)

        for layer in range(self.n_layer):
            xs = mtf.layers.reversible_half_residual_and_swap(*xs, self._block_fn)

        src = self._generic_feed_forward(xs[0] + xs[2], self.feature_dims, input_features)

        with tf.variable_scope("reduce_mean_final"):
            if self.token_embedding:
                loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(logits=src, targets=tgt,
                                                                                    vocab_dim=input_features,
                                                                                    z_loss=1e-4))
            else:
                loss = mtf.reduce_mean(mtf.abs(src - tgt))

        self._layer_idx = 0

        return src, loss

    def attribute_accesses(self):
        return {'GET':    self._get_count,
                'SET':    self._set_count,
                'unread': [k for k, v in {**self.__dict__, **self._set_count}.items() if k not in self._get_count]
                }
