import base64
import random
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf


def random_name():
    return base64.b64encode(random.getrandbits(256).to_bytes(length=32, byteorder='little')).decode().replace("+", "").replace("/", "").replace("=", "")


def anonymize(inp: mtf.Tensor, dim_name: str):
    return mtf.rename_dimension(inp, dim_name, '_' + dim_name)


def unanonymize(inp: mtf.Tensor, dim_name: str):
    return mtf.rename_dimension(inp, '_' + dim_name, dim_name)

class LishtFunction:
    def __init__(self):
        self.saved_tensor: typing.Optional[tf.Tensor] = None
    def forward(self, input):
        self.saved_tensor = input
        return tf.tanh(input) * input
    def backward(self, grad_output):
        tanh = tf.tanh(self.saved_tensor)
        return grad_output * (tanh + self.saved_tensor * (1 - tanh * tanh))

class EinsumOperation(mtf.Operation):
  """Einstein summation (matmul, etc).
  The equation follows the dimensions in the input and output shapes.
  Every dimension must occur in at least two of the input/output Tensors.
  i.e. no new dimensions in the output, and no reduction of dimensions that
  occur in only one input.
  """

  def __init__(self, inputs, output_shape, output_fn=None, name=None, _input_fn=None):
    super(EinsumOperation, self).__init__(inputs, name=name or "einsum")
    if not inputs:
      raise ValueError("Einsum needs at least one input")
    for x in inputs:
      if x.dtype != inputs[0].dtype:
        raise ValueError("Input dtypes must be equal got %s"
                         % ([y.dtype for y in inputs],))
    identity = lambda x: x
    self._input_fn = (identity,) * len(inputs) if _input_fn is None else _input_fn
    self._output_fn = identity if output_fn is None else output_fn[0]
    self._output_fn_grad = identity if output_fn is None else output_fn[1] 
    self._outputs = [mtf.Tensor(self, output_shape, inputs[0].dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    xs = self.inputs
    ret = [einsum([dy] + xs[:i] + xs[i+1:], xs[i].shape,
                           _input_fn=(self._output_fn_grad,) + self._input_fn[:i] + self._input_fn[i+1:])
          for i in range(len(xs))]
    return ret

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    xs = self.inputs
    input_shape_set = set(sum([x.shape.dims for x in xs], []))
    output_shape = self.outputs[0].shape
    intersection_shape = mtf.Shape(
        [d for d in output_shape.dims if d in input_shape_set])
    einsum_slice_fn, reduced_mesh_axes = mtf.ops._einsum_helper(
        [x.shape for x in self.inputs], intersection_shape, mesh_impl)
    y = mesh_impl.slicewise(lambda *xs: self._output_fn(einsum_slice_fn(*[fn(x) for fn, x in zip(self._input_fn, xs)])), *[lowering.tensors[x] for x in self.inputs])
    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter(
            f"allreduce/{reduced_mesh_axes}/einsum_op",
            mesh_impl.laid_out_size(intersection_shape))
      y = mtf.LazyAllreduceSum(
          mesh_impl, y, reduced_mesh_axes, add_counter_fn=add_counter_fn)
    # broadcast from intersection_shape to output_shape
    if intersection_shape != output_shape:
      y = mesh_impl.broadcast_impl(y, intersection_shape, output_shape)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = mtf.Shape(list(input_shape_set))
    lowering.add_counter("einsum", mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("einsum_unique", computation_shape.size)

def einsum(inputs, output_shape, output_fn=None, name=None, _input_fn=None):
    return EinsumOperation(inputs, output_shape, output_fn, name, _input_fn).outputs[0]


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
        self.the_batch_size = self.eval_batch_size if self.eval else self.train_batch_size
        self.use_language = self.language_token_per_frame > 0
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
        return mtf.get_variable(self.mesh, random_name(), shape, dtype=tf.float32, initializer=initializer)

    def _get_scalar(self, value):
        return self._get_variable([], tf.constant_initializer(value))

    def _rezero(self, block_input: mtf.Tensor, init: float = 0.):
        with tf.variable_scope(random_name()):
            return block_input * self._get_scalar(init)

    def _linear(self, block_input: mtf.Tensor, old: typing.List[mtf.Dimension], new: typing.List[mtf.Dimension],
                activate=None):
        with tf.variable_scope(random_name()):
            return einsum([block_input, self._get_variable(old + new, tf.orthogonal_initializer())],
                          block_input.shape - old + new,
                          activate)

    def _intermediate_dimensions(self, dimensions, intermediate_factor: float = 1.):
        return [mtf.Dimension('_intermediate',
                              int(np.prod([dim.size for dim in dimensions])
                                  * intermediate_factor
                                  * self.intermediate_feed_forward_multiplier))]
        
    def _generic_feed_forward(self,
                              block_input: mtf.Tensor,
                              reduced: typing.List[mtf.Dimension],
                              new: typing.List[mtf.Dimension],
                              intermediate_factor: float = 1.):
        intermediate = self._intermediate_dimensions(new, intermediate_factor)
        lisht = LishtFunction()
        with tf.variable_scope(random_name()):
            block_input = self._linear(block_input, reduced, intermediate, (lisht.forward, lisht.backward))
            return self._linear(block_input, intermediate, new)

    def _feed_forward(self, x: mtf.Tensor, intermediate_factor: float = 1., grouped=False):
        if grouped:
            return self._generic_feed_forward(x, [self.key_dim], [self.key_dim], intermediate_factor / self.intermediate_feed_forward_multiplier)
        return self._generic_feed_forward(x, self.feature_dims, self.feature_dims, intermediate_factor)

    def _block_fn(self, block_input):
        self._layer_idx += 1
        attention_dims = (block_input.shape - self.feature_dims)[1:]  # Ex: Shape[Sequence, Width, Height]

        if self._layer_idx % (self.feed_forward_per_attention + 1) < self.feed_forward_per_attention:
            with tf.variable_scope(random_name()):
                return self._rezero(self._feed_forward(block_input, grouped=True))

        idx = (self._layer_idx // (self.feed_forward_per_attention + 1)) % len(attention_dims)
        dim = attention_dims[idx]
        tmp_dim = mtf.Dimension(f'_{dim.name}', dim.size)

        with tf.variable_scope(random_name()):
            intermediate = self._intermediate_dimensions(self.feature_dims)
            base = self._linear(block_input, self.feature_dims, intermediate)
            q = self._linear(base, intermediate, self.feature_dims)
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
        context_dimension = x.shape[1]
        input_features = x.shape[-1:]
        spatial_ctx = x.shape[2]

        tgt = mtf.slice(x, 1, context_dimension.size - 1, context_dimension.name) / 255
        src = mtf.slice(x, 0, context_dimension.size - 1, context_dimension.name) / self._get_scalar(127.5) + self._get_scalar(-1)

        src = self._linear(src, input_features, self.feature_dims)

        if self.use_language:
            tkn_src = self._linear(mtf.one_hot(tkn_src, vocab_dim, dtype=tf.float32), [vocab_dim], self.feature_dims)
            src = unanonymize(mtf.concat([anonymize(src, spatial_ctx.name), anonymize(tkn_src, spatial_ctx.name)],
                                         '_' + spatial_ctx.name),
                              spatial_ctx.name)

        xs = (src, None, src, None)

        for layer in range(self.n_layer):
            xs = mtf.layers.reversible_half_residual_and_swap(*xs, self._block_fn)

        out = self._rezero(xs[0], 1) + self._rezero(xs[2], 1)
        loss = 0

        if self.use_language:
            out = anonymize(out, spatial_ctx)
            tkn = unanonymize(mtf.slice(out, spatial_ctx.size, self.language_token_per_frame, '_' + spatial_ctx.name),
                              spatial_ctx.name)
            out = unanonymize(mtf.slice(out, 0, spatial_ctx.size, '_' + spatial_ctx.name), spatial_ctx.name)
            tkn = self._generic_feed_forward(tkn, self.feature_dims, [vocab_dim])
            loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(logits=tkn,
                                                                                targets=tkn_tgt,
                                                                                vocab_dim=vocab_dim,
                                                                                z_loss=1e-4))

        src = mtf.sigmoid(self._generic_feed_forward(out, self.feature_dims, input_features))
        loss += mtf.reduce_mean(mtf.abs(src - tgt))

        self._layer_idx = 0

        return src, loss

    def attribute_accesses(self):
        return {'GET':    self._get_count,
                'SET':    self._set_count,
                'unread': [k for k, v in {**self.__dict__, **self._set_count}.items() if k not in self._get_count]
                }
