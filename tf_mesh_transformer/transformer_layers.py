# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers for the Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import gin

import mesh_tensorflow as mtf
from mesh_tensorflow import layers
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer import transformer

import tensorflow.compat.v1 as tf


@gin.configurable
class DenseReluDense(transformer.TransformerLayer):
  """Two dense layers with ReLU or other activation on hidden layer."""

  def __init__(self, hidden_size=4096, dropout_rate=0.0, activation="relu",
               use_bias=False):
    """Create a DenseReluDense.

    Args:
      hidden_size: an integer - size of the hidden layer
      dropout_rate: a floating-point number
      activation: an activation function or a list of activation functions.
        see documentation for mtf.layers.dense_product()
      use_bias: a boolean, whether to use bias in the dense layers.
    """
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.activation = activation
    self.use_bias = use_bias

  def call(self, context, x, losses=None):
    """Call the layer."""
    io_channels = x.shape.dims[-1]
    hidden_channels = mtf.Dimension("d_ff", self.hidden_size)
    h = mtf.layers.dense_product(x,
                                 reduced_dims=x.shape.dims[-1:],
                                 new_dims=hidden_channels,
                                 activation_functions=self.activation,
                                 use_bias=self.use_bias,
                                 variable_dtype=context.variable_dtype,
                                 name="wi",
                                 expert_dims=context.model.ensemble_dims)
    if context.train and self.dropout_rate != 0.0:
      h = mtf.dropout(h, 1.0 - self.dropout_rate,
                      noise_shape=h.shape - context.length_dim)
    return mtf.layers.dense(h, io_channels,
                            use_bias=self.use_bias,
                            activation=None,
                            variable_dtype=context.variable_dtype,
                            reduced_dims=h.shape.dims[-1:],
                            name="wo",
                            expert_dims=context.model.ensemble_dims)


def attention_params(context,
                     kv_dim,
                     num_heads,
                     num_memory_heads=0,
                     shared_kv=False,
                     no_query=False,
                     combine_dims=True,
                     keep_query_heads_dims=False,
                     fold_scaling_into_initializer=True):
  """Attention Parameters for Transformer Layers.

  The num_heads argument indicates the number of read-heads.

  For the familiar behavior described in "Attention Is All You Need", set
  num_memory_heads=0.

  If num_memory_heads==1, then there is only a single write-head, and multiple
  read-heads.  This leads to faster incremental decoding, since the
  recurrent state is smaller

  If num_memory_heads > 1, then num_memory_heads indicates the number of
  write-heads.  A fraction of the read-heads read each write-head.
  num_memory_heads must divide num_heads. This behavior has not yet been tested.

  no query flag is set to true when we do not want to create parameters
  for query params (for synthesizer model).

  Args:
    context: a transformer.Context
    kv_dim: a dimension (for key and value channels)
    num_heads: an integer
    num_memory_heads: an optional integer
    shared_kv: a boolean
    no_query: a boolean
    combine_dims: a boolean
    keep_query_heads_dims: a boolean
    fold_scaling_into_initializer: a boolean
  Returns:
    an attention.AttentionParams object
  """
  if num_heads == 1:
    query_heads_dims = None
    memory_heads_dims = None
  elif num_memory_heads == 0:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = query_heads_dims
  elif num_memory_heads == 1:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = None
  else:
    if num_heads % num_memory_heads != 0:
      raise ValueError("num_memory_heads must divide num_heads")
    memory_heads_dims = [mtf.Dimension("heads", num_memory_heads)]
    query_heads_dims = memory_heads_dims + [
        mtf.Dimension("query_heads", num_heads // num_memory_heads)]
  return attention.AttentionParams(
      context.mesh,
      query_input_dim=context.model.model_dim,
      memory_input_dim=context.model.model_dim,
      output_dim=context.model.model_dim,
      key_dim=kv_dim,
      value_dim=kv_dim,
      query_heads_dims=query_heads_dims,
      memory_heads_dims=memory_heads_dims,
      variable_dtype=context.variable_dtype,
      shared_kv=shared_kv,
      no_query=no_query,
      ensemble_dim=context.model.ensemble_dim,
      combine_dims=combine_dims,
      keep_query_heads_dims=keep_query_heads_dims,
      fold_scaling_into_initializer=fold_scaling_into_initializer)


@gin.configurable
class SelfAttention(transformer.TransformerLayer):
  """Multi-head self-attention layer."""

  def __init__(self,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,
               relative_attention_type=None,
               relative_attention_num_buckets=32,
               attention_func=None,
               combine_dims=True,
               keep_query_heads_dims=False,
               fold_scaling_into_initializer=True):
    """Create a SelfAttention Layer.

    Args:
      num_heads: an integer
      num_memory_heads: an optional integer
      key_value_size: an integer
      shared_kv: a boolean
      dropout_rate: a float
      attention_kwargs: a dictionary of kwargs for attention.attention
      relative_attention_type: an optional string - one of
        (None, "bias", "bias_shared", "contextual")
      relative_attention_num_buckets: an integer
      attention_func: attention function: None/'hybrid'.
      combine_dims: a boolean
      keep_query_heads_dims: a boolean
      fold_scaling_into_initializer: a boolean
    """
    self.num_heads = num_heads
    self.num_memory_heads = num_memory_heads
    self.key_value_size = key_value_size
    self.shared_kv = shared_kv
    self.dropout_rate = dropout_rate
    self.attention_kwargs = attention_kwargs or {}
    self.relative_attention_type = relative_attention_type
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.attention_func = attention_func
    self.combine_dims = combine_dims
    self.keep_query_heads_dims = keep_query_heads_dims
    self.fold_scaling_into_initializer = fold_scaling_into_initializer

  def layer_output_from_attention_output(self, context, attention_output,
                                         losses):
    return attention_output

  def expected_attention_output_shape(self, x, params):
    if self.keep_query_heads_dims:
      return mtf.Shape(x.shape[:-1] + params.query_heads_dims + x.shape[-1:])
    return x.shape

  def attention_kwargs_from_context(self, context):
    kwargs = copy.copy(self.attention_kwargs)
    kwargs["dropout_rate"] = self.dropout_rate if context.train else 0.0
    if "dropout_broadcast_dims" not in kwargs:
      kwargs["dropout_broadcast_dims"] = [context.length_dim]
    return kwargs

  def make_params(self, context):
    return attention_params(
        context=context,
        kv_dim=self.kv_dim,
        num_heads=self.num_heads,
        num_memory_heads=self.num_memory_heads,
        shared_kv=self.shared_kv,
        combine_dims=self.combine_dims,
        keep_query_heads_dims=self.keep_query_heads_dims,
        fold_scaling_into_initializer=self.fold_scaling_into_initializer)

  def call(self, context, x, losses=None):
    """Call the layer."""
    params = self.make_params(context)
    q = params.compute_q(x)
    memory_length = self.memory_length(context)
    if context.mode == "incremental":
      m = x
    else:
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    if self.shared_kv:
      kv = params.compute_kv(m)
    else:
      k = params.compute_k(m)
      v = params.compute_v(m)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      if self.shared_kv:
        old_kv = context.get_states(1)
        kv = old_kv * inv_one_hot + kv * one_hot
      else:
        old_k, old_v = context.get_states(2)
        k = old_k * inv_one_hot + k * one_hot
        v = old_v * inv_one_hot + v * one_hot
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([kv] if self.shared_kv else [k, v])
    if self.shared_kv:
      k = kv
      v = kv
    if self.attention_func == "hybrid":
      o = attention.hybrid_attention(
          q, k, v, context, memory_length, self.kv_dim, self.kv_dim,
          self.compute_bias(context, memory_position, x,
                            params.query_heads_dims, q),
          **self.attention_kwargs_from_context(context))
    else:
      o = attention.attention(
          q, k, v, memory_length, self.kv_dim, self.kv_dim,
          self.compute_bias(context, memory_position, x,
                            params.query_heads_dims, q),
          context=context,
          **self.attention_kwargs_from_context(context))

    attention_output_shape = self.expected_attention_output_shape(x, params)
    attention_output = params.compute_output(
        o, output_shape=attention_output_shape)
    return self.layer_output_from_attention_output(context, attention_output,
                                                   losses)

  def compute_bias(self, context, memory_position, x, heads_dims, q):
    """Compute attention bias.

    Args:
      context: a transformer.Context
      memory_position: an int32 tensor containing memory_length dimension.
      x: a Tensor - the query antecedent - required for relative attention
      heads_dims: a list of dimensions
      q: a Tensor - the queries - required for contextual relative attention
    Returns:
      a Tensor or None
    """
    min_relative_position = self.min_relative_position(context)
    max_relative_position = self.max_relative_position(context)
    biases = []
    relative_position = memory_position - context.position
    if min_relative_position is not None:
      visible = mtf.greater_equal(relative_position, min_relative_position)
      biases.append(attention.visibility_mask_to_attention_bias(
          visible, context.activation_dtype))
    if max_relative_position is not None:
      visible = mtf.less_equal(relative_position, max_relative_position)
      biases.append(attention.visibility_mask_to_attention_bias(
          visible, context.activation_dtype))
    if context.read_priority is not None:
      visible = mtf.greater_equal(
          context.read_priority,
          mtf.layers.rename_length_to_memory_length(context.write_priority))
      biases.append(attention.visibility_mask_to_attention_bias(
          visible, context.activation_dtype))

    sequence_id = None
    # Subsequence id should only be set if we are in the decoder and have
    # multiple targets per input. This will allow each sub-target to only attend
    # to itself.
    if isinstance(context.subsequence_id, mtf.Tensor):
      sequence_id = context.subsequence_id
    elif isinstance(context.sequence_id, mtf.Tensor):
      sequence_id = context.sequence_id
    if (sequence_id is not None and context.length_dim in sequence_id.shape):
      visible = mtf.equal(
          sequence_id,
          self.rename_length_to_memory_length(sequence_id, context))
      biases.append(attention.visibility_mask_to_attention_bias(
          visible, context.activation_dtype))
    if self.relative_attention_type is not None:
      buckets_dim = mtf.Dimension(
          "buckets", self.relative_attention_num_buckets)
      bidirectional = not context.model.fully_autoregressive
      rp_bucket = _relative_position_bucket(
          relative_position,
          bidirectional=bidirectional,
          num_buckets=buckets_dim.size)
      if (self.relative_attention_type == "bias" or
          self.relative_attention_type == "bias_shared"):
        bias_shape = context.model.ensemble_dims + heads_dims + [buckets_dim]
        values = None
        cache = self.relative_attention_type == "bias_shared"
        if cache:
          cache_key = ("self_attention_bias",
                       min_relative_position,
                       max_relative_position,
                       tuple(heads_dims))
          if cache_key in context.cache:
            values = context.cache[cache_key]
        if values is None:
          values = mtf.get_variable(
              context.mesh, "relative_attention_bias",
              bias_shape, dtype=context.variable_dtype)
        if cache:
          context.cache[cache_key] = values
      elif self.relative_attention_type == "contextual":
        values = layers.dense(
            q, reduced_dims=[self.kv_dim],
            new_dims=[buckets_dim],
            variable_dtype=context.variable_dtype,
            name="relative_attention_ak",
            use_bias=False,
            expert_dims=context.model.ensemble_dims + heads_dims)
      else:
        raise ValueError("unrecognized relative_attention_type \"%s\"" %
                         self.relative_attention_type)
      biases.append(mtf.gather(values, rp_bucket, buckets_dim))
    return mtf.add_n(biases) if biases else None

  @property
  def kv_dim(self):
    return mtf.Dimension("d_kv", self.key_value_size)

  def memory_length(self, context):
    return mtf.Dimension("memory_length", context.length_dim.size)

  def rename_length_to_memory_length(self, x, context):
    return mtf.replace_dimensions(
        x, context.length_dim, self.memory_length(context))

  def min_relative_position(self, context):
    return None

  def max_relative_position(self, context):
    return None


@gin.configurable
class Synthesizer(SelfAttention):
  """Multi-head Synthesizer layer https://arxiv.org/abs/2005.00743."""

  def __init__(self,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,
               relative_attention_type=None,
               relative_attention_num_buckets=32,
               attention_func=None,
               combine_dims=True,
               keep_query_heads_dims=False,
               synthesize_mode="random_plus_alpha",
               **kwargs):
    """Create a Synthesizer Layer.

    Args:
      num_heads: an integer
      num_memory_heads: an optional integer
      key_value_size: an integer
      shared_kv: a boolean
      dropout_rate: a float
      attention_kwargs: a dictionary of kwargs for attention.attention
      relative_attention_type: an optional string - one of
        (None, "bias", "bias_shared", "contextual")
      relative_attention_num_buckets: an integer
      attention_func: attention function: None/'hybrid'.
      combine_dims: a boolean
      keep_query_heads_dims: a boolean
      synthesize_mode: a string to select synthesizer variant
      **kwargs: additional constructor params
    """
    super(Synthesizer, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.num_memory_heads = num_memory_heads
    self.key_value_size = key_value_size
    self.shared_kv = shared_kv
    self.dropout_rate = dropout_rate
    self.attention_kwargs = attention_kwargs or {}
    self.relative_attention_type = relative_attention_type
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.attention_func = attention_func
    self.combine_dims = combine_dims
    self.keep_query_heads_dims = keep_query_heads_dims
    self.synthesize_mode = synthesize_mode
    self.no_query = False
    if "plus" in self.synthesize_mode:
      self.shared_kv = False
      self.no_query = False
    elif "minus" in self.synthesize_mode:
      # We still keep the query as first projection
      self.shared_kv = True
      self.no_query = False
    else:
      self.shared_kv = True
      self.shared_q = True

  def make_params(self, context):
    return attention_params(context=context,
                            kv_dim=self.kv_dim,
                            num_heads=self.num_heads,
                            num_memory_heads=self.num_memory_heads,
                            shared_kv=self.shared_kv,
                            no_query=self.no_query)

  def call(self, context, x, losses=None):
    """Call the layer."""
    params = self.make_params(context)
    q = params.compute_q(x)
    memory_length = self.memory_length(context)
    if context.mode == "incremental":
      m = x
    else:
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    if self.shared_kv:
      kv = params.compute_kv(m)
    else:
      k = params.compute_k(m)
      v = params.compute_v(m)
    if self.no_query:
      # we don't use q for some synthesizer modes that don't use QKV at all.
      q = x
    else:
      q = params.compute_q(x)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      if self.shared_kv:
        old_kv = context.get_states(1)
        kv = old_kv * inv_one_hot + kv * one_hot
      else:
        old_k, old_v = context.get_states(2)
        k = old_k * inv_one_hot + k * one_hot
        v = old_v * inv_one_hot + v * one_hot
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([kv] if self.shared_kv else [k, v])
    if self.shared_kv:
      k = kv
      v = kv
    o = attention.synthetic_attention(q, k, v, memory_length,
                                      self.kv_dim, self.kv_dim,
                                      self.compute_bias(context,
                                                        memory_position,
                                                        x,
                                                        params.query_heads_dims,
                                                        q),
                                      synthesize=True,
                                      synthesize_mode=self.synthesize_mode,
                                      context=context,
                                      **self.attention_kwargs_from_context(
                                          context))
    attention_output_shape = self.expected_attention_output_shape(x, params)
    attention_output = params.compute_output(
        o, output_shape=attention_output_shape)
    return self.layer_output_from_attention_output(context, attention_output,
                                                   losses)


@gin.configurable
def relative_position_spans(context, num_sentinels=gin.REQUIRED):
  """Compute relative positions between inputs and targets.

  Used by enc_dec_attention_bias.

  Assumes that inputs and targets were generated by a span-filling objective:
  The inputs consist of the original text with some spans removed and replaced
  by single sentinels.
  The targets consist of the dropped spans, each preceded by a single sentinel.
  Sentinels are the last tokens in the vocabulary.

  e.g.
  inputs:  A B C <S> F G H <S>
  shifted-targets: <BOS> <S> D E <S> I J K

  Relative positions are computed by identifying a target token with the
  corresponding sentinel in the input and returning the distance between these
  two tokens in the input.

  Target tokens which precede all sentinels get identified with the beginning of
  the input.  So if we apply this to a problem with no sentinels, all target
  tokens will be indentified with the beginning of the input.  We assume this is
  the case during incremental decoding, so this code will not work properly to
  incrementally decode a problem with sentinels.  This may not be an issue,
  since the span-filling objective is primarily used for unsupervised
  pre-training.

  Args:
    context: a Context
    num_sentinels: an integer.  Should have the same value as
       sentencepiece_vocabulary.SentencePieceVocabulary.extra_ids
  Returns:
    a Tensor
  """
  decoder_id = context.inputs
  encoder_id = context.encoder_inputs
  decoder_length = context.length_dim
  encoder_length = context.encoder_length_dim
  mesh = encoder_id.mesh
  encoder_pos = mtf.range(mesh, encoder_length, tf.int32)
  if decoder_length not in decoder_id.shape.dims:
    # we are doing incremental decoding.
    # Map the target token to the beginning of the input.
    dec_to_enc_pos = 0
  else:
    vocab_size = context.model.input_vocab_size_unpadded
    def sentinel_mask(t):
      return mtf.cast(mtf.greater_equal(
          t, vocab_size - num_sentinels), tf.int32)
    decoder_is_sentinel = sentinel_mask(decoder_id)
    encoder_is_sentinel = sentinel_mask(encoder_id)
    encoder_segment_id = mtf.cumsum(encoder_is_sentinel, encoder_length)
    decoder_segment_id = mtf.cumsum(decoder_is_sentinel, decoder_length)
    encoder_sequence_id = context.encoder_sequence_id
    decoder_sequence_id = context.sequence_id
    if encoder_sequence_id is not None:
      # distinguish segments from different sequences
      multiplier = max(encoder_length.size, decoder_length.size)
      encoder_segment_id += encoder_sequence_id * multiplier
      decoder_segment_id += decoder_sequence_id * multiplier
    dec_to_enc_pos = mtf.reduce_sum(
        mtf.cast(mtf.less(encoder_segment_id, decoder_segment_id), tf.int32),
        reduced_dim=encoder_length)
  return dec_to_enc_pos - encoder_pos


@gin.configurable
def enc_dec_attention_bias(layer,
                           context,
                           heads_dims,
                           relative_position_fn=relative_position_spans):
  """Compute bias term for encoder-decoder attention.

  Args:
    layer: a TransformerLayer
    context: a Context
    heads_dims: a list of Dimension
    relative_position_fn: an optional function
  Returns:
    a Tensor
  """
  biases = []
  if context.encoder_sequence_id and context.sequence_id:
    visible = mtf.equal(context.sequence_id, context.encoder_sequence_id)
    biases.append(attention.visibility_mask_to_attention_bias(
        visible, context.activation_dtype))
  if (layer.relative_attention_type == "bias" or
      layer.relative_attention_type == "bias_shared"):
    buckets_dim = mtf.Dimension(
        "buckets", layer.relative_attention_num_buckets)
    bias_shape = context.model.ensemble_dims + heads_dims + [buckets_dim]
    values = None
    cache = layer.relative_attention_type == "bias_shared"
    if cache:
      cache_key = ("enc_dec_relative_attention_bias", tuple(heads_dims))
      if cache_key in context.cache:
        values = context.cache[cache_key]
    if values is None:
      values = mtf.get_variable(
          context.mesh, "enc_dec_relative_attention_bias",
          bias_shape, dtype=context.variable_dtype)
    if cache:
      context.cache[cache_key] = values
    rel_pos = relative_position_fn(context)
    rp_bucket = _relative_position_bucket(
        rel_pos,
        bidirectional=True,
        num_buckets=buckets_dim.size)
    biases.append(mtf.gather(values, rp_bucket, buckets_dim))
  elif layer.relative_attention_type is not None:
    raise ValueError("unrecognized relative_attention_type \"%s\"" %
                     layer.relative_attention_type)
  return mtf.add_n(biases) if biases else None


@gin.configurable
def enc_dec_attention(self_attention_layer, memory_antecedent, context, x,
                      losses):
  """Multi-head attention over the encoder outputs."""
  memory_input_dim = memory_antecedent.shape[-1]
  if memory_input_dim != context.model.model_dim:
    raise NotImplementedError(
        "TODO(noam): support different model_dim in encoder and decoder.")
  params = self_attention_layer.make_params(context)
  q = params.compute_q(x)
  if context.mode == "incremental":
    k, v, memory_length = context.get_constant_state()
  else:
    m = memory_antecedent
    if self_attention_layer.shared_kv:
      kv = params.compute_kv(m)
      k = kv
      v = kv
    else:
      k = params.compute_k(m)
      v = params.compute_v(m)
    memory_length, = [d for d in m.shape.dims if d.name == "memory_length"]
    if context.mode == "first_part":
      context.record_constant_state((k, v, memory_length))
  bias = enc_dec_attention_bias(self_attention_layer,
                                context,
                                params.query_heads_dims)
  a = attention.attention(
      q, k, v, memory_length, self_attention_layer.kv_dim,
      self_attention_layer.kv_dim, bias,
      context=context,
      **self_attention_layer.attention_kwargs_from_context(context))
  attention_output_shape = self_attention_layer.expected_attention_output_shape(
      x, params)
  attention_output = params.compute_output(
      a, output_shape=attention_output_shape)
  return self_attention_layer.layer_output_from_attention_output(
      context, attention_output, losses)


@gin.configurable
class EncDecAttention(SelfAttention):
  """Multi-head attention over encoder output."""

  def __init__(self, relative_attention_type=None, **kwargs):
    super(EncDecAttention, self).__init__(
        relative_attention_type=relative_attention_type, **kwargs)

  def _get_memory_antecedent(self, context):
    return context.encoder_output

  def call(self, context, x, losses=None):
    """Call the layer."""
    return enc_dec_attention(self, self._get_memory_antecedent(context),
                             context, x, losses)


@gin.configurable
class TransparentEncDecAttention(EncDecAttention):
  """Transparent multi-head attention over encoder output."""

  def __init__(self,
               layers_per_encoder_module=gin.REQUIRED,
               layers_per_decoder_module=gin.REQUIRED,
               encoder_num_modules=gin.REQUIRED,
               decoder_num_modules=gin.REQUIRED,
               dropout_rate=0.0,
               **kwargs):
    """Create a transparent attention EncDec Layer.

    Args:
      layers_per_encoder_module: positive integer telling how many layer are in
        each repeated module in the encoder
      layers_per_decoder_module: positive integer telling how many layer are in
        each repeated module in the decoder
      encoder_num_modules: positive integer of how many repeated modules there
        are in the encoder
      decoder_num_modules: positive integer of how many repeated modules there
        are in the decoder
      dropout_rate: positive float, the dropout rate for the matrix relating
        encoder outputs to decoder inputs
      **kwargs: additional constructor params
    """
    super(TransparentEncDecAttention, self).__init__(**kwargs)
    self.layers_per_encoder_module = layers_per_encoder_module
    self.layers_per_decoder_module = layers_per_decoder_module
    self.encoder_num_modules = encoder_num_modules
    self.decoder_num_modules = decoder_num_modules
    self.dropout_rate = dropout_rate

  def _get_memory_antecedent(self, context):
    decoder_module_index = context.layer_index // self.layers_per_decoder_module
    decoder_inputs = self._get_decoder_inputs(context)
    return decoder_inputs[decoder_module_index]

  def _get_decoder_inputs(self, context):
    """Computes the inputs to the decoder when using transparent attention.

    We must cache on the context in order to ensure that we are not replicating
    variables when the layer's call function is called in different tf variable
    scopes.

    Args:
      context: a Context

    Returns:
      a list containing `self.num_decoder_modules` of tensors with shape
        [<batch_dims>, length_dim, output_vocab_dim]
    """
    if hasattr(context, "decoder_layers_per_module"):
      return context.decoder_layers_per_module

    encoder_layer_outputs = [
        mtf.layers.rename_length_to_memory_length(output)
        for output in context.encoder_layer_outputs
    ]

    layers_per_module = self.layers_per_encoder_module
    encoder_module_outputs_dim = mtf.Dimension(
        "encoder_module_outputs", size=self.encoder_num_modules + 1)
    decoder_module_inputs_dim = mtf.Dimension(
        "decoder_module_inputs", size=self.decoder_num_modules)
    encoder_module_outputs = mtf.stack(
        [encoder_layer_outputs[0]] +
        encoder_layer_outputs[layers_per_module::layers_per_module],
        dim_name="encoder_module_outputs")
    stddev = 1.0
    if not mtf.layers.unit_scaling_convention():
      stddev *= encoder_module_outputs_dim.size ** -0.5
    w = mtf.get_variable(
        context.mesh,
        "w",
        mtf.Shape([encoder_module_outputs_dim, decoder_module_inputs_dim]),
        initializer=tf.random_normal_initializer(stddev=stddev),
        dtype=context.variable_dtype)
    if context.train and self.dropout_rate != 0.0:
      w = mtf.dropout(w, 1.0 - self.dropout_rate)
    s = mtf.softmax(w, reduced_dim=encoder_module_outputs_dim)
    z = mtf.layers.us_einsum([s, encoder_module_outputs],
                             reduced_dims=[encoder_module_outputs_dim])
    input_per_decoder = mtf.split(
        z,
        split_dim=decoder_module_inputs_dim,
        num_or_size_splits=decoder_module_inputs_dim.size)
    context.decoder_layers_per_module = [
        mtf.reshape(inpt, z.shape.dims[1:]) for inpt in input_per_decoder
    ]
    return context.decoder_layers_per_module


@gin.configurable
class LocalSelfAttention(SelfAttention):
  """Multi-head local self-attention layer."""

  def __init__(self,
               radius=128,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,):
    super(LocalSelfAttention, self).__init__(
        num_heads,
        num_memory_heads,
        key_value_size,
        shared_kv,
        dropout_rate,
        attention_kwargs)
    self.radius = radius

  def call(self, context, x, losses=None):
    """Call the layer."""
    params = self.make_params(context)
    q = params.compute_q(x)
    if self.shared_kv:
      kv = params.compute_kv(x)
      k = kv
      v = kv
    else:
      k = params.compute_k(x)
      v = params.compute_v(x)
    if context.mode == "incremental":
      if self.shared_kv:
        prev_kv, = context.get_states(1)
      else:
        prev_k, prev_v = context.get_states(2)
      current_position = mtf.equal(
          mtf.range(context.mesh, self.window_dim, dtype=tf.int32),
          mtf.mod(context.position, self.radius))
      if self.shared_kv:
        kv = mtf.where(current_position, kv, prev_kv,
                       output_shape=prev_kv.shape)
        k = kv
        v = kv
        context.record_new_states([kv])
      else:
        k = mtf.where(current_position, params.compute_k(x), prev_k,
                      output_shape=prev_k.shape)
        v = mtf.where(current_position, params.compute_v(x), prev_v,
                      output_shape=prev_v.shape)
        context.record_new_states([k, v])
      window_pos = mtf.range(context.mesh, self.window_dim, tf.int32)
      visible = mtf.greater_equal(context.position, window_pos)
      bias = attention.visibility_mask_to_attention_bias(
          visible, context.activation_dtype)
      o = attention.attention(
          q,
          k,
          v,
          self.window_dim,
          self.kv_dim,
          self.kv_dim,
          bias,
          **self.attention_kwargs_from_context(context))
    elif context.length_dim.size <= max(256, self.radius * 4):
      # nothing fancy - just do full attention and mask
      memory_length = self.rename_length_to_memory_length(
          context.position, context)
      o = attention.attention(
          q, self.rename_length_to_memory_length(k, context),
          self.rename_length_to_memory_length(v, context),
          self.memory_length(context), self.kv_dim, self.kv_dim,
          self.compute_bias(context, memory_length, x, params.query_heads_dims,
                            q), **self.attention_kwargs_from_context(context))
    else:
      # fancy local attention algorithm
      o = attention.local_attention_1d(
          q=q,
          k=k,
          v=None if self.shared_kv else v,
          length_dim=context.length_dim,
          key_dim=self.kv_dim,
          value_dim=self.kv_dim,
          length_dim_num_splits=1,  # TODO(noam): look at the layout
          autoregressive=context.model.fully_autoregressive,
          radius=self.radius,
          sequence_id=context.sequence_id,
          write_priority=context.write_priority,
          read_priority=context.read_priority,
          attention_kwargs=self.attention_kwargs_from_context(context))
    if context.mode == "first_part":
      window_pos = mtf.range(context.mesh, self.window_dim, tf.int32)
      pos = mtf.range(context.mesh, context.length_dim, tf.int32)
      select_recent = mtf.cast(
          mtf.equal(mtf.mod(pos, self.radius), window_pos), x.dtype)
      select_recent *= mtf.cast(
          mtf.less(pos, context.initial_position), x.dtype)
      select_recent *= mtf.cast(
          mtf.greater_equal(
              pos, context.initial_position - self.radius), x.dtype)
      state_shape = (k.shape - [context.length_dim, self.kv_dim]
                     + [self.window_dim, self.kv_dim])
      k_state = mtf.einsum(
          [k, select_recent], output_shape=state_shape,
          reduced_dims=[context.length_dim])
      context.new_states.append(k_state)
      if not self.shared_kv:
        v_state = mtf.einsum(
            [v, select_recent], output_shape=state_shape,
            reduced_dims=[context.length_dim])
        context.new_states.append(v_state)
    return params.compute_output(o, output_shape=x.shape)

  def min_relative_position(self, context):
    return 1 - self.radius

  def max_relative_position(self, context):
    return None if context.model.fully_autoregressive else self.radius

  @property
  def window_dim(self):
    return mtf.Dimension("window", self.radius)


def _relative_position_bucket(relative_position,
                              bidirectional=True,
                              num_buckets=32,
                              max_distance=128):
  """Translate relative position to a bucket number for relative attention.

  The relative position is defined as memory_position - query_position, i.e.
  the distance in tokens from the attending position to the attended-to
  position.  If bidirectional=False, then positive relative positions are
  invalid.

  We use smaller buckets for small absolute relative_position and larger buckets
  for larger absolute relative_positions.  All relative positions >=max_distance
  map to the same bucket.  All relative positions <=-max_distance map to the
  same bucket.  This should allow for more graceful generalization to longer
  sequences than the model has been trained on.

  Args:
    relative_position: an int32 Tensor
    bidirectional: a boolean - whether the attention is bidirectional
    num_buckets: an integer
    max_distance: an integer
  Returns:
    a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
  """
  ret = 0
  n = -relative_position
  if bidirectional:
    num_buckets //= 2
    ret += mtf.to_int32(mtf.less(n, 0)) * num_buckets
    n = mtf.abs(n)
  else:
    n = mtf.maximum(n, 0)
  # now n is in the range [0, inf)
  max_exact = num_buckets // 2
  is_small = mtf.less(n, max_exact)
  val_if_large = max_exact + mtf.to_int32(
      mtf.log(mtf.to_float(n) / max_exact)
      / math.log(max_distance / max_exact) * (num_buckets - max_exact))
  val_if_large = mtf.minimum(val_if_large, num_buckets - 1)
  ret += mtf.where(is_small, n, val_if_large)
  return ret


@gin.configurable
class TalkingHeadsSelfAttention(SelfAttention):
  """Experimental Talking-heads self-attention layer.

  https://arxiv.org/abs/2003.02436

  This is a variant where there are (optionally) extra learned linear
  projections on the attention logits and attention weights.  These linear
  projections are across attention heads (but not across different query or
  memory positions).

  The user specifies three sets of mtf.Dimension:
    key_heads_dims: "heads" dimensions the queries, keys and ther dot-product
    softmax_heads_dims: "heads" dimensions for the logits and their softmax
    value_heads_dims: "heads" dimensions for the values

  If these three sets are identical, then this layer is identical to ordinary
  multi-head attention.

  If key_heads_dims != softmax_heads_dims, then a learned linear projection
  is applied to compute the logits.  This projection reduces out dimensions
  in (key_heads_dims-softmax_heads_dims) and inserts dimensions in
  (softmax_heads_dims-key_heads_dims).

  If softmax_heads_dims != value_heads_dims, then a learned linear
  projection is applied to the weights (the output of the softmax).  This
  projection reduces out dimensions in (softmax_heads_dims-value_heads_dims)
  and inserts dimensions in (value_heads_dims-softmax_heads_dims).

  TPU performance is lousy due to small matrix sizes.

  Early experiments show that quality can be significantly better than baseline.

  An additional supported option is dynamic talking-heads projections where the
  talking-heads projections themselves contain terms that depend on the inputs.
  Each of the logits-projection and the weights-projection can depend on either
  or both of the query-antecedent X or the memory-antecedent Y.  This gives
  a total of four dynamic projections which can be enabled individually.
  To enable, set the dynamic_projections argument to a list containing a
  some or all of the strings ["x2l", "m2l", "x2w", "m2w"].

  Example:
    TalkingHeadsSelfAttention.key_heads_dims = [("key_heads", 12)]
    TalkingHeadsSelfAttention.softmax_heads_dims = [("heads", 32)]
    TalkingHeadsSelfAttention.value_heads_dims = [("value_heads", 12)]
    TalkingHeadsSelfAttention.key_size = 64
    TalkingHeadsSelfAttention.value_size = 64
    d_model = 1024

    We start with an input x
      x: [length, d_model]

    The input is first transformed into queries, keys and values:
      queries: [query_length, key_heads, key_size]
      keys: [memory_length, key_heads, key_size]
      values: [memory_length, value_heads, value_size]

    queries and keys get einsummed to produce a tensor p:
      p: [query_length, memory_length, key_heads]

    p gets linearly transformed with a learned weight matrix with shape
      [key_heads, softmax_heads] to produce logits
    logits: [query_length, memory_length, softmax_heads]

    take the softmax of logits (across memory_length to produce weights)
    h: [query_length, memory_length, softmax_heads]

    Now a learned linear projection with shape [softmax_heads, value_heads]
      on h produces the weights.
    weights: [query_length, memory_length, value_heads]

    As usual, we einsum the weights with the values.
    o: [query_length, value_heads, value_size]

    Finally, project o back to the desired output dimension
    y: [query_length, d_model]


  Also, this doesn't model-parallelize trivially.  To model-parallelize, you
  should add one heads-dimension that is present in all of key_heads_dims,
  softmax_heads_dims, value_heads_dims.  Call this dimension "heads" and shard
  that over multiple devices.  Then also include additional different
  heads-dimension for the keys, softmax, and values.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               key_heads_dims=(("heads", 12),),
               softmax_heads_dims=(("heads", 12),),
               value_heads_dims=(("heads", 12),),
               key_size=64,
               value_size=64,
               dropout_rate=0.0,
               relative_attention_type=None,
               relative_attention_num_buckets=32,
               dynamic_projections=None,
               dynamic_projections_init_scale=1e-2):
    """Create a SelfAttention Layer.

    Args:
      key_heads_dims: a list of mtf.Dimension or (name, size) pairs
      softmax_heads_dims: a list of mtf.Dimension or (name, size) pairs
      value_heads_dims: a list of mtf.Dimension or (name, size) pairs
      key_size: an integer
      value_size: an integer
      dropout_rate: a float
      relative_attention_type: an optional string - one of
        (None, "bias", "bias_shared", "contextual")
      relative_attention_num_buckets: an integer
      dynamic_projections: an optional sequence containing a subset of
        ["x2l", "m2l", "x2w", "m2w"] (see class comments)
      dynamic_projections_init_scale: a float - initializer variance scaling
        factor for these dynamic projections.  We have observed learning
        difficulties when this value is too large.
    """
    self.key_heads_dims = [mtf.convert_to_dimension(d) for d in key_heads_dims]
    self.softmax_heads_dims = [
        mtf.convert_to_dimension(d) for d in softmax_heads_dims]
    self.value_heads_dims = [
        mtf.convert_to_dimension(d) for d in value_heads_dims]
    self.key_dim = mtf.Dimension("d_k", key_size)
    self.value_dim = mtf.Dimension("d_v", value_size)
    self.dropout_rate = dropout_rate
    self.relative_attention_type = relative_attention_type
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.dynamic_projections = dynamic_projections or []
    self.dynamic_projections_init_scale = dynamic_projections_init_scale

  def compute_q(self, context, x):
    # Scale the initializer variance by 1.0/d_k
    # This scales the initializer by rsqrt(d_k)
    init_scale = 1.0
    if not mtf.layers.unit_scaling_convention():
      init_scale /= self.key_dim.size
    kernel_initializer = mtf.layers.VarianceScalingInitializer(init_scale)
    return mtf.layers.dense(
        x, reduced_dims=[context.model.model_dim],
        new_dims=self.key_heads_dims + [self.key_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="q", expert_dims=context.model.ensemble_dims,
        kernel_initializer=kernel_initializer)

  def compute_k(self, context, x):
    return mtf.layers.dense(
        x, reduced_dims=[context.model.model_dim],
        new_dims=self.key_heads_dims + [self.key_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="k", expert_dims=context.model.ensemble_dims)

  def compute_v(self, context, x):
    return mtf.layers.dense(
        x, reduced_dims=[context.model.model_dim],
        new_dims=self.value_heads_dims + [self.value_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="v", expert_dims=context.model.ensemble_dims)

  def compute_y(self, context, u):
    return mtf.layers.dense(
        u, reduced_dims=self.value_heads_dims + [self.value_dim],
        new_dims=[context.model.model_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="y", expert_dims=context.model.ensemble_dims)

  def call(self, context, x, losses=None):
    """Call the layer."""
    memory_length = self.memory_length(context)
    q = self.compute_q(context, x)
    if context.mode == "incremental":
      m = x
    else:
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    k = self.compute_k(context, m)
    v = self.compute_v(context, m)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      old_k, old_v = context.get_states(2)
      k = old_k * inv_one_hot + k * one_hot
      v = old_v * inv_one_hot + v * one_hot
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([k, v])
    bias = self.compute_bias(context, memory_position, x,
                             self.softmax_heads_dims, q)
    return self.attention_internal(context, x, m, q, k, v, memory_length, bias)

  def attention_internal(self, context, x, m, q, k, v, memory_length, bias):
    p = mtf.layers.us_einsum([q, k], reduced_dims=[self.key_dim])
    logits = self.talking_heads(
        context, p, "logits", self.key_heads_dims, self.softmax_heads_dims,
        dynamic_projections_from=(
            ([x] if "x2l" in self.dynamic_projections else []) +
            ([m] if "m2l" in self.dynamic_projections else [])))
    if bias is not None:
      logits += bias
    h = mtf.softmax(logits, memory_length)
    weights = self.talking_heads(
        context, h, "weights", self.softmax_heads_dims, self.value_heads_dims,
        dynamic_projections_from=(
            ([x] if "x2w" in self.dynamic_projections else []) +
            ([m] if "m2w" in self.dynamic_projections else [])))
    # TODO(noam): make dropout_broadcast_dims configurable
    dropout_broadcast_dims = [context.length_dim]
    weights = mtf.dropout(
        weights, rate=self.dropout_rate if context.train else 0.0,
        noise_shape=weights.shape - dropout_broadcast_dims)
    u = mtf.einsum([weights, v], reduced_dims=[memory_length])
    return self.compute_y(context, u)

  def talking_heads(
      self, context, inp, name, input_heads_dims, output_heads_dims,
      dynamic_projections_from=None):
    shared_dims = [d for d in input_heads_dims if d in output_heads_dims]
    reduced_dims = [d for d in input_heads_dims if d not in output_heads_dims]
    new_dims = [d for d in output_heads_dims if d not in input_heads_dims]
    if not (reduced_dims or new_dims):
      # Output dimensions are same as input dimensions.  Return the input
      return inp
    elif dynamic_projections_from:
      # There are one or more dynamic talking-heads-projections
      with tf.variable_scope(name):
        # static projection - this is the same as the static projection in the
        # "else" case below.  We create the weight matrix with get_variable
        # instead of calling mtf.layers.dense() so that we can fold the
        # static projection into one of the dynamic projections.
        static_p_initializer = mtf.layers.VarianceScalingInitializer()(
            reduced_dims, new_dims)
        static_p_shape = (
            context.model.ensemble_dims + shared_dims + reduced_dims + new_dims)
        static_p = mtf.get_variable(inp.mesh,
                                    "kernel",
                                    static_p_shape,
                                    initializer=static_p_initializer,
                                    dtype=context.variable_dtype)
        ps = []
        for i, dp_from in enumerate(dynamic_projections_from):
          init_scale = self.dynamic_projections_init_scale
          if not mtf.layers.unit_scaling_convention():
            init_scale /= mtf.Shape(reduced_dims).size
          kernel_initializer = mtf.layers.VarianceScalingInitializer(
              init_scale)
          ps.append(
              mtf.layers.dense(
                  dp_from, reduced_dims=[context.model.model_dim],
                  new_dims=shared_dims + reduced_dims + new_dims,
                  use_bias=False, activation=None,
                  variable_dtype=context.variable_dtype,
                  name="%s_dynamic_%d" % (name, i),
                  expert_dims=context.model.ensemble_dims,
                  kernel_initializer=kernel_initializer))
        # Fold the static projection into one of the static projections.
        # Mathematically, we could add all the dynamic projections together
        #   here, but it would create a very large tensor which contained
        #   both the query-length and memory-length dimensions, and would
        #   probably be slower in practice.
        ps[0] += static_p
        return mtf.add_n(
            [mtf.layers.us_einsum([inp, p], reduced_dims=reduced_dims)
             for p in ps])
    else:
      # No dynamic projections.  Static talking-heads projection only
      return mtf.layers.dense(
          inp, reduced_dims=reduced_dims,
          new_dims=new_dims,
          use_bias=False, activation=None,
          variable_dtype=context.variable_dtype,
          name=name, expert_dims=context.model.ensemble_dims + shared_dims)


@gin.configurable
class TalkingHeadsEncDecAttention(TalkingHeadsSelfAttention):
  """Talking-heads attention over encoder output.

  See comments on TalkingHeadsSelfAttention.
  """

  def __init__(self, relative_attention_type=None, **kwargs):
    super(TalkingHeadsEncDecAttention, self).__init__(
        relative_attention_type=relative_attention_type, **kwargs)

  def _get_memory_antecedent(self, context):
    return context.encoder_output

  def call(self, context, x, losses=None):
    """Call the layer."""
    m = self._get_memory_antecedent(context)
    memory_input_dim = m.shape[-1]
    if memory_input_dim != context.model.model_dim:
      raise NotImplementedError(
          "TODO(noam): support different model_dim in encoder and decoder.")
    q = self.compute_q(context, x)
    if context.mode == "incremental":
      k, v, memory_length = context.get_constant_state()
    else:
      k = self.compute_k(context, m)
      v = self.compute_v(context, m)
      memory_length, = [d for d in m.shape.dims if d.name == "memory_length"]
      if context.mode == "first_part":
        context.record_constant_state((k, v, memory_length))
    bias = enc_dec_attention_bias(self,
                                  context,
                                  self.softmax_heads_dims)
    return self.attention_internal(context, x, m, q, k, v, memory_length, bias)


@gin.configurable
class GeneralBilinearSelfAttention(SelfAttention):
  """General Bilinear Self-Attention.

  Described in the forthcoming talking-heads paper.

  Equivalent to multi-head attentino where d_kv == d_model.
  It is redundant to have projections on both q and k.
  It is redundant to have projections on both v and output.
  We therefore omit the projections on k and v, making the two identical.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               heads_dims=(("heads", 12),),
               dropout_rate=0.0,
               relative_attention_type=None,
               relative_attention_num_buckets=32):
    """Create a GeneralBilinearSelfAttention Layer.

    Args:
      heads_dims: a list of mtf.Dimension or (name, size) pairs
      dropout_rate: a float
      relative_attention_type: an optional string - one of
        (None, "bias", "bias_shared", "contextual")
      relative_attention_num_buckets: an integer
    """
    self.heads_dims = [
        mtf.convert_to_dimension(d) for d in heads_dims]
    self.dropout_rate = dropout_rate
    self.relative_attention_type = relative_attention_type
    self.relative_attention_num_buckets = relative_attention_num_buckets

  def compute_q(self, context, x):
    # Scale the initializer variance by 1.0/d_k
    # This scales the initializer by rsqrt(d_k)
    init_scale = 1.0
    if not mtf.layers.unit_scaling_convention():
      init_scale /= context.model.model_dim.size
    return mtf.layers.dense(
        x, reduced_dims=[context.model.model_dim],
        new_dims=self.heads_dims + [context.model.model_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="q", expert_dims=context.model.ensemble_dims,
        kernel_initializer=mtf.layers.VarianceScalingInitializer(init_scale))

  def compute_y(self, context, u):
    return mtf.layers.dense(
        u, reduced_dims=self.heads_dims + [context.model.model_dim],
        new_dims=[context.model.model_dim],
        use_bias=False, activation=None,
        variable_dtype=context.variable_dtype,
        name="y", expert_dims=context.model.ensemble_dims)

  def call(self, context, x, losses=None):
    """Call the layer."""
    memory_length = self.memory_length(context)
    q = self.compute_q(context, x)
    if context.mode == "incremental":
      m = x
    else:
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      old_m, = context.get_states(1)
      m = old_m * inv_one_hot + one_hot * m
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([m])
    bias = self.compute_bias(context, memory_position, x, self.heads_dims, q)
    return self.attention_internal(context, q, m, memory_length, bias)

  def attention_internal(self, context, q, m, memory_length, bias):
    logits = mtf.layers.us_einsum(
        [q, m], reduced_dims=[context.model.model_dim])
    if bias is not None:
      logits += bias
    weights = mtf.softmax(logits, memory_length)
    # TODO(noam): make dropout_broadcast_dims configurable
    dropout_broadcast_dims = [context.length_dim]
    weights = mtf.dropout(
        weights, rate=self.dropout_rate if context.train else 0.0,
        noise_shape=weights.shape - dropout_broadcast_dims)
    u = mtf.einsum([weights, m], reduced_dims=[memory_length])
    return self.compute_y(context, u)


@gin.configurable
class GeneralBilinearEncDecAttention(GeneralBilinearSelfAttention):
  """Talking-heads attention over encoder output.

  See comments on GBMSelfAttention.
  """

  def __init__(self, relative_attention_type=None, **kwargs):
    super(GeneralBilinearEncDecAttention, self).__init__(
        relative_attention_type=relative_attention_type, **kwargs)

  def _get_memory_antecedent(self, context):
    return context.encoder_output

  def call(self, context, x, losses=None):
    """Call the layer."""
    memory_antecedent = self._get_memory_antecedent(context)
    memory_input_dim = memory_antecedent.shape[-1]
    if memory_input_dim != context.model.model_dim:
      raise NotImplementedError(
          "TODO(noam): support different model_dim in encoder and decoder.")
    q = self.compute_q(context, x)
    if context.mode == "incremental":
      m, memory_length = context.get_constant_state()
    else:
      m = memory_antecedent
      memory_length, = [d for d in m.shape.dims if d.name == "memory_length"]
      if context.mode == "first_part":
        context.record_constant_state((m, memory_length))
    bias = enc_dec_attention_bias(self,
                                  context,
                                  self.heads_dims)
    return self.attention_internal(context, q, m, memory_length, bias)


