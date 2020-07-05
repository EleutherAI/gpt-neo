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

"""Extension to implement universal transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer

import tensorflow.compat.v1 as tf


@gin.configurable
class UTLayerStack(transformer.TransformerLayer):
  """A stack of layers for Universal Transformer.

  This implementation is largely adapted from t2t universal transformer
  implementation. Reference:
  third_party/py/tensor2tensor/models/research
  """

  def __init__(
      self,
      layers,
      dropout_rate=0.0,
      norm_epsilon=1e-6,
      num_vanilla_transformer_layers=2,
      couple_carry_transform_gates=True,
      act_type=gin.REQUIRED,
      recurrence_type=gin.REQUIRED,
      act_max_steps=gin.REQUIRED,
      act_epsilon=gin.REQUIRED,
      num_rec_steps=gin.REQUIRED,
      num_inrecurrence_layers=gin.REQUIRED,
      position_start_index=gin.REQUIRED,
      add_or_concat_timing_signal=gin.REQUIRED,
      step_timing_signal_type=gin.REQUIRED,
      add_position_timing_signal=gin.REQUIRED,
      add_step_timing_signal=gin.REQUIRED,
      mix_with_transformer_before_ut=gin.REQUIRED,
      mix_with_transformer_after_ut=gin.REQUIRED,
      gates_inputs=gin.REQUIRED,
      gate_ffn_layer=gin.REQUIRED,
      use_gated_transformer=gin.REQUIRED,
      gating_type=gin.REQUIRED,
  ):
    """Create a LayerStack for Universal Transformer.

    Args:
      layers: a list of TransformerLayer
      dropout_rate: a floating-point number
      norm_epsilon: a floating-point number
      num_vanilla_transformer_layers: number of vanilla transformer layers
        before the ACT layer.
      couple_carry_transform_gates: whether to couple carry and transform gates.
      act_type: act type
      recurrence_type: recurrence type (allowable values: "act").
      act_max_steps: maximum number of act steps
      act_epsilon: halting threshold
      num_rec_steps: maximum number of recurrent steps
      num_inrecurrence_layers: number of inrecurrence layers
      position_start_index: start index in embedding
      add_or_concat_timing_signal: bool,
      whether to add or concat the timing signal
      step_timing_signal_type: step timing signal type
      add_position_timing_signal: bool, whether to add position timing signal
      add_step_timing_signal: bool, whether to add step timing signal
      mix_with_transformer_before_ut: whether to mix transformer layers before
        ut.
      mix_with_transformer_after_ut: whether to mix transformer layers after ut.
      gates_inputs: controlling the cary/transform gate.
      gate_ffn_layer: gate ff layer type
      use_gated_transformer: whether to use gated transformer.
      gating_type: gating type.
    """
    self._layers = layers
    self._dropout_rate = dropout_rate
    self._norm_epsilon = norm_epsilon
    self.num_vanilla_transformer_layers = num_vanilla_transformer_layers
    self.act_type = act_type
    self.recurrence_type = recurrence_type
    self.act_max_steps = act_max_steps
    self.act_epsilon = act_epsilon
    self.num_rec_steps = num_rec_steps
    self.num_inrecurrence_layers = num_inrecurrence_layers
    self.position_start_index = position_start_index
    self.add_or_concat_timing_signal = add_or_concat_timing_signal
    self.step_timing_signal_type = step_timing_signal_type
    self.add_position_timing_signal = add_position_timing_signal
    self.add_step_timing_signal = add_step_timing_signal
    self.mix_with_transformer_before_ut = mix_with_transformer_before_ut
    self.mix_with_transformer_after_ut = mix_with_transformer_after_ut
    self.gates_inputs = gates_inputs
    self.gate_ffn_layer = gate_ffn_layer
    self.couple_carry_transform_gates = couple_carry_transform_gates
    self.use_gated_transformer = use_gated_transformer
    self.gating_type = gating_type

  def get_timing_signal_1d(self,
                           context,
                           length,
                           channels,
                           min_timescale=1.0,
                           max_timescale=1.0e4,
                           start_index=0):
    """Gets a bunch of sinusoids of different frequencies.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
      context: mtf context.
      length: a mtf.Dimension, length of timing signal sequence.
      channels: a mtf.Dimension, size of timing embeddings to create.
      The number of different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position

    Returns:
      a Tensor of timing signals [1, length, channels]
    """

    position = context.get_position() + start_index
    num_timescales = mtf.constant(context.mesh, channels.size // 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        mtf.maximum(num_timescales - 1, 1))
    channel_dim_name = channels.name
    inv_timescales = (
        min_timescale * mtf.exp(
            mtf.mtf_range(context.mesh,
                          mtf.Dimension(channel_dim_name, channels.size // 2),
                          context.activation_dtype) * -log_timescale_increment))

    scaled_time = position * inv_timescales
    # Please note that this slightly differs from the published paper.
    # See a discussion here:
    # https://github.com/tensorflow/tensor2tensor/pull/177
    #    concat_dim_name = scaled_time.shape.dimension_names[1]
    concat_dim_name = channels.name
    signal = mtf.concat(
        [mtf.sin(scaled_time), mtf.cos(scaled_time)],
        concat_dim_name=concat_dim_name)

    if channels.size % 2 != 0:
      raise NotImplementedError("Odd channel size not implemented.")
    new_dims = [mtf.Dimension("expanded", 1)
               ] + length.shape.dims + channels.shape.dim
    signal = mtf.reshape(signal, mtf.Shape(new_dims))
    return signal

  def add_position_timing_signal_func(self, context, x, step):
    """Add n-dimensional embedding as the position (horizontal) timing signal.

    Args:
      context: mtf context
      x: a tensor with shape [batch, length, depth]
      step: step

    Returns:
      a Tensor with the same shape as x.

    """

    if not self.position_start_index:
      index = 0

    elif self.position_start_index == "random":
      # Shift all positions randomly
      # TODO(dehghani): What would be reasonable for max number of shift?
      index = mtf.random_uniform(
          context.mesh, [], maxval=x.shape.dims[1].size, dtype=tf.int32)

    elif self.position_start_index == "step":
      # Shift positions based on the step
      if self.recurrence_type == "act":
        num_steps = self.act_max_steps
      else:
        num_steps = self.num_rec_steps
      index = mtf.cast(x.shape.dims[1].size * step / num_steps, dtype=tf.int32)

    length = context.length_dim
    channels = context.model.model_dim
    signal = self.get_timing_signal_1d(
        context, length, channels, start_index=index)

    if self.add_or_concat_timing_signal == "add":
      x_with_timing = x + mtf.cast(signal, x.dtype)
    # Unimplemented
    if self.add_or_concat_timing_signal == "concat":
      batch_dim = x.shape.dims[0]
      out_shape = mtf.Shape([batch_dim] + signal.shape.dims[1:])
      signal_tiled = mtf.broadcast(signal, out_shape)
      x_with_timing = mtf.concat(
          (x, signal_tiled), concat_dim_name=signal_tiled.dimension_names[-1])

    return x_with_timing

  def get_layer_timing_signal_learned_1d(self, context, channels, layer,
                                         num_layers):
    """get n-dimensional embedding as the layer (vertical) timing signal.

    Adds embeddings to represent the position of the layer in the tower.

    Args:
      context: mtf context
      channels: dimension of the timing signal
      layer: layer num
      num_layers: total number of layers

    Returns:
      a Tensor of timing signals [channels].
    """
    layer_dim = mtf.Dimension("layer", num_layers)
    shape = mtf.Shape([layer_dim, channels])
    layer_embedding = (
        mtf.get_variable(
            context.mesh,
            "layer_embedding",
            shape,
            dtype=context.variable_dtype,
            initializer=tf.random_normal_initializer(0, channels.size**-0.5)) *
        (channels.size**0.5))
    return mtf.gather(layer_embedding, layer, layer_dim)

  def add_step_timing_signal_func(self, context, x, step):
    """Add n-dimensional embedding as the step (vertical) timing signal.

    Args:
      context: mtf context
      x: a tensor with shape [batch, length, depth]
      step: step

    Returns:
      a Tensor with the same shape as x.

    """
    if self.recurrence_type == "act":
      num_steps = self.act_max_steps
    else:
      num_steps = self.num_rec_steps
    channels = x.shape.dims[-1]

    if self.step_timing_signal_type == "learned":
      signal = self.get_layer_timing_signal_learned_1d(context, channels, step,
                                                       num_steps)
    elif self.step_timing_signal_type == "sinusoid":
      signal = self.get_layer_timing_signal_sinusoid_1d(context, channels, step,
                                                        num_steps)
    if self.add_or_concat_timing_signal == "add":
      x_with_timing = x + mtf.cast(signal, x.dtype)
    elif self.add_or_concat_timing_signal == "concat":
      batch_dim = x.shape.dims[0]
      out_shape = mtf.Shape([batch_dim] + x.shape.dims[1:])
      signal_tiled = mtf.broadcast(signal, out_shape)
      x_with_timing = mtf.concat(
          (x, signal_tiled), concat_dim_name=signal_tiled.dimension_names[-1])

    return x_with_timing

  def step_preprocess(self, context, x, step):
    """Preprocess the input at the beginning of each step.

    Args:
      context: mtf context
      x: input tensor
      step: step

    Returns:
      preprocessed input.

    """
    original_channel_size = x.shape.dims[-1]

    if self.add_step_timing_signal:
      x = self.add_step_timing_signal_func(context, x, step)
    if ((self.add_position_timing_signal or self.add_position_timing_signal) and
        self.add_or_concat_timing_signal == "concat"):
      # linear projection to the original dimension of x
      new_dims = x.shape.dims[:-1] + [original_channel_size]
      x = mtf.layers.dense(
          x, variable_dtype=context.variable_dtype,
          new_dims=new_dims, activation=None, use_bias=False)
      # TODO(yanqiz): implement sru in a separate CL

    return x

  def vanilla_transformer_layer(self, context, x, mask):
    """Build a vanilla transformer layer."""

    for lnum, layer in enumerate(self._layers):
      scope_name = layer.name
      with tf.variable_scope(scope_name or ""):
        norm_x = self._layer_norm(context, (x * mask) if mask else x)
        with tf.variable_scope(layer.__class__.__name__):
          y = layer.call(context, norm_x)
          if y.shape != x.shape:
            raise ValueError("Layer %s returned misshaped output x=%s y=%s" %
                             (layer.__class__.__name__, x, y))
          if self.use_gated_transformer:
            y = self.gating(context, x, y, mask)
        x += self._dropout(context, y)
      if lnum != len(self._layers) - 1:
        context.layer_outputs.append(x)
      context.layer_index += 1
    return x

  def gating(self, context, x, transformed_x, mask):
    """Implementation of various gating layers."""
    gate_ffn_layer = self.gate_ffn_layer
    if self.gating_type == "highway":
      gate_inputs = [x]
      transform_gate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)
      carry_gate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)
      new_state = x * carry_gate + transformed_x * transform_gate
      return new_state
    elif self.gating_type == "gru":
      gate_inputs = [x, transformed_x]
      transition_function_update_gate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)
      transition_function_reset_gate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)

      reset_state = transition_function_reset_gate * x
      gate_inputs = [reset_state, transformed_x]
      transition_function_candidate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)

      transition_function_output = (
          (1 - transition_function_update_gate) * transformed_x +
          transition_function_update_gate * transition_function_candidate)
      return transition_function_output

  def ut_basic(self, context, x, mask):
    def ut_function(x, step):
      new_state = self.step_preprocess(context, x, step)
      for _ in range(self.num_inrecurrence_layers):
        new_state = self.vanilla_transformer_layer(context, new_state, mask)
      return new_state
    for i in range(self.num_rec_steps):
      x = ut_function(x, i)
    return x

  def act_layer(self, context, x, mask):
    """Build a Universal Transformer ACT layer."""
    state = x
    act_max_steps = self.act_max_steps
    threshold = 1.0 - self.act_epsilon
    state_shape_static = state.shape.dims

    state_slice = slice(0, 3)
    if self.act_type == "global":
      state_slice = slice(0, 2)

    # Dynamic shape for update tensors below
    update_shape = state_shape_static[state_slice]

    # Halting probabilities (p_t^n in the paper)
    halting_probability = mtf.zeros(
        context.mesh, update_shape, dtype=context.activation_dtype)

    # Remainders (R(t) in the paper)
    remainders = mtf.zeros(
        context.mesh, update_shape, dtype=context.activation_dtype)

    # Number of updates performed (N(t) in the paper)
    n_updates = mtf.zeros(
        context.mesh, update_shape, dtype=context.activation_dtype)

    # Previous cell states (s_t in the paper)
    previous_state = mtf.zeros_like(state)
    step = mtf.constant(context.mesh, 0, dtype=tf.int32)

    def ut_function(state, step, halting_probability, remainders, n_updates,
                    previous_state):
      """implements act (position-wise halting).

      Args:
        state: 3-D Tensor: [batch_size, length, channel]
        step: indicates number of steps taken so far
        halting_probability: halting probability
        remainders: act remainders
        n_updates: act n_updates
        previous_state: previous state

      Returns:
        transformed_state: transformed state
        step: step+1
        halting_probability: halting probability
        remainders: act remainders
        n_updates: act n_updates
        new_state: new state
      """
      state = self.step_preprocess(context, state, step)

      if self.act_type == "random":
        # random as halting probability
        p = mtf.random_uniform(
            context.mesh,
            shape=halting_probability.shape.dims,
            dtype=context.variable_dtype)
      else:
        last_dim_name = state.shape.dimension_names[-1]
        new_dims = [mtf.Dimension(last_dim_name, 1)]
        with tf.variable_scope(
            "sigmoid_activation_for_pondering", reuse=tf.AUTO_REUSE):
          p = mtf.layers.dense(
              state,
              variable_dtype=context.variable_dtype,
              reduced_dims=[state.shape.dims[-1]],
              new_dims=new_dims,
              activation=mtf.sigmoid,
              use_bias=True)
          if self.act_type == "global":
            # average over all positions (as a global halting prob)
            p = mtf.reduce_mean(p, reduced_dim=p.shape.dims[1])
            p = mtf.squeeze(p)
          else:
            # maintain position-wise probabilities
            new_shape = p.shape.dims[:-1]
            p = mtf.reshape(p, new_shape)
      # Mask for inputs which have not halted yet
      still_running = mtf.cast(
          mtf.less(halting_probability, 1.0), context.activation_dtype)

      # Mask of inputs which halted at this step
      new_halted = mtf.cast(
          mtf.greater(halting_probability + p * still_running, threshold),
          context.activation_dtype) * still_running
      # Mask of inputs which haven't halted, and didn't halt this step
      still_running = mtf.cast(
          mtf.less_equal(halting_probability + p * still_running, threshold),
          context.activation_dtype) * still_running

      # Add the halting probability for this step to the halting
      # probabilities for those input which haven't halted yet
      halting_probability += p * still_running

      # Compute remainders for the inputs which halted at this step
      remainders += new_halted * (1 - halting_probability)

      # Add the remainders to those inputs which halted at this step
      halting_probability += new_halted * remainders

      # Increment n_updates for all inputs which are still running
      n_updates += still_running + new_halted

      # Compute the weight to be applied to the new state and output
      # 0 when the input has already halted
      # p when the input hasn't halted yet
      # the remainders when it halted this step
      input_tensor = p * still_running + new_halted * remainders
      update_weights = input_tensor

      # apply transformation on the state
      transformed_state = state

      for _ in range(self.num_inrecurrence_layers):
        transformed_state = self.vanilla_transformer_layer(
            context, transformed_state, mask)

      # update running part in the weighted state and keep the rest
      new_state = ((transformed_state * update_weights) +
                   (previous_state * (1 - update_weights)))

      if self.act_type == "accumulated":
        # Add in the weighted state
        new_state = (transformed_state * update_weights) + previous_state

      step += 1

      return (transformed_state, step, halting_probability, remainders,
              n_updates, new_state)

    for _ in range(act_max_steps + 1):
      (state, step, halting_probability, remainders, n_updates,
       previous_state) = ut_function(state, step, halting_probability,
                                     remainders, n_updates, previous_state)
    ponder_times = n_updates

    mtf.scalar_summary("ponder_times", mtf.reduce_mean(ponder_times))
    return previous_state

  def ffn_layer_multi_inputs(self,
                             context,
                             mask,
                             inputs_list,
                             ffn_layer_type="dense",
                             kernel_initializer=None,
                             activation=None,
                             preprocess=False,
                             postprocess=False):
    """Implements a Feed-forward layer with multiple inputs, pad-removing, etc.

    Args:
      context: mtf context
      mask: mask
      inputs_list: list of input tensors
      ffn_layer_type: dense / dense_dropconnect/ dense_relu_dense
      kernel_initializer: kernel initializer
      activation: activation function
      preprocess: if preprocess the input --> default: layer-norm
      postprocess: if postprocess the output --> default: drop-out and residual

    Returns:
      a tensor
    Raises:
      ValueError: Unknown ffn_layer type.

    """

    # need at least one inputs
    num_inputs = len(inputs_list)
    assert num_inputs > 0

    if preprocess:
      # In case of having more than one input to the ffn,
      # we just apply layer norm on them independently as preprocessing
      for i, inputs in enumerate(inputs_list):
        inputs_list[i] = self._layer_norm(
            context, (inputs * mask) if mask else inputs)

    # the output size is the hidden size of the main inputs
    ffn_inputs = inputs_list[0]
    if len(inputs_list) != 1:
      ffn_inputs = mtf.concat(inputs_list, context.model.model_dim.name)
    if ffn_layer_type == "dense":
      # last_dims = [
      #     mtf.Dimension(ffn_inputs.shape.dims[-1].name, hidden_size)
      # ]
      output = mtf.layers.dense(
          ffn_inputs,
          reduced_dims=[ffn_inputs.shape.dims[-1]],
          new_dims=[context.model.model_dim],
          activation=activation,
          use_bias=True,
          variable_dtype=context.variable_dtype,
          expert_dims=context.model.ensemble_dims,
          kernel_initializer=kernel_initializer)
    elif ffn_layer_type == "dense_relu_dense":
      output = mtf.layers.dense_relu_dense(
          ffn_inputs,
          hidden_channels=context.model.model_dim,
          dropout=self.relu_dropout
      )

    else:
      raise ValueError("Unknown ffn_layer type: %s" % ffn_layer_type)

    if postprocess:
      output = self._layer_norm(context, (output * mask) if mask else output)

    return output

  def ut_highway(self, context, layer_inputs, mask):
    """A highway network layer."""
    def ut_function(x, step):
      """highway layer implementation."""
      state, inputs, memory = x
      new_state = self.step_preprocess(context, state, step)
      for _ in range(self.num_inrecurrence_layers):
        new_state = self.vanilla_transformer_layer(context, new_state, mask)
      transformed_state = new_state

      gate_inputs = []
      if "s" in self.gates_inputs:
        gate_inputs.append(state)
      if "t" in self.gates_inputs:
        gate_inputs.append(transformed_state)
      if "i" in self.gates_inputs:
        gate_inputs.append(inputs)
      gate_ffn_layer = self.gate_ffn_layer

      transform_gate = self.ffn_layer_multi_inputs(
          context,
          mask,
          gate_inputs,
          ffn_layer_type=gate_ffn_layer,
          activation=mtf.sigmoid,
          preprocess=True)
      if self.couple_carry_transform_gates:
        carry_gate = mtf.sub(1.0, transform_gate, name="carry")
      else:
        carry_gate = self.ffn_layer_multi_inputs(
            context,
            mask,
            gate_inputs,
            ffn_layer_type=gate_ffn_layer,
            activation=mtf.sigmoid,
            preprocess=True)
      new_state = state * carry_gate + transformed_state * transform_gate

      mtf.scalar_summary("highway_transform_gate_layer",
                         mtf.reduce_mean(transform_gate))
      mtf.scalar_summary("highway_carry_gate_layer",
                         mtf.reduce_mean(carry_gate))

      return new_state, inputs, memory
    for i in range(self.num_rec_steps):
      layer_inputs = ut_function(layer_inputs, i)
    output, _, _ = layer_inputs
    return output

  def call(self, context, x):
    """Call the layer stack."""
    if isinstance(context.sequence_id, mtf.Tensor):
      # We use this mask to zero out the padding regions at each layer.
      # This "fixes" a bug where extreme values leak from the padding into the
      # non-padding regions.
      # TODO(noam): understand this better and make a more principled fix.
      mask = mtf.cast(
          mtf.not_equal(context.sequence_id, 0), context.activation_dtype)
    else:
      mask = None
    x = self._dropout(context, x)
    context.layer_outputs.append(x)
    if self.mix_with_transformer_before_ut:
      for _ in range(self.num_vanilla_transformer_layers):
        x = self.vanilla_transformer_layer(context, x, mask)
    # Call a ACT layer
    if self.recurrence_type == "act":
      x = self.act_layer(context, x, mask)
    elif self.recurrence_type == "basic":
      x = self.ut_basic(context, x, mask)
    elif self.recurrence_type == "highway":
      layer_inputs = (x, x, x)
      x = self.ut_highway(context, layer_inputs, mask)
    if self.mix_with_transformer_after_ut:
      for _ in range(self.num_vanilla_transformer_layers):
        x = self.vanilla_transformer_layer(context, x, mask)
    x = self._layer_norm(context, x, name="final_layer_norm")
    x = self._dropout(context, x)
    if mask:
      x *= mask
    context.layer_outputs.append(x)
    return x

  def _dropout(self, context, x):
    if context.train and self._dropout_rate > 0:
      return mtf.dropout(
          x,
          rate=self._dropout_rate,
          noise_shape=mtf.Shape(context.batch_dims + [context.model.model_dim]))
    else:
      return x

  def _layer_norm(self, context, x, name=None):
    """Layer normalization.

    Deprecated - can we remove this?

    Args:
      context: a Context
      x: a Tensor
      name: an optional string

    Returns:
      a Tensor
    """
    return transformer.layer_norm(context, x, self._norm_epsilon, name)

  @property
  def num_layers(self):
    return len(self.layers)

  @property
  def layers(self):
    return self._layers
