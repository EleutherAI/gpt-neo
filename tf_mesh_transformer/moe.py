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

"""Mixture-of-experts code.

Interfaces and algorithms are under development and subject to rapid change
without notice.

TODO(noam): Remove the other copy of this code from tensor2tensor.
TODO(noam): Write a new, simpler, cleaner version of this code.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer

import tensorflow.compat.v1 as tf


@gin.configurable
class MoE1D(transformer.TransformerLayer):
  """Mixture of Experts Layer."""

  def __init__(self,
               num_experts=16,
               loss_coef=1e-2,
               hidden_size=4096,
               group_size=1024,
               capacity_factor_train=1.25,
               capacity_factor_eval=2.0,
               use_second_place_loss=False,
               second_policy_train="random",
               second_policy_eval="random",
               second_threshold_train=0.2,
               second_threshold_eval=0.2,
               dropout_rate=0.0,
               activation="relu"):
    self._hparams = HParams(
        moe_gating="top_2",
        moe_num_experts=num_experts,
        moe_loss_coef=loss_coef,
        moe_hidden_size=hidden_size,
        moe_group_size=group_size,
        moe_capacity_factor_train=capacity_factor_train,
        moe_capacity_factor_eval=capacity_factor_eval,
        moe_use_second_place_loss=use_second_place_loss,
        moe_second_policy_train=second_policy_train,
        moe_second_policy_eval=second_policy_eval,
        moe_second_threshold_train=second_threshold_train,
        moe_second_threshold_eval=second_threshold_eval,
        moe_dropout_rate=dropout_rate)
    self._activation = activation

  def call(self, context, x, losses=None):
    """Call the layer."""
    if context.model.ensemble_dim:
      raise NotImplementedError("MoE not yet implemented with ensembles")

    has_length_dim = context.length_dim in x.shape.dims
    if not has_length_dim:
      x_shape = x.shape
      shape_with_length = mtf.Shape(
          x_shape.dims[:-1] + [mtf.Dimension("length", 1)]
          + x_shape.dims[-1:])
      x = mtf.reshape(x, shape_with_length)
    y, loss = transformer_moe_layer_v1(
        x,
        context.model.model_dim,
        self._hparams,
        context.train,
        context.variable_dtype,
        layout=context.model.layout,
        mesh_shape=context.model.mesh_shape,
        nonpadding=context.nonpadding,
        activation=self._activation)
    if context.losses is not None:
      context.losses.append(loss)
    if not has_length_dim:
      y = mtf.reshape(y, x_shape)
    return y


class MoE2D(transformer.TransformerLayer):
  """Mixture of Experts Layer."""

  def __init__(self,
               expert_x=8,
               expert_y=8,
               loss_coef=1e-2,
               hidden_size=4096,
               group_size=1024,
               capacity_factor_train=1.25,
               capacity_factor_eval=2.0,
               capacity_factor_second_level=1.0,
               use_second_place_loss=False,
               second_policy_train="random",
               second_policy_eval="random",
               second_threshold_train=0.2,
               second_threshold_eval=0.2):
    self._hparams = HParams(
        moe_gating="top_2",
        moe_num_experts=[expert_x, expert_y],
        moe_loss_coef=loss_coef,
        moe_hidden_size=hidden_size,
        moe_group_size=group_size,
        moe_capacity_factor_train=capacity_factor_train,
        moe_capacity_factor_eval=capacity_factor_eval,
        moe_capacity_factor_second_level=capacity_factor_second_level,
        moe_use_second_place_loss=use_second_place_loss,
        moe_second_policy_train=second_policy_train,
        moe_second_policy_eval=second_policy_eval,
        moe_second_threshold_train=second_threshold_train,
        moe_second_threshold_eval=second_threshold_eval)

  def call(self, context, x, losses=None):
    """Call the layer."""
    if context.model.ensemble_dim:
      raise NotImplementedError("MoE not yet implemented with ensembles")
    has_length_dim = context.length_dim in x.shape.dims
    if not has_length_dim:
      x_shape = x.shape
      shape_with_length = mtf.Shape(
          x_shape.dims[:-1] + [mtf.Dimension("length", 1)]
          + x_shape.dims[-1:])
      x = mtf.reshape(x, shape_with_length)
    y, loss = transformer_moe_layer_v2(
        x,
        context.model.model_dim,
        self._hparams,
        context.train,
        context.variable_dtype,
        layout=context.model.layout,
        mesh_shape=context.model.mesh_shape,
        nonpadding=context.nonpadding)
    if context.losses is not None:
      context.losses.append(loss)
    if not has_length_dim:
      y = mtf.reshape(y, x_shape)
    return y


def transformer_moe_layer_v1(
    inputs, output_dim, hparams, train, variable_dtype,
    layout=None, mesh_shape=None, nonpadding=None, activation=mtf.relu):
  """Local mixture of experts that works well on TPU.

  Adapted from the paper https://arxiv.org/abs/1701.06538

  Note: until the algorithm and inferface solidify, we pass in a hyperparameters
  dictionary in order not to complicate the interface in mtf_transformer.py .
  Once this code moves out of "research", we should pass the hyperparameters
  separately.

  Hyperparameters used:
    hparams.moe_num_experts: number of experts
    hparams.moe_hidden_size: size of hidden layer in each expert
    hparams.moe_group_size: size of each "group" for gating purposes
    hparams.moe_capacity_factor_train: a float
    hparams.moe_capacity_factor_eval: a float
    hparams.moe_gating: a string
    + all hyperparmeters used by _top_2_gating()

  The number of parameters in the gating network is:
    (input_dim.size * hparams.num_experts) +

  The number of parameters in the experts themselves is:
    (hparams.num_experts
     * (input_dim.size + output_dim.size)
     * hparams.moe_hidden_size)

  The input is n-dimensional: [<batch_and_length_dims>, input_dim], consisting
  of the representations of all positions in a batch of sequences.

  Each position of each sequence is sent to 0-2 experts.  The expert
  choices and the combination weights are determined by a learned gating
  function.

  This function returns a small auxiliary loss that should be added to the
  training loss of the model.  This loss helps to balance expert usage.
  Without the loss, it is very likely that a few experts will be trained and
  the rest will starve.

  Several hacks are necessary to get around current TPU limitations:

  - To ensure static shapes, we enforce (by truncation/padding)
    that each sequence send the same number of elements to each expert.

    It would make more sense to enforce this equality over the entire batch,
    but due to our hacked-up gather-by-matmul implementation, we need to divide
    the batch into "groups".  For each group, the same number of elements
    are sent to each expert.

  TODO(noam): Factor this code better.  We want to be able to substitute
  different code for the experts themselves.

  Dimensions cheat sheet:
  B: batch dim(s)
  L: original sequence length
  M: input depth
  N: output depth
  G: number of groups
  S: group size
  E: number of experts
  C: expert capacity

  Args:
    inputs: a mtf.Tensor with shape [batch_dim(s), length_dim, input_dim]
    output_dim: a mtf.Dimension (for Transformer, this is input_dim)
    hparams: model hyperparameters
    train: a boolean
    variable_dtype: a mtf.VariableDType
    layout: optional - an input to mtf.convert_to_layout_rules
    mesh_shape: optional - an input to mtf.convert_to_shape
    nonpadding: an optional Tensor with shape [batch_dim(s), length_dim]
      and the same dtype as inputs, consisting of ones(nonpadding)
      and zeros(padding).
    activation: a function.

  Returns:
    outputs: a Tensor with shape [batch_dim(s), length_dim, output_dim]
    loss: a mtf scalar

  Raises:
    ValueError: on unrecognized hparams.moe_gating
  """
  # pylint: disable=line-too-long
  #
  # O outer_batch dimension can be used for expert replication, e.g.
  # outer_batch=4 for placing 128 experts on 512 cores with 4 replicas of each
  # expert.
  #
  # E.g. 16x16 basic example:
  #   moe_num_experts=512, num_groups=1024, batch=4096, length=256, d_model=1024
  # ---
  # Below ` indicates common way of splitting along mesh dimension.
  #
  # orig_inputs      OB`LM Tensor
  #                  Shape[outer_batch=1, batch=4096, length=256, d_model=1024]
  #                  v (reshaped)
  # inputs           OG`SM
  #                  Shape[outer_batch=1, batch=1024, group=1024, d_model=1024]
  #
  # combine_tensor,
  # dispatch_tensor  OG`SEC
  #                  Shape[outer_batch=1, batch=1024, group=1024, expert_unsplit=512, expert_capacity=4]
  #
  # (dispatched inputs)
  # expert_inputs    OEG`CM
  #                  Shape[outer_batch=1, expert_unsplit=512, batch=1024, expert_capacity=4, d_model=1024]
  #                  v (re-split via ReshapeOperation)
  #                  OE`GCM
  #                  Shape[outer_batch=1, experts=512, batch_unsplit=1024, expert_capacity=4, d_model=1024]
  #
  # (hidden representation)
  # h                OE`GCH
  #                  Shape[outer_batch=1, experts=512, batch_unsplit=1024, expert_capacity=4, expert_hidden=8192]
  #
  # expert_output    OE`GCM
  #                  Shape[outer_batch=1, experts=512, batch_unsplit=1024, expert_capacity=4, d_model=1024]
  #                  v (re-split via ReshapeOperation)
  #                  OEG`CM
  #                  Shape[outer_batch=1, expert_unsplit=512, batch=1024, expert_capacity=4, d_model=1024]
  #
  # (combined expert_output)
  # output           OG`SM
  #                  Shape[outer_batch=1, batch=1024, group=1024, d_model=1024
  #                  v (reshape)
  #                  OB`LM
  #                  Shape[outer_batch=1, batch=4096, length=256, d_model=1024]
  #
  # pylint: enable=line-too-long
  orig_inputs = inputs
  hidden_dim = mtf.Dimension("expert_hidden", hparams.moe_hidden_size)
  experts_dim = mtf.Dimension("experts", hparams.moe_num_experts)

  # We "cheat" here and look at the mesh shape and layout. This is to ensure
  # that the number of groups is a multiple of the mesh dimension
  # over which those groups are split.
  batch_and_length_dims, input_dim = (orig_inputs.shape.dims[:-1],
                                      orig_inputs.shape.dims[-1])
  # Hack: we assume that
  #   "outer_batch" == replication of experts
  #   mesh_dim_size can be derived from mesh_shape and orig_batch_dim
  #
  # We then reqire num_groups to be a multiple of mesh_dim_size.
  if orig_inputs.shape.dims[0].name == "outer_batch":
    outer_batch_dim, orig_batch_dim = orig_inputs.shape.dims[:2]
  else:
    outer_batch_dim, orig_batch_dim = (mtf.Dimension("outer_batch", 1),
                                       orig_inputs.shape.dims[0])

  # Number of MoE inputs (total number of position across batch_and_length_dims
  # per replica.
  n = 1
  for d in batch_and_length_dims:
    n *= d.size

  n = n // outer_batch_dim.size

  mesh_dim_size = mtf.tensor_dim_to_mesh_dim_size(layout, mesh_shape,
                                                  orig_batch_dim)
  num_groups, group_size = _split_into_groups(n, hparams.moe_group_size,
                                              mesh_dim_size)

  group_size_dim = mtf.Dimension("group", group_size)
  num_groups_dim = mtf.Dimension(orig_batch_dim.name, num_groups)

  moe_input_dims = [outer_batch_dim, num_groups_dim, group_size_dim, input_dim]
  # OGSM Tensor
  inputs = mtf.reshape(inputs, moe_input_dims)

  # Each sequence sends expert_capacity positions to each expert.
  if train:
    capacity_factor = hparams.moe_capacity_factor_train
  else:
    capacity_factor = hparams.moe_capacity_factor_eval
  expert_capacity = min(
      group_size_dim.size,
      int((group_size_dim.size * capacity_factor) / experts_dim.size))
  expert_capacity_dim = mtf.Dimension("expert_capacity", expert_capacity)

  experts_dim_unsplit = mtf.Dimension("expert_unsplit", experts_dim.size)
  batch_dim_unsplit = mtf.Dimension("batch_unsplit", num_groups_dim.size)
  if nonpadding is not None:
    nonpadding = mtf.zeros(
        inputs.mesh, batch_and_length_dims, dtype=inputs.dtype) + nonpadding
    nonpadding = mtf.reshape(nonpadding, moe_input_dims[:-1])
  if hparams.moe_gating == "top_2":
    # combine_tensor,
    # dispatch_tensor  OG`SEC Tensors
    # (G is generally split along mesh dim)
    dispatch_tensor, combine_tensor, loss = _top_2_gating(
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=experts_dim_unsplit,
        expert_capacity_dim=expert_capacity_dim,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=nonpadding)
  else:
    raise ValueError("unknown hparams.moe_gating=%s" % hparams.moe_gating)

  expert_inputs = mtf.einsum([inputs, dispatch_tensor],
                             mtf.Shape([
                                 outer_batch_dim, experts_dim_unsplit,
                                 num_groups_dim, expert_capacity_dim, input_dim
                             ]))

  expert_inputs = mtf.reshape(
      expert_inputs,
      mtf.Shape([
          outer_batch_dim, experts_dim, batch_dim_unsplit, expert_capacity_dim,
          input_dim
      ]))

  # Now feed the expert inputs through the experts.
  h = mtf.layers.dense_product(
      expert_inputs,
      reduced_dims=expert_inputs.shape.dims[-1:],
      new_dims=[hidden_dim],
      expert_dims=[experts_dim],
      activation_functions=activation, use_bias=False,
      variable_dtype=variable_dtype, name="wi")

  if train and hparams.moe_dropout_rate != 0.0:
    h = mtf.dropout(h, 1.0 - hparams.moe_dropout_rate)

  expert_output = mtf.layers.dense(
      h, output_dim, expert_dims=[experts_dim], use_bias=False,
      reduced_dims=h.shape.dims[-1:], variable_dtype=variable_dtype,
      name="wo")

  expert_output = mtf.reshape(
      expert_output,
      mtf.Shape([
          outer_batch_dim,
          experts_dim_unsplit,
          num_groups_dim,
          expert_capacity_dim,
          output_dim,
      ]))

  moe_output_dims = moe_input_dims[:-1] + [output_dim]
  output = mtf.einsum([expert_output, combine_tensor],
                      mtf.Shape(moe_output_dims))
  output = mtf.reshape(output, batch_and_length_dims + [output_dim])

  return output, loss * hparams.moe_loss_coef


def transformer_moe_layer_v2(
    inputs, output_dim, hparams, train, variable_dtype,
    layout=None, mesh_shape=None, nonpadding=None):
  """2-level mixture of experts.

  Adapted from the paper https://arxiv.org/abs/1701.06538

  Note: until the algorithm and inferface solidify, we pass in a hyperparameters
  dictionary in order not to complicate the interface in mtf_transformer.py .
  Once this code moves out of "research", we should pass the hyperparameters
  separately.

  Hyperparameters used:
    hparams.moe_num_experts: number of experts
    hparams.moe_hidden_size: size of hidden layer in each expert
    hparams.moe_group_size: size of each "group" for gating purposes
    hparams.moe_capacity_factor_train: a float
    hparams.moe_capacity_factor_eval: a float
    hparams.moe_capacity_factor_second_level: a float
    hparams.moe_gating: a string
    + all hyperparmeters used by _top_2_gating()

  One set of params for experts in first level and different of hparams
  per expert in the second level.
  The number of parameters in the gating network is:
    (input_dim.size * (hparams.num_experts) +
      (moe_hidden_size * hparams.num_experts) * hparams.num_experts


  The number of parameters in the experts themselves is:
    (hparams.num_experts
     * (input_dim.size + output_dim.size)
     * hparams.moe_hidden_size)

  The input is n-dimensional: [<batch_and_length_dims>, input_dim], consisting
  of the representations of all positions in a batch of sequences.

  Each position of each sequence is sent to 0-3 experts.  The expert
  choices and the combination weights are determined by a learned gating
  function.

  This function returns a small auxiliary loss that should be added to the
  training loss of the model.  This loss helps to balance expert usage.
  Without the loss, it is very likely that a few experts will be trained and
  the rest will starve.

  Several hacks are necessary to get around current TPU limitations:

  - To ensure static shapes, we enforce (by truncation/padding)
    that each sequence send the same number of elements to each expert.

    It would make more sense to enforce this equality over the entire batch,
    but due to our hacked-up gather-by-matmul implementation, we need to divide
    the batch into "groups".  For each group, the same number of elements
    are sent to each expert.

  TODO(noam): Factor this code better.  We want to be able to substitute
  different code for the experts themselves.

  Dimensions cheat sheet:
  a, b: batch size
  l: original sequence length
  m: input depth
  n: output depth
  g, h: number of groups
  s, t: group size
  x, y: number of experts
  c, d: expert capacity

  input: [a0, b1, l, m]
  input: [a0, g1, s, m]
  dispatch_tensor_x: [a0, g1, s, x, c]
  expert_input: [a0, g1, x, c, m]
  alltoall: [a0, g, x1, c, m]
  alltoall: [a0, g, x1, c, m]
  transpose: [x1, a0, g, c, m]
  reshape: [x1, h0, s, m]
  assignment2: [x1, h0, t, y, d]
  expert_input2: [x1, h0, y, d, m]
  alltoall: [x1, h, y0, d, m]
  ...
  reverse of that

  gating params 0: [m, x]
  gating params 1: [x1, m, y]

  expert params:
     [x1, y0, m, hidden]
     [x1, y0, hidden, n]

  Args:
    inputs: a mtf.Tensor with shape [a, b, l, m]
    output_dim: a mtf.Dimension (for Transformer, this is input_dim)
    hparams: model hyperparameters
    train: a boolean
    variable_dtype: a mtf.VariableDType
    layout: optional - an input to mtf.convert_to_layout_rules
    mesh_shape: optional - an input to mtf.convert_to_shape
    nonpadding: an optional mtf.Tensor with shape [a, b, l]
      and the same dtype as inputs, consisting of ones(nonpadding)
      and zeros(padding).

  Returns:
    outputs: a Tensor with shape [a, b, l, n]
    loss: a mtf scalar

  Raises:
    ValueError: on unrecognized hparams.moe_gating
  """
  if nonpadding is not None:
    nonpadding = mtf.zeros(inputs.mesh, inputs.shape.dims[:-1],
                           dtype=inputs.dtype) + nonpadding
  insert_outer_batch_dim = (len(inputs.shape.dims) == 3)
  if insert_outer_batch_dim:
    inputs = mtf.reshape(
        inputs, [mtf.Dimension("outer_batch", 1)] + inputs.shape.dims)

  assert len(hparams.moe_num_experts) == 2
  a0, b1, l, m = inputs.shape.dims
  hidden_dim = mtf.Dimension("expert_hidden", hparams.moe_hidden_size)
  x1 = mtf.Dimension("expert_x", hparams.moe_num_experts[0])
  y0 = mtf.Dimension("expert_y", hparams.moe_num_experts[1])
  x = mtf.Dimension("expert_x_unsplit", hparams.moe_num_experts[0])
  y = mtf.Dimension("expert_y_unsplit", hparams.moe_num_experts[1])
  n = output_dim

  # We "cheat" here and look at the mesh shape and layout. This is to ensure
  # that the number of groups (g.size) is a multiple of the mesh dimension
  # over which those groups are split.
  num_groups, group_size = _split_into_groups(
      b1.size * l.size, hparams.moe_group_size,
      mtf.tensor_dim_to_mesh_dim_size(layout, mesh_shape, b1))
  g1 = mtf.Dimension(b1.name, num_groups)
  g = mtf.Dimension(b1.name + "_unsplit", g1.size)
  s = mtf.Dimension("group_size_x", group_size)

  # Each sequence sends (at most?) expert_capacity positions to each expert.
  # Static expert_capacity dimension is needed for expert batch sizes
  if train:
    capacity_factor = hparams.moe_capacity_factor_train
  else:
    capacity_factor = hparams.moe_capacity_factor_eval
  expert_capacity = min(s.size, int((s.size * capacity_factor) / x.size))
  expert_capacity = max(expert_capacity, 4)
  c = mtf.Dimension("expert_capacity_x", expert_capacity)

  # We "cheat" here and look at the mesh shape and layout. This is to ensure
  # that the number of groups (h.size) is a multiple of the mesh dimension
  # over which those groups are split.
  num_groups, group_size = _split_into_groups(
      a0.size * g.size * c.size,
      hparams.moe_group_size,
      mtf.tensor_dim_to_mesh_dim_size(layout, mesh_shape, a0))
  t = mtf.Dimension("group_size_y", group_size)
  h0 = mtf.Dimension(a0.name, num_groups)
  h = mtf.Dimension(a0.name + "_unsplit", h0.size)

  expert_capacity = min(
      t.size,
      int((t.size * hparams.moe_capacity_factor_second_level) / y.size))
  expert_capacity = max(expert_capacity, 4)
  d = mtf.Dimension("expert_capacity_y", expert_capacity)

  # First level of expert routing
  # Reshape the inner batch size to a multiple of group_dim g1 and
  # group_size_dim s.
  inputs = mtf.reshape(inputs, [a0, g1, s, m])
  if nonpadding is not None:
    nonpadding = mtf.reshape(nonpadding, [a0, g1, s])

  # Get the assignments for the first level.
  # dispatch_tensor_x has shape [a0, g1, s, x, c]
  if hparams.moe_gating == "top_2":
    dispatch_tensor_x, combine_tensor_x, loss_outer = _top_2_gating(
        inputs=inputs,
        outer_expert_dims=None,
        experts_dim=x,
        expert_capacity_dim=c,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        name="outer_gating",
        importance=nonpadding)
  else:
    raise ValueError("unknown hparams.moe_gating=%s" % hparams.moe_gating)

  # Now create expert_inputs based on the assignments.
  # put num_experts dimension first to make split easier in alltoall
  expert_inputs_x = mtf.einsum([inputs, dispatch_tensor_x], [x, a0, g1, c, m])

  # we construct an "importance" Tensor for the inputs to the second-level
  # gating.  The importance of an input is 1.0 if it represents the
  # first-choice expert-group and 0.5 if it represents the second-choice expert
  # group.  This is used by the second-level gating.
  importance = mtf.reduce_sum(combine_tensor_x, output_shape=[x, a0, g1, c])
  importance = 0.5 * (
      mtf.to_float(mtf.greater(importance, 0.5)) +
      mtf.to_float(mtf.greater(importance, 0.0)))

  # First level, all to all. Here we change the split dimension from g1 to x1.
  expert_inputs_x = mtf.reshape(expert_inputs_x, mtf.Shape(
      [x1, a0, g, c, m]))
  importance = mtf.reshape(importance, [x1, a0, g, c])

  # Second level of expert routing
  # Reshape the expert_inputs outer batch dim to be a multiple of group_dim h0
  # and group_size_dim t.
  inputs_y = mtf.reshape(expert_inputs_x, [x1, h0, t, m])
  importance = mtf.reshape(importance, [x1, h0, t])

  # Get the assignments for the second level.
  # dispatch_tensor_y has shape [x1, h0, t, y, d]
  if hparams.moe_gating == "top_2":
    dispatch_tensor_y, combine_tensor_y, loss_inner = _top_2_gating(
        inputs=inputs_y,
        outer_expert_dims=[x1],
        experts_dim=y,
        expert_capacity_dim=d,
        hparams=hparams,
        train=train,
        variable_dtype=variable_dtype,
        importance=importance,
        name="inner_gating")
  else:
    raise ValueError("unknown hparams.moe_gating=%s" % hparams.moe_gating)

  # Now create expert_inputs based on the assignments.
  # put num_experts dimension first to make split easier in alltoall
  expert_inputs_y = mtf.einsum([inputs_y, dispatch_tensor_y], [y, x1, h0, d, m])

  # Second level, all to all. Here we change the split dimension from h0 to y0.
  expert_inputs_y = mtf.reshape(expert_inputs_y, mtf.Shape(
      [y0, x1, h, d, m]))

  hidden_output = mtf.layers.dense(
      expert_inputs_y, hidden_dim, expert_dims=[y0, x1],
      reduced_dims=expert_inputs_y.shape.dims[-1:],
      activation=mtf.relu, use_bias=False, variable_dtype=variable_dtype,
      name="wi")
  expert_output = mtf.layers.dense(
      hidden_output, output_dim, expert_dims=[y0, x1],
      reduced_dims=hidden_output.shape.dims[-1:],
      use_bias=False, variable_dtype=variable_dtype,
      name="wo")

  # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
  # expert_output has shape [y0, x1, h, d, n]

  # alltoall
  expert_output = mtf.reshape(expert_output, mtf.Shape(
      [y, x1, h0, d, n]))

  # combine results from inner level
  output_y = mtf.einsum([expert_output, combine_tensor_y], [x1, h0, t, n])

  # Reshape the combined tensor from inner level to now contain outer_batch_dim
  # a0 and group_dim g
  output = mtf.reshape(output_y, [x1, a0, g, c, n])

  # alltoall from expert_dim x to group_dim g1
  expert_output_x = mtf.reshape(output, mtf.Shape([x, a0, g1, c, n]))

  # combine results from outer level
  output_x = mtf.einsum([expert_output_x, combine_tensor_x], [a0, g1, s, n])

  # Reshape the combined tensor to now contain inner_batch_dim
  # b1 and the original sequence length
  output = mtf.reshape(output_x, [a0, b1, l, n])
  if insert_outer_batch_dim:
    output = mtf.reshape(output, [b1, l, n])
  return output, (loss_outer + loss_inner) * hparams.moe_loss_coef


def _top_2_gating(
    inputs, outer_expert_dims, experts_dim, expert_capacity_dim,
    hparams, train, variable_dtype, importance=None, name="top_2_gating"):
  """Compute gating for mixture-of-experts in TensorFlow.

  Note: until the algorithm and inferface solidify, we pass in a hyperparameters
  dictionary in order not to complicate the interface in mtf_transformer.py .
  Once this code moves out of "research", we should pass the hyperparameters
  separately.

  Hyperparameters used:
    hparams.moe_use_second_place_loss: a boolean
    hparams.moe_second_policy_train: a string
    hparams.moe_second_policy_eval: a string
    hparams.moe_second_threshold: a float

  The returned forward assignment is a tensor used to map (via einsum) from the
  inputs to the expert_inputs.  Likewise, the returned combine_tensor is
  used to map (via einsum) from the expert outputs to the outputs.  Both the
  forward and backward assignments are mostly zeros.  The shapes of the tensors
  are as follows.

  inputs: [<batch_dims>, group_size_dim, input_dim]
  importance: [<batch_dims>, group_size_dim]
  dispatch_tensor:
    [<batch_dims>, group_size_dim, experts_dim, expert_capacity_dim]
  expert_inputs:
    [<batch_dims>, experts_dim, expert_capacity_dim, input_dim]

  expert_outputs: [<batch_dims>, experts_dim, expert_capacity_dim, output_dim]
  combine_tensor:
    [<batch_dims>, group_size_dim, experts_dim, expert_capacity_dim]
  outputs: [<batch_dims>, group_size_dim, output_dim]

  "importance" is an optional tensor with one floating-point value for each
  input vector.  If the importance of an input is 1.0, then we send it to
  up to 2 experts.  If 0.0 < importance < 1.0, then we send it to at most
  one expert.  If importance == 0.0, then we send it to no experts.

  We use "importance" at the second-level gating function of a hierarchical
  mixture of experts.  Inputs to the first-choice expert-group get importance
  1.0.  Inputs to the second-choice expert group get importance 0.5.
  Inputs that represent padding get importance 0.0.

  Args:
    inputs: a mtf.Tensor with shape [<batch_dims>, group_size_dim, input_dim]
    outer_expert_dims: an optional list of dimensions.  This is for the case
      where we are at an inner level of a hierarchical MoE.
    experts_dim: a Dimension (the number of experts)
    expert_capacity_dim: a Dimension (number of examples per group per expert)
    hparams: model hyperparameters.
    train: a boolean
    variable_dtype: a mtf.VariableDType
    importance: an optional tensor with shape [<batch_dims>, group_size_dim]
    name: an optional string

  Returns:
    dispatch_tensor: a Tensor with shape
      [<batch_dims>, group_size_dim, experts_dim, expert_capacity_dim]
    combine_tensor: a Tensor with shape
      [<batch_dims>, group_size_dim, experts_dim, expert_capacity_dim]
    loss: a mtf scalar

  Raises:
    ValueError: on illegal hyperparameters
  """
  group_size_dim, unused_input_dim = inputs.shape.dims[-2:]

  raw_gates = mtf.layers.dense(
      inputs, experts_dim, use_bias=False,
      expert_dims=outer_expert_dims,
      variable_dtype=variable_dtype,
      name=name)
  raw_gates = mtf.softmax(raw_gates, experts_dim)

  # The internals of this function run in float32.
  #   bfloat16 seems to reduce quality.
  raw_gates = mtf.to_float(raw_gates)

  expert_capacity_f = float(expert_capacity_dim.size)

  # FIND TOP 2 EXPERTS PER POSITON
  # Find the top expert for each position. shape=[batch, group]
  gate_1, index_1 = mtf.top_1(raw_gates, experts_dim)
  # [batch, group, experts]
  mask_1 = mtf.one_hot(index_1, experts_dim, dtype=raw_gates.dtype)
  density_1_proxy = raw_gates
  if importance is not None:
    mask_1 *= mtf.to_float(mtf.equal(importance, 1.0))
    gate_1 *= mtf.to_float(mtf.equal(importance, 1.0))
    density_1_proxy *= mtf.to_float(mtf.equal(importance, 1.0))
  gates_without_top_1 = raw_gates * (1.0 - mask_1)
  # [batch, group]
  gate_2, index_2 = mtf.top_1(gates_without_top_1, experts_dim)
  # [batch, group, experts]
  mask_2 = mtf.one_hot(index_2, experts_dim, dtype=raw_gates.dtype)
  if importance is not None:
    mask_2 *= mtf.to_float(mtf.greater(importance, 0.0))

  denom = gate_1 + gate_2 + 1e-9
  gate_1 /= denom
  gate_2 /= denom

  # BALANCING LOSSES
  # shape = [batch, experts]
  # We want to equalize the fraction of the batch assigned to each expert
  density_1 = mtf.reduce_mean(mask_1, reduced_dim=group_size_dim)
  # Something continuous that is correlated with what we want to equalize.
  density_1_proxy = mtf.reduce_mean(density_1_proxy, reduced_dim=group_size_dim)
  loss = (mtf.reduce_mean(density_1_proxy * density_1)
          * float(experts_dim.size * experts_dim.size))

  if hparams.moe_use_second_place_loss:
    # Also add a loss to encourage all experts to be used equally also as the
    # second-place expert.  Experimentally, this seems to be a wash.
    # We want to equalize the fraction of the batch assigned to each expert:
    density_2 = mtf.reduce_mean(mask_2, reduced_dim=group_size_dim)
    # As a proxy for density_2, we renormalize the raw gates after the top one
    # has been removed.
    normalized = gates_without_top_1 / (
        mtf.reduce_sum(gates_without_top_1, reduced_dim=experts_dim) + 1e-9)
    density_2_proxy = mtf.reduce_mean(normalized, reduced_dim=group_size_dim)
    loss_2 = (mtf.reduce_mean(density_2_proxy * density_2)
              * float(experts_dim.size * experts_dim.size))
    loss += loss_2 * 0.5

  # Depending on the policy in the hparams, we may drop out some of the
  # second-place experts.
  if train:
    policy = hparams.moe_second_policy_train
    threshold = hparams.moe_second_threshold_train
  else:
    policy = hparams.moe_second_policy_eval
    threshold = hparams.moe_second_threshold_eval
  if policy == "all":
    # Use second-place experts for all examples.
    pass
  elif policy == "none":
    # Never use second-place experts for all examples.
    mask_2 = mtf.zeros_like(mask_2)
  elif policy == "threshold":
    # Use second-place experts if gate_2 > threshold.
    mask_2 *= mtf.to_float(mtf.greater(gate_2, threshold))
  elif policy == "random":
    # Use second-place experts with probablity min(1.0, gate_2 / threshold).
    mask_2 *= mtf.to_float(
        mtf.less(mtf.random_uniform(gate_2.mesh, gate_2.shape),
                 gate_2 / max(threshold, 1e-9)))
  else:
    raise ValueError("Unknown policy %s" % policy)

  # COMPUTE ASSIGNMENT TO EXPERTS
  # [batch, group, experts]
  # This is the position within the expert's mini-batch for this sequence
  position_in_expert_1 = mtf.cumsum(
      mask_1, group_size_dim, exclusive=True) * mask_1
  # Remove the elements that don't fit. [batch, group, experts]
  mask_1 *= mtf.to_float(mtf.less(position_in_expert_1, expert_capacity_f))
  # [batch, experts]
  # How many examples in this sequence go to this expert
  mask_1_count = mtf.reduce_sum(mask_1, reduced_dim=group_size_dim)
  # [batch, group] - mostly ones, but zeros where something didn't fit
  mask_1_flat = mtf.reduce_sum(mask_1, reduced_dim=experts_dim)
  # [batch, group]
  position_in_expert_1 = mtf.reduce_sum(
      position_in_expert_1, reduced_dim=experts_dim)
  # Weight assigned to first expert.  [batch, group]
  gate_1 *= mask_1_flat

  # [batch, group, experts]
  position_in_expert_2 = (
      mtf.cumsum(mask_2, group_size_dim, exclusive=True) + mask_1_count)
  position_in_expert_2 *= mask_2
  mask_2 *= mtf.to_float(mtf.less(position_in_expert_2, expert_capacity_f))
  # mask_2_count = mtf.reduce_sum(mask_2, reduced_dim=experts_dim)
  mask_2_flat = mtf.reduce_sum(mask_2, reduced_dim=experts_dim)
  gate_2 *= mask_2_flat
  position_in_expert_2 = mtf.reduce_sum(
      position_in_expert_2, reduced_dim=experts_dim)

  # [batch, group, experts, expert_capacity]
  combine_tensor = (
      gate_1 * mask_1_flat
      * mtf.one_hot(index_1, experts_dim)
      * mtf.one_hot(mtf.to_int32(position_in_expert_1), expert_capacity_dim) +
      gate_2 * mask_2_flat
      * mtf.one_hot(index_2, experts_dim)
      * mtf.one_hot(mtf.to_int32(position_in_expert_2), expert_capacity_dim))

  combine_tensor = mtf.cast(combine_tensor, inputs.dtype)
  loss = mtf.cast(loss, inputs.dtype)

  dispatch_tensor = mtf.cast(
      mtf.cast(combine_tensor, tf.bool), combine_tensor.dtype)

  return dispatch_tensor, combine_tensor, loss


def set_default_moe_hparams(hparams):
  """Add necessary hyperparameters for mixture-of-experts."""
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-2
  hparams.add_hparam("moe_gating", "top_2")
  # Experts have fixed capacity per batch.  We need some extra capacity
  # in case gating is not perfectly balanced.
  # moe_capacity_factor_* should be set to a value >=1.
  hparams.add_hparam("moe_capacity_factor_train", 1.25)
  hparams.add_hparam("moe_capacity_factor_eval", 2.0)
  hparams.add_hparam("moe_capacity_factor_second_level", 1.0)
  # Each expert has a hidden layer with this size.
  hparams.add_hparam("moe_hidden_size", 4096)
  # For gating, divide inputs into groups of this size before gating.
  # Each group sends the same number of inputs to each expert.
  # Ideally, the group size would be the whole batch, but this is expensive
  # due to our use of matrix multiplication for reordering.
  hparams.add_hparam("moe_group_size", 1024)
  # For top_2 gating, whether to impose an additional loss in order to make
  # the experts equally used as the second-place expert.
  hparams.add_hparam("moe_use_second_place_loss", 0)
  # In top_2 gating, policy for whether to use a second-place expert.
  # Legal values are:
  #    "all": always
  #    "none": never
  #    "threshold": if gate value > the given threshold
  #    "random": if gate value > threshold*random_uniform(0,1)
  hparams.add_hparam("moe_second_policy_train", "random")
  hparams.add_hparam("moe_second_policy_eval", "random")
  hparams.add_hparam("moe_second_threshold_train", 0.2)
  hparams.add_hparam("moe_second_threshold_eval", 0.2)


def _split_into_groups(n, max_group_size, mesh_dim_size):
  """Helper function for figuring out how to split a dimension into groups.

  We have a dimension with size n and we want to split it into
  two dimensions: n = num_groups * group_size

  group_size should be the largest possible value meeting the constraints:
    group_size <= max_group_size
    (num_groups = n/group_size) is a multiple of mesh_dim_size

  Args:
    n: an integer
    max_group_size: an integer
    mesh_dim_size: an integer

  Returns:
    num_groups: an integer
    group_size: an integer

  Raises:
    ValueError: if n is not a multiple of mesh_dim_size
  """
  if n % mesh_dim_size != 0:
    raise ValueError(
        "n=%d is not a multiple of mesh_dim_size=%d" % (n, mesh_dim_size))
  num_groups = max(1, n // max_group_size)
  while (num_groups % mesh_dim_size != 0 or n % num_groups != 0):
    num_groups += 1
  group_size = n // num_groups
  tf.logging.info(
      "_split_into_groups(n=%d, max_group_size=%d, mesh_dim_size=%d)"
      " = (num_groups=%d group_size=%d)" %
      (n, max_group_size, mesh_dim_size, num_groups, group_size))
  return num_groups, group_size


class HParams(object):
  """Replacement for tf.contrib.training.HParams.

  TODO(noam): remove this class and rewrite the methods in this file.
  """

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def add_hparam(self, k, v):
    setattr(self, k, v)
