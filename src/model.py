import random
import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter


def rezero(block_input: tf.Tensor, dtype: mtf.VariableDType):
    with tf.variable_scope(f'rezero_{random.getrandbits(64):x}'):
        g = mtf.get_variable(block_input.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        block_input = block_input * g
    return block_input


def generic_feed_forward(block_input: mtf.Tensor,
                         reduced_dims: typing.List[mtf.Dimension],
                         new_dimensions: typing.List[mtf.Dimension],
                         dtype: typing.Union[mtf.VariableDType, tf.DType] = tf.float32,
                         dropout_rate: float = 0):
    intermediate_dimensions = [mtf.Dimension('_' + dim.name, dim.size) for dim in new_dimensions]
    with tf.variable_scope(f'feed_forward_{random.getrandbits(64):x}'):
        with tf.variable_scope('layer0'):
            weight0 = mtf.get_variable(block_input.mesh, "weight0", reduced_dims + intermediate_dimensions,
                                       initializer=tf.orthogonal_initializer(), dtype=dtype)
            block_input = mtf.einsum([block_input, weight0], block_input.shape - reduced_dims + intermediate_dimensions)
        if dropout_rate > 0:
            block_input = mtf.dropout(block_input, 1 - dropout_rate)
        block_input = block_input * mtf.tanh(block_input)  # LiSHT: https://arxiv.org/abs/1901.05894
        with tf.variable_scope('layer1'):
            weight1 = mtf.get_variable(block_input.mesh, "weight1", intermediate_dimensions + new_dimensions,
                                       initializer=tf.orthogonal_initializer(), dtype=dtype)
            block_input = mtf.einsum([block_input, weight1],
                                     block_input.shape - intermediate_dimensions + new_dimensions)
    return block_input


def model(mtf_features: dict, other_features: dict, params: ModelParameter, mesh: mtf.Mesh,
          variable_dtype: mtf.VariableDType):
    """A GPT style model implemented in mesh tensorflow."""
    x = mtf_features["inputs"]

    middle_dimensions = x.shape[1:-1]  # Ex: Shape[Sequence, Width, Height]
    input_features = x.shape[-1]

    dim_heads = mtf.Dimension("heads", params.n_head)
    key_dim = mtf.Dimension("features_per_head", params.n_embd // params.n_head)

    x = mtf.cast(x, tf.float32) / 255.
    tgt = x[:, 1:params.n_ctx // params.time_patch + 1]
    src = x[:, :params.n_ctx // params.time_patch]

    embedding = mtf.add_n([mtf.reshape(mtf.range(mesh, dim, dtype=tf.float32) / (dim.size - 1) / 3,
                                       (1,) + (1,) * idx + (dim.size,) + (1,) * (len(middle_dimensions) - idx))
                           for idx, dim in enumerate(middle_dimensions)])

    embedding = mtf.broadcast(embedding, [src.shape[:-1], mtf.Dimension(input_features.name, 1)])
    src = mtf.concat([src, embedding], input_features.name)

    output = generic_feed_forward(src, src.shape[-1:], [dim_heads, key_dim], tf.float32, params.dropout_rate)

    def _feed_forward(x):
        return generic_feed_forward(x, [dim_heads, key_dim], [dim_heads, key_dim], tf.float32, params.dropout_rate)

    for layer in range(params.n_layer):
        def _block_fn(block_input):
            with tf.variable_scope(f"attention_block{layer}"):
                summed_a = None

                for idx, dim in enumerate(middle_dimensions):
                    tmp_dim = mtf.Dimension(f'anonymous_{dim.name}', dim.size)

                    with tf.variable_scope(f"attn{dim.name}"):
                        q = _feed_forward(block_input)
                        k = _feed_forward(block_input)
                        v = _feed_forward(block_input)
                        k = mtf.rename_dimension(k, dim.name, tmp_dim.name)
                        v = mtf.rename_dimension(v, dim.name, tmp_dim.name)

                        with tf.variable_scope("attention"):
                            logits = mtf.einsum([q, k], q.shape - key_dim + tmp_dim) / tmp_dim.size ** 0.5
                            if idx == 0:
                                i = mtf.range(mesh, tmp_dim, tf.int32) + dim.size - tmp_dim.size
                                j = mtf.range(mesh, dim, tf.int32)
                                i = mtf.broadcast(i, [tmp_dim, dim])
                                j = mtf.broadcast(j, [tmp_dim, dim])
                                bias = mtf.cast(mtf.less(i, j), variable_dtype.activation_dtype) * -1e12
                                logits += mtf.broadcast(bias, logits.shape)
                            weights = mtf.softmax(logits, dim)
                            a = mtf.einsum([weights, v], q.shape)

                        a = _feed_forward(a)

                    summed_a = a if summed_a is None else (summed_a + a)

                block_input += rezero(summed_a, variable_dtype)
                block_input += rezero(_feed_forward(block_input), variable_dtype)

                return block_input

        output = mtf.recompute_grad(_block_fn, [output])
    output = generic_feed_forward(output, [dim_heads, key_dim], [input_features], tf.float32, params.dropout_rate)

    with tf.variable_scope("reduce_mean_final"):
        loss = mtf.reduce_mean(mtf.abs(output - tgt))

    return output, loss
