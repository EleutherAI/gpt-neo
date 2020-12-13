import collections
import random
import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


def rezero(block_input: tf.Tensor, dtype: mtf.VariableDType):
    with tf.variable_scope(f'rezero_{random.getrandbits(64):x}'):
        g = mtf.get_variable(block_input.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        block_input = block_input * g
    return block_input


def generic_feed_forward(block_input: mtf.Tensor,
                         reduced_dims: typing.List[mtf.Dimension],
                         new_dimensions: typing.List[mtf.Dimension],
                         variable_dtype: typing.Union[mtf.VariableDType, tf.DType] = tf.float32,
                         dropout_rate: float = 0):
    with tf.variable_scope(f'feed_forward_{random.getrandbits(64):x}'):
        block_input = mtf.layers.dense(block_input, new_dims=new_dimensions, reduced_dims=reduced_dims, use_bias=True,
                                       kernel_initializer=tf.orthogonal_initializer(),
                                       variable_dtype=variable_dtype, name="dense0")
        if dropout_rate > 0:
            block_input = mtf.dropout(block_input, 1 - dropout_rate)
        block_input = mtf.gelu(block_input)
        block_input = mtf.layers.dense(block_input, new_dims=new_dimensions, reduced_dims=new_dimensions, use_bias=True,
                                       kernel_initializer=tf.orthogonal_initializer(),
                                       variable_dtype=variable_dtype, name="dense1")
    return block_input


def model(mtf_features: dict, other_features: dict, params: collections.defaultdict, mesh: mtf.Mesh,
          variable_dtype: mtf.VariableDType):
    """A GPT style model implemented in mesh tensorflow."""
    embd_dim = other_features["embd_dim"]
    x = mtf_features["inputs"]
    original_shape = x.shape
    sequence_dim = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]
    dropout_rate = params.get('dropout_rate', 0)

    dim_heads = mtf.Dimension("heads", params["n_head"])
    key_dim = mtf.Dimension("features_per_head", embd_dim.size // params["n_head"])

    output = generic_feed_forward(x, x.shape[-1:], [dim_heads, key_dim], tf.float32, dropout_rate)

    def _feed_forward(x):
        return generic_feed_forward(x, [dim_heads, key_dim], [dim_heads, key_dim], tf.float32, dropout_rate)

    for layer in range(params["n_layer"]):
        def _block_fn(block_input):
            with tf.variable_scope(f"attention_block{layer}"):
                summed_a = None

                for idx, dim in enumerate([sequence_dim, width, height]):
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
    output = generic_feed_forward(output, [dim_heads, key_dim], original_shape.shape[-1:], tf.float32, dropout_rate)
    output = mtf.reshape(output, original_shape)

    with tf.variable_scope("reduce_mean_final"):
        loss = mtf.reduce_mean(mtf.abs(output - mtf_features["labels"]))

    return output, loss
