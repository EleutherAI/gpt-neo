import collections
import typing

import mesh_tensorflow as mtf
import mesh_tensorflow.transformer as mtf_transformer
import tensorflow.compat.v1 as tf

sentinel = object()


def rezero(x: tf.Tensor, scope: str, dtype: mtf.VariableDType):
    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        return x * g


def feed_forward(x: mtf.Tensor,
                 nf: float,
                 variable_dtype: typing.Union[mtf.VariableDType, tf.DType] = tf.float32,
                 dropout_rate: float = 0):
    x = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], use_bias=True,
                         kernel_initializer=tf.orthogonal_initializer(),
                         variable_dtype=variable_dtype, name="feed_forward0")
    if dropout_rate > 0:
        x = mtf.dropout(x, 1 - dropout_rate)
    x = mtf.gelu(x)
    x = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[nf], use_bias=True,
                         kernel_initializer=tf.orthogonal_initializer(),
                         variable_dtype=variable_dtype, name="feed_forward1")
    return x


# --------------------------------------------------------------------------------
# MODEL:

def model(mtf_features: dict, other_features: dict, params: collections.defaultdict, mesh: mtf.Mesh,
          variable_dtype: mtf.VariableDType):
    """A GPT style model implemented in mesh tensorflow."""

    x = mtf_features["inputs"]

    sequence_dim = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]
    dropout_rate = params['dropout_rate']

    embd_dim = other_features["embd_dim"]

    h = feed_forward(x, embd_dim, tf.float32, dropout_rate)

    for layer in range(params["n_layer"]):
        def block_fn(x):
            with tf.variable_scope(f"h{layer}"):
                summed_a = None

                for idx, dim in enumerate([sequence_dim, width, height]):
                    tmp_dim = mtf.Dimension(f'anonymous_{dim.name}', dim.size)
                    dim_heads = mtf.Dimension("heads", params["n_head"])
                    key_dim = mtf.Dimension("features_per_head", embd_dim.size // params["n_head"])

                    with tf.variable_scope(f"attn{dim.name}"):
                        mtfparams = mtf.transformer.attention.attention_params_simple(x.mesh,
                                                                                      io_dim=embd_dim,
                                                                                      kv_dim=key_dim,
                                                                                      heads_dim=dim_heads,
                                                                                      variable_dtype=variable_dtype
                                                                                      )
                        q = mtfparams.compute_q(x)
                        k = mtfparams.compute_k(x)
                        v = mtfparams.compute_v(x)
                        k = mtf.rename_dimension(k, dim.name, tmp_dim.name)
                        v = mtf.rename_dimension(v, dim.name, tmp_dim.name)

                        with tf.variable_scope("attention"):
                            logits = mtf.einsum([q, k], q.shape - key_dim + tmp_dim)
                            if idx == 0:
                                i = mtf.range(mesh, tmp_dim, tf.int32) + dim.size - tmp_dim.size
                                j = mtf.range(mesh, dim, tf.int32)
                                i, j = map(lambda t: mtf.broadcast(t, [tmp_dim, dim]), (i, j))
                                bias = mtf.cast(mtf.less(i, j), variable_dtype.activation_dtype) * -1e12
                                logits += mtf.broadcast(bias, logits.shape)
                            weights = mtf.softmax(logits, dim)
                            a = mtf.einsum([weights, v], q.shape)

                        with tf.variable_scope("compute_output"):
                            a = mtfparams.compute_output(a, x.shape)

                    summed_a = a if summed_a is None else (summed_a + a)

                x = x + rezero(summed_a, "norm_rezero_1", variable_dtype)

                with tf.variable_scope(f"h{layer}"):
                    m = feed_forward(x, embd_dim, variable_dtype=tf.float32)

                x = x + rezero(m, "norm_rezero_2", variable_dtype)
                return x

        h = mtf.recompute_grad(block_fn, [h])

    output = mtf.cast(h, tf.float32)

    with tf.variable_scope("reduce_mean_final"):
        loss = mtf.reduce_mean(mtf.abs(output - mtf_features["labels"]))

    loss = mtf.cast(loss, variable_dtype.slice_dtype)

    # Cast back to checkpoint dtype
    output = mtf.cast(output, variable_dtype.master_dtype)

    return output, loss
