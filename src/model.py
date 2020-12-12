import math

import mesh_tensorflow.transformer as mtf_transformer
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import collections


sentinel = object()


def biasmask_attn_weights(mesh: mtf.Mesh, nd: mtf.Dimension, ns: mtf.Dimension, variable_dtype: mtf.VariableDType):
    '''
    The old mask_attn_weights applied directly to the QK;
    this returns a bias that the attention code from mtf adds to the attention matrix.
    w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    n_src and n_dest are both the same, i.e equal to sequence length
    We rename ns because we want bias to have shape [batch, heads, memory_length, sequence] to match up with QK^T
    Information flows from k and v (memory_length) to q (sequence)
    '''

    i = mtf.range(mesh, nd, tf.int32) + ns.size - nd.size
    j = mtf.range(mesh, ns, tf.int32)
    i, j = map(lambda t: mtf.broadcast(t, [nd, ns]), (i, j))
    dtype = variable_dtype.activation_dtype

    return mtf.cast(mtf.less(i, j), dtype) * -1e10


def parse_inputs(mtf_features: dict, other_features: dict):
    '''
    Parse inputs and labels from the mtf_features / other_features input dicts
    All dimensions are defined inside model_fn for efficiency
    '''

    x = mtf_features["inputs"]

    batch_dim = x.shape[0]
    sequence_dim = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]
    color_channels = x.shape[4]
    embd_dim = other_features["embd_dim"]
    vocab_dim = other_features["vocab_dim"]
    embed_sequence_dim = other_features["embed_sequence_dim"]

    return x, batch_dim, sequence_dim, width, height, color_channels, embd_dim, vocab_dim, embed_sequence_dim


def rezero(x: tf.Tensor, scope: str, dtype: mtf.VariableDType):
    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        return x * g


def linear(x: mtf.Tensor, scope: str, nf: float, *, w_init_stdev: float = 0.02, variable_dtype: tf.DType = None,
           params: collections.defaultdict = None,
           scale=False):

    # nf = number of features
    if params["scale_by_depth"] and scale:
        # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))

    if params["scale_by_in"]:  # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].size))  # Dimension is a namedtuple of (name, size)

    # Not in the variable_scope because mtf already has a variable_scope in it
    with tf.variable_scope("conv1d_main"):
        c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                             variable_dtype=variable_dtype, )
        return c


def block(params: collections.defaultdict, scope: str, layer_num: int, bias: mtf.Tensor, width: mtf.Dimension,
          height: mtf.Dimension, sequence_dim: mtf.Dimension, memory_length_dim: mtf.Dimension,
          variable_dtype: mtf.VariableDType):

    use_moe = params["moe_layers"] is not None and layer_num in params["moe_layers"]
    use_mlp = params.get('use_mlp', True)

    def fn(x):
        dim_embd = x.shape[4]

        with tf.variable_scope(scope):
            summed_a = None

            for idx, dim in enumerate([sequence_dim, width, height]):
                tmp_dim = mtf.Dimension(f'anonymous_{dim.name}', dim.size)

                dim_heads = mtf.Dimension("heads", params["n_head"])
                key_dim = mtf.Dimension("features_per_head", dim_embd.size // params["n_head"])

                with tf.variable_scope(f"attn{dim.name}"):

                    # Compute attention inputs
                    mtfparams = mtf.transformer.attention.attention_params_simple(x.mesh,
                                                                                  io_dim=dim_embd,
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
                            logits += mtf.broadcast(biasmask_attn_weights(x.mesh, tmp_dim, dim, variable_dtype),
                                                    logits.shape)
                        weights = mtf.softmax(logits, dim)
                        a = mtf.einsum([weights, v], q.shape)

                    with tf.variable_scope("compute_output"):
                        a = mtfparams.compute_output(a, x.shape)

                summed_a = a if summed_a is None else (summed_a + a)

            x = x + rezero(summed_a, "norm_rezero_1", dtype=variable_dtype)

            if use_moe:
                moe_params = mtf.transformer.moe.HParams()
                mtf.transformer.moe.set_default_moe_hparams(moe_params)

                # Override defaults
                for k, v in params["moe_params"].items():
                    moe_params.add_hparam(k, v)

                moe_train = params["mode"] == "train"

                m, aux_loss = mtf.transformer.moe.transformer_moe_layer_v1(x, x.shape[-1], moe_params,
                                                                           train=moe_train,
                                                                           mesh_shape=params["mesh_shape"],
                                                                           layout=params["layout"],
                                                                           variable_dtype=variable_dtype)
            else:
                if use_mlp:
                    with tf.variable_scope(scope):
                        nx = x.shape[-1]
                        h = mtf.gelu(linear(x, "c_fc", mtf.Dimension("intermediate_expanded", nx.size * 4),
                                            variable_dtype=variable_dtype, params=params))

                        m = linear(h, "c_proj", nx, variable_dtype=variable_dtype, params=params, scale=True)
                        if params["mode"] == "train" and params["res_dropout"] > 0:
                            m = mtf.dropout(m, rate=params["res_dropout"], name="mlp_dropout")
                else:
                    m = x
                aux_loss = mtf.zeros(x.mesh, mtf.Shape([]), dtype=variable_dtype.slice_dtype)

            x = x + rezero(m, "norm_rezero_2", variable_dtype)
            return x, aux_loss

    return fn


# --------------------------------------------------------------------------------
# MODEL:

def model(mtf_features: dict, other_features: dict, params: collections.defaultdict, mesh: mtf.Mesh,
          variable_dtype: mtf.VariableDType):
    """A GPT style model implemented in mesh tensorflow."""

    x, batch_dim, sequence_dim, width, height, color_channels, embd_dim, vocab_dim, embed_sequence_dim = parse_inputs(
        mtf_features, other_features)

    # add positional embeddings to the input tensor.
    h = x
    h += mtf.range(mesh, sequence_dim, tf.float32) / (sequence_dim.size * 2.)
    h += mtf.range(mesh, width, tf.float32) / (width.size * 2.)
    h += mtf.range(mesh, height, tf.float32) / (height.size * 2.)
    h -= 1.5

    # instantiate auxiliary losses (for MOE models)
    aux_losses = 0

    # Initial linear projection.
    h = linear(h, "input_projection", embd_dim, variable_dtype=tf.float32, params=params, scale=True)

    for layer in range(params["n_layer"]):
        # attn blocks
        block_fn = block(params=params, scope=f"h{layer}", layer_num=layer,
                         bias=other_features["attn_bias"],
                         sequence_dim=sequence_dim,
                         width=width, height=height,
                         memory_length_dim=other_features["memory_length_dim"],
                         variable_dtype=variable_dtype)

        # If true and in train mode, enable gradient checkpointing
        h, loss = mtf.recompute_grad(block_fn, [h])
        aux_losses += loss

    output = h

    if params["mode"] == "train":
        labels = mtf_features["labels"]
        output = mtf.cast(output, tf.float32)

        with tf.variable_scope("reduce_mean_final"):
            loss = mtf.reduce_mean(mtf.abs(output - labels))

        loss += aux_losses  # Add on auxiliary losses (currently only used for MoE)
        loss = mtf.cast(loss, variable_dtype.slice_dtype)
    else:
        loss = None

    # Cast back to checkpoint dtype
    output = mtf.cast(output, variable_dtype.master_dtype)

    return output, loss
