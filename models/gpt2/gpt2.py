"""GPT-like model in Mesh-Tensorflow"""
import math

import mesh_tensorflow as mtf
import mesh_tensorflow.transformer as mtf_transformer
import tensorflow.compat.v1 as tf

from models.utils import parse_inputs

# --------------------------------------------------------------------------------
# LAYERS:

sentinel = object()


def rezero(x, scope, dtype):
    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        return x * g


def linear(x, scope, nf, *, w_init_stdev=0.02, variable_dtype, params=None, scale=False):
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
                             variable_dtype=variable_dtype,
                             )
        return c


def attn(x, scope, n_state, *, attention_type, params, bias, dim_seq, memory_length_dim, variable_dtype):
    # X Shape: [dim_batch, sequence_length, dim_embd]
    print(x.shape)
    x_shape, dim_batch, sequence_length, dim_embd, mesh = x.shape, *x.shape, x.mesh

    n_embd = params["n_embd"]

    dim_heads = mtf.Dimension("heads", params["n_head"])
    key_dim = mtf.Dimension("features_per_head", n_embd // params["n_head"])

    with tf.variable_scope(scope):
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

        with tf.variable_scope("attention"):
            logits = mtf.layers.us_einsum([q, k], reduced_dims=[key_dim])
            if bias is not None:
                logits += mtf.broadcast(bias, [dim_batch, dim_heads, bias.shape[-2], bias.shape[-1]])
            weights = mtf.softmax(logits, memory_length_dim)
            a = mtf.einsum([weights, v], q.shape)

        with tf.variable_scope("compute_output"):
            a = mtfparams.compute_output(a, x_shape)

        return a


def block(params, scope, layer_num, bias, width, height, sequence_dim, memory_length_dim, variable_dtype):
    use_moe = params["moe_layers"] is not None and layer_num in params["moe_layers"]
    use_mlp = params.get('use_mlp', True)

    def fn(x):
        with tf.variable_scope(scope):
            nx = x.shape[-1]  # Grab last dimension from input

            attention_type = params["attention_types"][layer_num]

            summed_a = None
            original_shape = list(x.shape)
            for idx, dim in enumerate([sequence_dim, width, height]):
                current_shape = original_shape.copy()
                if idx + 1 != 3:
                    current_shape[3], current_shape[idx + 1] = current_shape[idx + 1], current_shape[3]
                    inp = mtf.transpose(x, current_shape)
                batch = mtf.Dimension("big_batch",
                                      current_shape[0].size * current_shape[1].size * current_shape[2].size)
                inp = mtf.reshape(inp, [batch, current_shape[3], current_shape[4]])
                a = attn(inp, "attn", nx, attention_type=attention_type,
                         params=params, bias=bias if idx == 0 else None,
                         dim_seq=dim, memory_length_dim=dim,
                         variable_dtype=variable_dtype)
                a = mtf.reshape(a, current_shape)
                a = mtf.transpose(a, original_shape)
                summed_a = a if summed_a is None else (summed_a + a)

            x = x + rezero(a, "norm_rezero_1", dtype=variable_dtype)

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


def axial_positional_emb(embd_dim, mesh, params, variable_dtype):
    # Use axial position encoding
    axial_dim_1, axial_dim_2 = params["axial_pos_emb"]

    axial_dim = mtf.Dimension("axial_dim", axial_dim_1 * axial_dim_2)
    dim_axials = [mtf.Dimension(f"axial_dim_{i}", t) for i, t in enumerate((axial_dim_1, axial_dim_2))]

    axial_wpe_1 = mtf.get_variable(mesh, "axial_wpe_1", mtf.Shape([dim_axials[0], embd_dim]),
                                   initializer=tf.random_normal_initializer(stddev=0.01),
                                   master_dtype=variable_dtype.master_dtype,
                                   slice_dtype=variable_dtype.slice_dtype,
                                   activation_dtype=variable_dtype.activation_dtype)

    axial_wpe_2 = mtf.get_variable(mesh, "axial_wpe_2", mtf.Shape([dim_axials[1], embd_dim]),
                                   initializer=tf.random_normal_initializer(stddev=0.01),
                                   master_dtype=variable_dtype.master_dtype,
                                   slice_dtype=variable_dtype.slice_dtype,
                                   activation_dtype=variable_dtype.activation_dtype)

    axial_wpe_1, axial_wpe_2 = map(lambda t: mtf.broadcast(t, [dim_axials[0], dim_axials[1], embd_dim]),
                                   (axial_wpe_1, axial_wpe_2))
    wpe = (axial_wpe_1 + axial_wpe_2) / 2

    wpe = mtf.reshape(wpe, [axial_dim, embd_dim])

    return wpe


# --------------------------------------------------------------------------------
# MODEL:

def model(mtf_features, other_features, params, mesh, variable_dtype):
    """A GPT style model implemented in mesh tensorflow."""
    x, batch_dim, sequence_dim, width, height, color_channels, embd_dim, vocab_dim, embed_sequence_dim = parse_inputs(
            mtf_features, other_features)

    model_input = x

    h = model_input
    h += mtf.range(mesh, sequence_dim, tf.float32) / (sequence_dim.size * 2.)
    h += mtf.range(mesh, width, tf.float32) / (width.size * 2.)
    h += mtf.range(mesh, height, tf.float32) / (height.size * 2.)
    h -= 1.5

    aux_losses = 0  # instantiate auxiliary losses (for MOE models)

    for layer in range(params["n_layer"]):
        # attn blocks
        share_parameters = params["share_parameters"] is not None and params["share_parameters"] is True
        block_scope = f"h{layer}" if not share_parameters else ""

        block_fn = block(params=params, scope=block_scope, layer_num=layer,
                         bias=other_features["attn_bias"],
                         sequence_dim=sequence_dim,
                         width=width, height=height,
                         memory_length_dim=other_features["memory_length_dim"],
                         variable_dtype=variable_dtype)

        # If true and in train mode, enable gradient checkpointing
        recompute_grad = params["recompute_grad"] and (params["mode"] == "train") == True
        h, loss = block_fn(h) if not recompute_grad else mtf.recompute_grad(block_fn, [h])
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
