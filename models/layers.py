import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import math
import mesh_tensorflow.transformer as mtf_transformer

from models.activations import get_activation_fn


# --------------------------------------------------------------------------------
# LAYERS:

sentinel = object()


def exists(x):
    return x is not None


def identity(x, *args, **kwargs):
    return x


def is_incremental_inference(context):
    return exists(context) and context.mode == "incremental"


def norm(x, axis, epsilon=1e-8):
    x -= mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
    s = mtf.reduce_mean(mtf.square(x), reduced_dim=axis, name="norm_reduce_mean_s")
    return x * mtf.rsqrt(s + epsilon)


def rezero(x, scope, dtype):
    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, "g", [], initializer=tf.constant_initializer(0), dtype=dtype)
        return x * g


def scale_norm(x, scope, *, variable_dtype, axis=sentinel, epsilon=1e-5, params=None):
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, "g", [], initializer=tf.constant_initializer(1),
                             master_dtype=variable_dtype.master_dtype,
                             slice_dtype=variable_dtype.slice_dtype,
                             activation_dtype=variable_dtype.activation_dtype)

        x = norm(x, axis, epsilon)
        x = x * g
        return x


def layer_norm(x, scope, *, variable_dtype, axis=sentinel, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        n_state = x.shape[-1]

        g = mtf.get_variable(x.mesh, "g", [n_state], initializer=tf.constant_initializer(1),
                             master_dtype=variable_dtype.master_dtype,
                             slice_dtype=variable_dtype.slice_dtype,
                             activation_dtype=variable_dtype.activation_dtype)
        b = mtf.get_variable(x.mesh, "b", [n_state], initializer=tf.constant_initializer(0),
                             master_dtype=variable_dtype.master_dtype,
                             slice_dtype=variable_dtype.slice_dtype,
                             activation_dtype=variable_dtype.activation_dtype)

        x = norm(x, axis, epsilon)
        x = x * g + b
        return x


def linear_attention(q, k, v):
    batch_dim, seq_dim, head_dim, dim_out = (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
    q = mtf.rename_dimension(q, "features_per_head", "features_per_head_in")
    k = mtf.rename_dimension(k, "features_per_head", "features_per_head_in")

    dim_in = k.shape[-1]

    q = mtf.softmax(q, dim_in)
    k = mtf.softmax(k, seq_dim)

    context = mtf.einsum([k, v], output_shape=[batch_dim, head_dim, dim_in, dim_out])
    attn = mtf.einsum([q, context], output_shape=[batch_dim, seq_dim, head_dim, dim_out])
    return attn


def causal_linear_attention(q, k, v, eps = 1e-6):
    batch_dim, seq_dim, head_dim, dim_out = (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
    q = mtf.rename_dimension(q, "features_per_head", "features_per_head_in")
    k = mtf.rename_dimension(k, "features_per_head", "features_per_head_in")

    dim_in = k.shape[-1]

    q = mtf.softmax(q, dim_in)
    k = mtf.exp(k)

    cumulative_k = mtf.cumsum(k, seq_dim) + eps
    D_inv = 1. / mtf.einsum([q, cumulative_k], output_shape=[batch_dim, seq_dim, head_dim])

    context = mtf.einsum([k, v], output_shape=[batch_dim, seq_dim, head_dim, dim_in, dim_out])
    cumulative_context = mtf.cumsum(context, seq_dim)

    attn = mtf.einsum([q, cumulative_context, D_inv], output_shape=[batch_dim, seq_dim, head_dim, dim_out])
    return attn


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


def memory_key_values(k, v, num_mem_kv, dim_batch, dim_heads, variable_dtype, mesh):
    """memory / key values from all attention paper"""

    dim_mem_kv = mtf.Dimension("mem_kv_sequence", num_mem_kv)
    emb_dim = k.shape[-1]
    mem_std = 1 / math.sqrt(emb_dim.size)

    mem_k = mtf.get_variable(mesh, "mem_k", mtf.Shape([dim_mem_kv, dim_heads, emb_dim]),
                             initializer=tf.random_normal_initializer(stddev=mem_std),
                             master_dtype=variable_dtype.master_dtype,
                             slice_dtype=variable_dtype.slice_dtype,
                             activation_dtype=variable_dtype.activation_dtype,
                             )
    mem_v = mtf.get_variable(mesh, "mem_v", mtf.Shape([dim_mem_kv, dim_heads, emb_dim]),
                             initializer=tf.random_normal_initializer(stddev=mem_std),
                             master_dtype=variable_dtype.master_dtype,
                             slice_dtype=variable_dtype.slice_dtype,
                             activation_dtype=variable_dtype.activation_dtype)

    mem_k, mem_v = map(lambda t: mtf.broadcast(t, [dim_batch, dim_mem_kv, dim_heads, emb_dim]),
                       (mem_k, mem_v))
    mem_k, mem_v = map(lambda t: mtf.rename_dimension(t, "mem_kv_sequence", "sequence"),
                       (mem_k, mem_v))

    k = mtf.concat([mem_k, k], "sequence")
    v = mtf.concat([mem_v, v], "sequence")
    return k, v


def attn(x, scope, n_state, *, attention_type, params, bias, dim_seq, memory_length_dim, variable_dtype, context=None, pos_emb=None):
    # x :: [batch, seq, n_embd]
    x_shape, dim_batch, *_, dim_embd, mesh = x.shape, *x.shape, x.mesh

    # n_state is the same as config["n_embd"], which is also the same as dim_embd.
    assert n_state.size % params["n_head"] == 0

    dim_heads = mtf.Dimension("heads", params["n_head"])

    num_mem_kv = params.get("num_mem_kv", 0)
    use_num_mem_kv = num_mem_kv > 0

    with tf.variable_scope(scope):
        # Compute attention inputs
        dim_kv = mtf.Dimension("features_per_head", params["n_embd"] // params["n_head"])
        mtfparams = mtf.transformer.attention.attention_params_simple(
            x.mesh,
            io_dim=dim_embd,
            kv_dim=dim_kv,
            heads_dim=dim_heads,
            variable_dtype=variable_dtype
        )
        q = mtfparams.compute_q(x)
        k = mtfparams.compute_k(x)
        v = mtfparams.compute_v(x)

        if is_incremental_inference(context):
            one_hot = mtf.one_hot(context.position - 1, dim_seq, dtype=variable_dtype.master_dtype)
            inv_one_hot = 1.0 - one_hot
            old_k, old_v = context.get_states(2)
            k = old_k * inv_one_hot + k * one_hot
            v = old_v * inv_one_hot + v * one_hot

        if exists(context):
            context.record_new_states([k, v])

        if exists(pos_emb):
            cos, sin = pos_emb
            k = apply_rotary_emb(k, cos, sin)

            if is_incremental_inference(context):
                seq_dim = cos.shape.get_dim_by_name('sequence')
                cos = mtf.gather(cos, context.position - 1, seq_dim)
                sin = mtf.gather(sin, context.position - 1, seq_dim)

            q = apply_rotary_emb(q, cos, sin)

        with tf.variable_scope("attention"):
            if attention_type == "local":
                # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
                radius = params.get("local_attention_radius", 256)

                if is_incremental_inference(context):
                    q *= one_hot

                a = mtf_transformer.attention.local_attention_1d(
                    q, k, v,
                    length_dim=k.shape[1],
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    radius=radius,
                    length_dim_num_splits=1,
                    fully_autoregressive=params["causal"],
                    attention_kwargs={},
                )

                if is_incremental_inference(context):
                    a = mtf.gather(a, context.position - 1, dim_seq)

            elif attention_type == "global":

                # TODO: pass in fake context
                # Broadcast mask bias across batch and heads
                if exists(bias):
                    if not is_incremental_inference(context):
                        broadcasted_bias = mtf.broadcast(bias, [dim_batch, dim_heads, bias.shape[-2], bias.shape[-1]])
                    else:
                        # In the incremental case, a custom mask needs to be built that masks out all key/values that are greater than the current position
                        bias = mtf.gather(bias, context.position - 1, dim_seq)
                        broadcasted_bias = mtf.broadcast(bias, [dim_batch, dim_heads, bias.shape[-1]])

                # memory key / values, from all-attention paper
                if use_num_mem_kv:
                    k, v = memory_key_values(k, v, num_mem_kv, dim_batch, dim_heads, variable_dtype, mesh)

                k = mtf.replace_dimensions(k, k.shape[1], memory_length_dim)
                v = mtf.replace_dimensions(v, v.shape[1], memory_length_dim)

                attn_dropout_rate = params["attn_dropout"] if params["mode"] == "train" else 0

                a = mtf_transformer.attention.attention(
                    q, k, v,
                    memory_length_dim=memory_length_dim,
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    bias=broadcasted_bias,
                    dropout_rate=attn_dropout_rate
                )

            elif attention_type == "linear":
                linear_attn_fn = causal_linear_attention if params["causal"] else linear_attention
                a = linear_attn_fn(q, k, v)

            else:
                raise NotImplementedError("Unknown attention type {}!".format(attention_type))

        with tf.variable_scope("compute_output"):
            a = mtfparams.compute_output(a, x_shape)

        with tf.variable_scope("compute_output_bias"):
            b = mtf.get_variable(x.mesh, "o_b", [dim_embd], initializer=tf.constant_initializer(0),
                                 master_dtype=variable_dtype.master_dtype,
                                 slice_dtype=variable_dtype.slice_dtype,
                                 activation_dtype=variable_dtype.activation_dtype)
            a += b

        if params["mode"] == "train" and params["res_dropout"] > 0:
            a = mtf.dropout(a, rate=params["res_dropout"], name="res_dropout")
        return a


def mlp(x, scope, n_state, *, variable_dtype, params):
    activation_fn = get_activation_fn(params)
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = activation_fn(linear(x, "c_fc", n_state, variable_dtype=variable_dtype, params=params))
        h2 = linear(h, "c_proj", nx, variable_dtype=variable_dtype, params=params, scale=True)
        if params["mode"] == "train" and params["res_dropout"] > 0:
            h2 = mtf.dropout(h2, rate=params["res_dropout"], name="mlp_dropout")
        return h2


def mlp_glu(x, scope, n_state, *, variable_dtype, params):
    activation_fn = get_activation_fn(params)
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = linear(x, "c_fc", n_state, params=params)

        h, gate = mtf.split(h, h.shape[-1], 2)
        h *= activation_fn(gate)

        h2 = linear(h, "c_proj", nx, variable_dtype=variable_dtype, params=params, scale=True)
        if params["mode"] == "train" and params["res_dropout"] > 0:
            h2 = mtf.dropout(h2, rate=params["res_dropout"], name="mlp_dropout")
        return h2


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

def rotary_positional_emb(mesh, sequence_dim, params, variable_dtype):
    dtype = variable_dtype.master_dtype
    dim_head = params["n_embd"] // params["n_head"]

    dim_head = mtf.Dimension("features_per_head", dim_head)
    half_dim_head = mtf.Dimension("half_features_per_head", dim_head.size // 2)

    dim_range = mtf.range(mesh, half_dim_head, dtype) * 2 / dim_head.size
    half_freqs = 1. / mtf.pow(mtf.constant(mesh, 10000, dtype = dtype), dim_range)

    seq = mtf.range(mesh, sequence_dim, dtype)
    half_freqs = mtf.einsum([half_freqs, seq], [sequence_dim, half_dim_head])

    freqs = mtf.concat((half_freqs, half_freqs), half_dim_head.name)
    freqs = mtf.rename_dimension(freqs, half_dim_head.name, dim_head.name)
    return mtf.cos(freqs), mtf.sin(freqs)

def rotate_half(x):
    dim_head_name = "features_per_head"
    dim_head = x.shape.get_dim_by_name(dim_head_name)
    half_dim_head_size = dim_head.size // 2
    x1 = mtf.slice(x, 0, half_dim_head_size, dim_head_name)
    x2 = mtf.slice(x, half_dim_head_size, half_dim_head_size, dim_head_name)
    return mtf.concat((-x2, x1), dim_head.name)

def apply_rotary_emb(x, cos, sin):
    rotated_x = rotate_half(x)
    return x * cos + rotated_x * sin
