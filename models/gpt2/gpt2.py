"""GPT-like model in Mesh-Tensorflow"""
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import math
import mesh_tensorflow.transformer as mtf_transformer
import random
from models.utils import parse_inputs, entmax_cross_entropy_with_logits

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


def causal_linear_attention(q, k, v, epsilon=1e-6):
    batch_dim, seq_dim, head_dim, dim_out = (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
    q = mtf.rename_dimension(q, "features_per_head", "features_per_head_in")
    k = mtf.rename_dimension(k, "features_per_head", "features_per_head_in")

    dim_in = k.shape[-1]

    q = mtf.softmax(q, dim_in)
    k = mtf.exp(k)

    cumulative_k = mtf.cumsum(k, seq_dim)
    context = mtf.einsum([k, v], output_shape=[batch_dim, seq_dim, head_dim, dim_in, dim_out])
    cumulative_context = mtf.cumsum(context, seq_dim)

    cumulative_context /= (cumulative_k + epsilon)
    attn = mtf.einsum([q, cumulative_context], output_shape=[batch_dim, seq_dim, head_dim, dim_out])
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


def attn(x, scope, n_state, *, attention_type, params, bias, dim_seq, memory_length_dim, variable_dtype, context=None):
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


def get_activation_fn(params):
    activation_fn = params.get("activation_fn", "gelu")
    
    
    def _arcsinh(x):
        return mtf.log(x + mtf.sqrt(1 + x ** 2))
    def _var(x, init):
        return mtf.get_variable(x.mesh, f"activation-{random.randint(0, 2**32):x}", [], initializer=tf.constant_initializer(init), dtype=x.dtype)
    def _pos_var(x, val):
        return mtf.softplus(_var(x, 0)) + val
    
    if activation_fn == "gelu": # https://arxiv.org/abs/1606.08415
        return mtf.gelu
    elif activation_fn == "relu":
        return mtf.relu
    elif activation_fn == "sigmoid":
        return mtf.sigmoid
    elif activation_fn == "tanh":
        return mtf.tanh
    elif activation_fn == "selu": # https://arxiv.org/abs/1706.02515
        return mtf.selu
    elif activation_fn == "elu": # https://arxiv.org/abs/1511.07289
        return mtf.elu
    elif activation_fn == "lrelu001":
        return lambda x: mtf.leaky_relu(x, alpha=0.01)
    elif activation_fn == "lrelu020":
        return lambda x: mtf.leaky_relu(x, alpha=0.20)

    elif activation_fn == "abs": 
        return mtf.abs
    elif activation_fn == "id":
        return lambda x: x
    elif activation_fn == "sin":
        return mtf.sin
    elif activation_fn == "cos":
        return mtf.cos
    elif activation_fn == "sign":
        return mtf.sign
    elif activation_fn == "triangle_relax":
        return lambda x: mtf.sin(x)-mtf.sin(3*x)/9+mtf.sin(5*x)/25-mtf.sin(7*x)/49
    elif activation_fn == "square_relax":
        return lambda x: mtf.cos(x)-mtf.cos(3*x)/3+mtf.cos(5*x)/5-mtf.cos(7*x)/7
    elif activation_fn == "spike":
        return lambda x: 1/(1+x**2)
    elif activation_fn == "spike2":
        return lambda x: mtf.exp(-x**2)
    
    elif activation_fn == "tanhshrink":
        return lambda x: x - tanh(x)
    elif activation_fn == "softsign":
        return lambda x: x / (mtf.abs(x) + 1)
    elif activation_fn == "softmax":
        return lambda x: mtf.softmax(x, x.shape[-1])
    elif activation_fn == "logsoftmax":
        return lambda x: mtf.log_softmax(x, x.shape[-1])
    elif activation_fn == "bipolarsigmoid":
        return lambda x: mtf.sigmoid(x) * 2 - 1
    elif activation_fn == "rrelu":  # https://arxiv.org/abs/1505.00853
        def _rrelu_fn(x):
            negative_scale = random.random()
            return (negative_scale * mtf.abs(x) + x) / (1 + negative_scale)
        return _rrelu_fn
    elif activation_fn == "elish":  # https://arxiv.org/abs/1808.00783v1
        def _elish_fn(x):
            cond = mtf.cast(mtf.greater(x, 0), x.dtype)
            exp = mtf.exp(x)
            return cond * x / (1 + exp) + (1 - cond) * (exp - 1) / (1 / exp + 1)
        return _elish_fn
    
    elif activation_fn == "silu": # https://arxiv.org/abs/1710.05941
        return mtf.swish
    
    elif activation_fn == "arcsinh":
        return _arcsinh
    
    
    # parametric
    elif activation_fn == "aria":  # https://arxiv.org/abs/1805.08878
        return lambda x: x * (_var(x, 0) + _var(x, 1) / (_pos_var(x, 0) + _var(x, 1) * mtf.exp(_var(x, -1) * x) ** (1 / _pos_var(x, 1))))
    elif activation_fn == "prelu":  # https://arxiv.org/abs/1502.01852
        return lambda x: mtf.leaky_relu(x, alpha=_var(x, 0.2))
    elif activation_fn == "parcsinh":
        return lambda x: _var(x, 1) * _arcsinh(x * _pos_var(x, 1))
    elif activation_fn == "psoftplus":
        return lambda x: _var(x, 1) * mtf.softplus(x * _var(x, 1)) + _var(x, 0)
    elif activation_fn == "proottanh":
        return lambda x: (x ** _pos_var(x, 2) + _pos_var(x, 1)) ** (1 / _pos_var(x, 3)) * mtf.tanh(x)
     
    # https://arxiv.org/abs/1710.05941, https://arxiv.org/abs/1901.02671
    elif activation_fn == "maxsig": 
        return lambda x: mtf.maximum(x, mtf.sigmoid(x))
    elif activation_fn == "cosid": 
        return lambda x: mtf.cos(x) - x
    elif activation_fn == "minsin": 
        return lambda x: mtf.minimum(x, mtf.sin(x))
    elif activation_fn == "maxtanh": 
        return lambda x: mtf.maximum(x, mtf.tanh(x))
    
    elif activation_fn == "softplus":
        return mtf.softplus
    elif activation_fn == "mish": # https://arxiv.org/abs/1908.08681
        return lambda x: x * mtf.tanh(mtf.softplus(x))
    elif activation_fn == "tanhexp": # https://arxiv.org/abs/2003.09855
        return lambda x: x * mtf.tanh(mtf.exp(x))
    elif activation_fn == "lisht": # https://arxiv.org/abs/1901.05894
        return lambda x: x * mtf.tanh(x)
    elif activation_fn == "seagull": # https://arxiv.org/abs/2011.11713
        return lambda x: mtf.log(1 + x ** 2)
    elif activation_fn == "snake": # https://arxiv.org/abs/2006.08195
        return lambda x: x + mtf.sin(x) ** 2
    
    elif activation_fn == "roottanh":  # made up
        return lambda x: (x ** 2 + 1) ** (1/3) * mtf.tanh(x)
    elif activation_fn == "softplusmone":  # made up
        return lambda x: mtf.softplus(x) - 1
    
    else:
        raise ValueError('unknown activation function "activation_fn" in config')

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


def block(params, scope, layer_num, bias, sequence_dim, memory_length_dim, variable_dtype, context=None):
    use_mlp_glu = params["mlp_glu"] == True
    use_scale_norm = params["scalenorm"] == True
    use_moe = exists(params["moe_layers"]) and (layer_num in params["moe_layers"])
    use_rezero = params["rezero"] == True
    macaron_attention = params["macaron"] == True

    def fn(x):
        with tf.variable_scope(scope):
            nx = x.shape[-1]  # Grab last dimension from input

            if use_rezero:
                prenorm = identity
            elif use_scale_norm:
                prenorm = scale_norm
            else:
                prenorm = layer_norm

            pre_residual_fn = rezero if use_rezero else identity

            attention_type = params["attention_types"][layer_num]
            
            if macaron_attention:
                mult = 0.5
                mlp_fn = mlp_glu if use_mlp_glu else mlp
                intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)
                # Define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension("intermediate_expanded", intermediate_size)
                m = mlp_fn(x, "mlp_macaron", dim_intermediate_expanded, variable_dtype=variable_dtype, params=params)
                
                x = x + (m * mult)
            else:
                mult = 1

            if attention_type != "none":
                res_x = prenorm(x, "norm_1", variable_dtype=variable_dtype, params=params)
                a = attn(res_x, "attn", nx, attention_type=attention_type,
                         params=params, bias=bias, dim_seq=sequence_dim, memory_length_dim=memory_length_dim,
                         variable_dtype=variable_dtype, context=context)
            else:
                a = x

            x = x + pre_residual_fn(a, "norm_rezero_1", dtype=variable_dtype)

            res_x = prenorm(x, "norm_2", variable_dtype=variable_dtype, params=params)

            if use_moe:
                moe_params = mtf.transformer.moe.HParams()
                mtf.transformer.moe.set_default_moe_hparams(moe_params)
                moe_params.add_hparam("moe_min_expert_capacity", 1)
                moe_params.add_hparam("moe_use_experts_attention", False)

                # Override defaults
                for k, v in params["moe_params"].items():
                    moe_params.add_hparam(k, v)

                moe_train = params["mode"] == "train"

                m, aux_loss = mtf.transformer.moe.transformer_moe_layer_v1(res_x, x.shape[-1], moe_params,
                                                                           train=moe_train,
                                                                           mesh_shape=params["mesh_shape"],
                                                                           layout=params["layout"],
                                                                           activation=params.get("moe_activation", "relu"),
                                                                           variable_dtype=variable_dtype,
                                                                           num_microbatches=params["num_microbatches"])
                m = mtf.dropout(m, rate=params["res_dropout"], name="moe_dropout")
            else:

                mlp_fn = mlp_glu if use_mlp_glu else mlp
                intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)

                # Define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension("intermediate_expanded", intermediate_size)

                m = mlp_fn(res_x, "mlp", dim_intermediate_expanded, variable_dtype=variable_dtype, params=params)
                aux_loss = mtf.zeros(x.mesh, mtf.Shape([]), dtype=variable_dtype.slice_dtype)

            x = x + pre_residual_fn((m*mult), "norm_rezero_2", variable_dtype)
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

def model(mtf_features, other_features, params, mesh, variable_dtype, context=None):
    """A GPT style model implemented in mesh tensorflow."""

    x, batch_dim, sequence_dim, embd_dim, vocab_dim, embed_sequence_dim = parse_inputs(mtf_features, other_features)

    if is_incremental_inference(context):
        # reshape inputs if in inference mode
        x = mtf.gather(x, context.position - 1, sequence_dim)
        x = mtf.reshape(x, [batch_dim])

    use_axial_pos_emb = params["axial_pos_emb"] is not None

    if not use_axial_pos_emb:
        # Use standard position encoding
        wpe = mtf.get_variable(mesh, "wpe", mtf.Shape([embed_sequence_dim, embd_dim]),
                               initializer=tf.random_normal_initializer(stddev=0.01),
                               master_dtype=variable_dtype.master_dtype,
                               slice_dtype=variable_dtype.slice_dtype,
                               activation_dtype=variable_dtype.activation_dtype)
    else:
        wpe = axial_positional_emb(embd_dim, mesh, params, variable_dtype)

    # Text encoding
    wte = mtf.get_variable(mesh, "wte", mtf.Shape([vocab_dim, embd_dim]),
                           initializer=tf.random_normal_initializer(stddev=0.02),
                           master_dtype=variable_dtype.master_dtype,
                           slice_dtype=variable_dtype.slice_dtype,
                           activation_dtype=variable_dtype.activation_dtype)

    with tf.variable_scope("token_embd"):
        # Text embedding
        h = mtf.gather(wte, x, vocab_dim)
        if params["embed_dropout"] > 0 and params["mode"] == "train":
            h = mtf.dropout(h, rate=params["embed_dropout"], name="wte_dropout")

    with tf.variable_scope("pos_embd"):
        # Positional embedding
        position_indices = mtf.range(mesh, sequence_dim, tf.int64) if not is_incremental_inference(context) else (
                context.position - 1)
        pos_emb = mtf.gather(wpe, position_indices, wpe.shape[0])
        if params["embed_dropout"] > 0 and params["mode"] == "train":
            pos_emb = mtf.dropout(pos_emb, rate=params["embed_dropout"], name="wte_dropout")
        h += pos_emb

    aux_losses = 0  # instantiate auxiliary losses (for MOE models)

    for layer in range(params["n_layer"]):
        # attn blocks
        share_parameters = exists(params["share_parameters"]) and params["share_parameters"] == True
        block_scope = f"h{layer}" if not share_parameters else ""

        block_fn = block(params=params, scope=block_scope, layer_num=layer,
                         bias=other_features["attn_bias"],
                         sequence_dim=sequence_dim,
                         memory_length_dim=other_features["memory_length_dim"],
                         variable_dtype=variable_dtype,
                         context=context)

        # If true and in train mode, enable gradient checkpointing
        recompute_grad = params["recompute_grad"] and (params["mode"] == "train") == True
        h, loss = block_fn(h) if not recompute_grad else mtf.recompute_grad(block_fn, [h])
        aux_losses += loss

    no_weight_tie_emb = params["no_weight_tie"] == True
    if no_weight_tie_emb:
        with tf.variable_scope("wte_final_linear"):
            logits = linear(h, "linear_out", vocab_dim, variable_dtype=variable_dtype, params=params)
    else:
        # Layer normalize & affine transform
        h = layer_norm(h, "ln_f", variable_dtype=variable_dtype)
        seq_dim = sequence_dim if not is_incremental_inference(context) else mtf.Dimension("sequence", 1)
        with tf.variable_scope("wte_final_einsum"):
            # Equivalent to tf.matmul
            logits = mtf.einsum([h, wte], output_shape=[batch_dim, seq_dim, vocab_dim])

    if params["mode"] in ["train", "eval"]:
        labels = mtf_features["labels"]
        z_loss = params.get("z_loss", 1e-4) # an auxiliary loss used to stabilize mtf xentropy

        # Go to full precision for the logits 
        logits = mtf.cast(logits, tf.float32)

        use_entmax_loss = params.get("entmax_loss", False)
        loss_fn = mtf.layers.softmax_cross_entropy_with_logits if not use_entmax_loss else entmax_cross_entropy_with_logits

        with tf.variable_scope("xentropy_final"):
            loss_batch = loss_fn(logits=logits, targets=labels,
                                 vocab_dim=logits.shape[-1], z_loss=z_loss)

        # For non-autoregressive models (masked language modeling training)
        # Make sure labels with padding tokens are not counted in the loss
        if not params["causal"]:
            padding_id = params.get("padding_id", 0)
            loss_batch = mtf.where(mtf.not_equal(labels, padding_id), loss_batch, mtf.zeros_like(loss_batch))

        with tf.variable_scope("reduce_mean_final"):
            loss = mtf.reduce_mean(loss_batch)

        loss += aux_losses  # Add on auxiliary losses (currently only used for MoE)
        loss /= params["num_microbatches"]
        # Convert to train dtype
        loss = mtf.cast(loss, variable_dtype.slice_dtype)
    else:
        loss = None
        loss_batch = None

    # Cast back to checkpoint dtype
    logits = mtf.cast(logits, variable_dtype.master_dtype)
    return logits, loss, loss_batch
