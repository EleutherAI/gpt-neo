"""GPT-like model in Mesh-Tensorflow"""
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import math
import mesh_tensorflow.transformer as mtf_transformer
from utils import loss_denominator
from models.utils import expand_tile

# --------------------------------------------------------------------------------
# LAYERS:

sentinel = object()


def identity(x, *args, **kwargs):
    return x


def positions_for(tokens: mtf.Tensor, past_length: int, batch_dim: mtf.Dimension):
    nsteps = tokens.shape[1]
    return expand_tile(past_length + mtf.range(tokens.mesh, nsteps, dtype=tf.int32), batch_dim)

def norm(x, axis, epsilon=1e-8):
    x -= mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
    s = mtf.reduce_mean(mtf.square(x), reduced_dim=axis, name="norm_reduce_mean_s")
    return x * mtf.rsqrt(s + epsilon)

def rezero(x, scope, dtype):
    with tf.variable_scope(scope):
        g = mtf.get_variable(x.mesh, 'g', [], initializer=tf.constant_initializer(0), dtype=dtype)
        return x * g

def scale_norm(x, scope, *, variable_dtype, axis=sentinel, epsilon=1e-5, params=None):
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):

        g = mtf.get_variable(x.mesh, 'g', [], initializer=tf.constant_initializer(1), 
                    master_dtype=variable_dtype.master_dtype, 
                    slice_dtype=variable_dtype.slice_dtype,
                    activation_dtype=x.dtype.activation_dtype) 

        x = norm(x, axis, epsilon)
        x = x * g
        return x


def layer_norm(x, scope, *, variable_dtype, axis=sentinel, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        n_state = x.shape[-1]

        g = mtf.get_variable(x.mesh, 'g', [n_state], initializer=tf.constant_initializer(1),
                                        master_dtype=variable_dtype.master_dtype,
                                        slice_dtype=variable_dtype.slice_dtype,
                                        activation_dtype=variable_dtype.activation_dtype)
        b = mtf.get_variable(x.mesh, 'b', [n_state], initializer=tf.constant_initializer(0),
                                        master_dtype=variable_dtype.master_dtype,
                                        slice_dtype=variable_dtype.slice_dtype,
                                        activation_dtype=variable_dtype.activation_dtype)

        x = norm(x, axis, epsilon)
        x = x * g + b
        return x


def linear_attention(q, k, v, epsilon=1e-6):
    batch_dim, seq_dim, head_dim, dim_out = (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
    q = mtf.rename_dimension(q, 'features_per_head', 'features_per_head_in')
    k = mtf.rename_dimension(k, 'features_per_head', 'features_per_head_in')

    dim_in = k.shape[-1]

    q = mtf.softmax(q, dim_in)
    k = mtf.elu(k) + 1

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

    # not in the variable_scope because mtf already has a variable_scope in it
    with tf.variable_scope('conv1d_main'):
        c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                            variable_dtype=variable_dtype,
                        )
        return c


def attn(x, scope, n_state, *, layer_num, past, params, bias, memory_length_dim, variable_dtype, context=None):
    # x :: [batch, seq, n_embd]
    x_shape, dim_batch, dim_seq, dim_embd, mesh = x.shape, *x.shape, x.mesh

    # n_state is the same as config['n_embd'], which is also the same as dim_embd.
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state.size % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
    assert past is None

    dim_heads = mtf.Dimension("heads", params['n_head'])

    num_mem_kv = params.get('num_mem_kv', 0)
    use_num_mem_kv = num_mem_kv > 0

    with tf.variable_scope(scope):
        # compute attention inputs
        dim_kv = mtf.Dimension("features_per_head", params['n_embd'] // params['n_head'])
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

        if context is not None:
            if context.mode == "incremental":
                one_hot = mtf.one_hot(
                    context.position, dim_seq, dtype=variable_dtype.master_dtype)
                inv_one_hot = 1.0 - one_hot
                old_k, old_v = context.get_states(2)
                k = old_k * inv_one_hot + k * one_hot
                v = old_v * inv_one_hot + v * one_hot

            # will probably need this later (related to masking) - not sure how it works exactly for now
            # memory_position = mtf.range(context.mesh, memory_length, tf.int32)
        if context is not None:
            if context.mode == "incremental" or context.mode == "first_part":
                context.record_new_states([k, v])

        present = None

        attention_type = params["attention_types"][layer_num]

        with tf.variable_scope('attention'):
            if attention_type == "local":
                # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
                radius = params.get("local_attention_radius", 256)

                a = mtf_transformer.attention.local_attention_1d(
                    q, k, v,
                    length_dim=dim_seq,  # TODO: should this be memory length?
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    radius=radius,
                    length_dim_num_splits=1,
                    attention_kwargs={}
                    # mtf argument here should be **kwargs but is just kwargs! so we have to actually give a dict
                    # TODO: we might need to split along length dimension at some point, when we do we'll need to
                    #  wire this up as a param
                )
            elif attention_type == "global":

                # TODO: the only use of context within attention is in _maybe_reshape...
                #   in that fn, context just needs to contain mesh / layout details:
                #   mesh_shape = mtf.convert_to_shape(context.model.mesh_shape)
                #   layout_rules = mtf.convert_to_layout_rules(context.model.layout)
                #   we should create a fake context, and pass to attention for the efficiency

                # broadcast mask bias across batch and heads
                broadcasted_bias = mtf.broadcast(bias, [dim_batch, dim_heads, bias.shape[-2], bias.shape[-1]])

                # rename sequence dim of k, v because otherwise the einsum calculating QK^T won't keep both sequence
                # dims.
                #
                # the reason they rename memory_length (k and v) instead of q, which we originally were going to do
                # because renaming less seems better, is because q's length dim is the one left at the end.
                #
                # QK^T (logits, in the `attention` code) has shape [batch, heads, sequence, memory_length]
                # V has shape [batch, heads, sequence, memory_length]
                # s(QK^T)V eliminates memory_length and we're left with sequence again

                # memory key / values, from all-attention paper
                if use_num_mem_kv:
                    dim_mem_kv = mtf.Dimension('mem_kv_sequence', num_mem_kv)

                    with tf.variable_scope('memory_key_values'):
                        emb_dim = k.shape[-1]
                        mem_std = 1 / math.sqrt(emb_dim.size)

                        mem_k = mtf.get_variable(mesh, 'mem_k', mtf.Shape([dim_mem_kv, dim_heads, emb_dim]),
                                                initializer=tf.random_normal_initializer(stddev=mem_std),
                                                master_dtype=variable_dtype.master_dtype,
                                                slice_dtype=variable_dtype.slice_dtype,
                                                activation_dtype=variable_dtype.activation_dtype,
                                            )
                        mem_v = mtf.get_variable(mesh, 'mem_v', mtf.Shape([dim_mem_kv, dim_heads, emb_dim]),
                                                 initializer=tf.random_normal_initializer(stddev=mem_std),
                                                master_dtype=variable_dtype.master_dtype,
                                                slice_dtype=variable_dtype.slice_dtype,
                                                activation_dtype=variable_dtype.activation_dtype)

                        mem_k, mem_v = map(lambda t: mtf.broadcast(t, [dim_batch, dim_mem_kv, dim_heads, emb_dim]),
                                           (mem_k, mem_v))
                        mem_k, mem_v = map(lambda t: mtf.rename_dimension(t, 'mem_kv_sequence', 'sequence'),
                                           (mem_k, mem_v))

                        k = mtf.concat([mem_k, k], 'sequence')
                        v = mtf.concat([mem_v, v], 'sequence')

                k = mtf.replace_dimensions(k, k.shape[1], memory_length_dim)
                v = mtf.replace_dimensions(v, v.shape[1], memory_length_dim)

                attn_dropout_rate = params["attn_dropout"] if params["mode"] == tf.estimator.ModeKeys.TRAIN else 0

                a = mtf_transformer.attention.attention(
                    q, k, v,
                    memory_length_dim=memory_length_dim,
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    bias=broadcasted_bias,
                    dropout_rate=attn_dropout_rate
                )

            elif attention_type == 'linear':
                a = linear_attention(q, k, v)

            else:
                raise NotImplementedError("Unknown attention type {}!".format(params["attention_types"][layer_num]))

        with tf.variable_scope('compute_output'):
            a = mtfparams.compute_output(a, x_shape)

        with tf.variable_scope('compute_output_bias'):
            b = mtf.get_variable(x.mesh, 'o_b', [dim_embd], initializer=tf.constant_initializer(0),
                                                master_dtype=variable_dtype.master_dtype,
                                                slice_dtype=variable_dtype.slice_dtype,
                                                activation_dtype=variable_dtype.activation_dtype)
            a += b

        # TODO: do we need this dropout?
        # if params["mode"] == "train" and params["res_dropout"] > 0:
        #     a = mtf.dropout(a, rate = params["res_dropout"], name="res_dropout")
        return a, present


def mlp(x, scope, n_state, *, variable_dtype, params):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = mtf.gelu(linear(x, 'c_fc', n_state, variable_dtype=variable_dtype, params=params))
        h2 = linear(h, 'c_proj', nx, variable_dtype=variable_dtype, params=params, scale=True)
        if params["mode"] == "train" and params["res_dropout"] > 0:
            h2 = mtf.dropout(h2, rate=params["res_dropout"], name="mlp_dropout")
        return h2


def mlp_glu(x, scope, n_state, *, variable_dtype, params):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = linear(x, 'c_fc', n_state, params=params)

        h, gate = mtf.split(h, h.shape[-1], 2)
        h *= mtf.gelu(gate)

        h2 = linear(h, 'c_proj', nx, variable_dtype=variable_dtype, params=params, scale=True)
        if params["mode"] == "train" and params["res_dropout"] > 0:
            h2 = mtf.dropout(h2, rate=params["res_dropout"], name="mlp_dropout")
        return h2


def block(params, scope, past, layer_num, bias, memory_length_dim, variable_dtype, context=None):
    use_mlp_glu = params["mlp_glu"] == True
    use_scale_norm = params["scalenorm"] == True
    use_moe = (params["moe_layers"] is not None) and (layer_num in params["moe_layers"])
    use_rezero = params["rezero"] == True

    def fn(x):
        with tf.variable_scope(scope):
            nx = x.shape[-1]  # grab last dimension from input

            if use_rezero:
                prenorm = identity
            elif use_scale_norm:
                prenorm = scale_norm
            else:
                prenorm = layer_norm

            pre_residual_fn = rezero if use_rezero else identity

            a, present = attn(prenorm(x, 'norm_1', variable_dtype=variable_dtype, params=params), 'attn', nx, layer_num=layer_num, past=past,
                              params=params, bias=bias, memory_length_dim=memory_length_dim,
                              variable_dtype=variable_dtype, context=context)

            x = x + pre_residual_fn(a, 'norm_rezero_1', dtype=variable_dtype)

            res_x = prenorm(x, 'norm_2', variable_dtype=variable_dtype, params=params)

            if use_moe:
                moe_params = mtf.transformer.moe.HParams()
                mtf.transformer.moe.set_default_moe_hparams(moe_params)
                for k, v in params["moe_params"].items():
                    moe_params.add_hparam(k, v)
                mtf.transformer.moe.set_default_moe_hparams(moe_params)
                moe_train = params["mode"] == "train"

                m, aux_loss = mtf.transformer.moe.transformer_moe_layer_v1(res_x, x.shape[-1], moe_params,
                                                                           train=moe_train,
                                                                           mesh_shape=params["mesh_shape"],
                                                                           layout=params["layout"],
                                                                           variable_dtype=variable_dtype)
            else:

                mlp_fn = mlp_glu if use_mlp_glu else mlp
                intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)

                # define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension('intermediate_expanded', intermediate_size)

                m = mlp_fn(res_x, 'mlp', dim_intermediate_expanded, variable_dtype=variable_dtype, params=params)
                aux_loss = mtf.zeros(x.mesh, mtf.Shape([]), dtype=variable_dtype.slice_dtype)

            x = x + pre_residual_fn(m, 'norm_rezero_2', variable_dtype)
            return x, aux_loss

    return fn


# --------------------------------------------------------------------------------
# MODEL:


def model(mtf_features, other_features, params, mesh, variable_dtype, past=None, context=None):
    """A GPT style model implemented in mesh tensorflow."""
    results = {}
    recompute_grad = params["recompute_grad"] == True  # if true, enable gradient checkpointing
    use_axial_pos_emb = params["axial_pos_emb"] != None
    no_weight_tie_emb = params["no_weight_tie"] == True
    share_parameters = params["share_parameters"] is not None and params["share_parameters"] == True

    # parse inputs and labels from the mtf_features / other_features input dicts
    # all dimensions are defined inside model_fn for efficiency
    x = mtf_features["inputs"]

    batch_dim = x.shape[0]
    sequence_dim = x.shape[1]  # define seq length dim
    embd_dim = other_features["embd_dim"]
    vocab_dim = other_features["vocab_dim"]
    embed_sequence_dim = other_features["embed_sequence_dim"]

    if not use_axial_pos_emb:
        wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([embed_sequence_dim, embd_dim]),  # Position encoding
                               initializer=tf.random_normal_initializer(stddev=0.01),
                               master_dtype=variable_dtype.master_dtype,
                               slice_dtype=variable_dtype.slice_dtype,
                               activation_dtype=variable_dtype.activation_dtype)
    else:
        axial_dim_1, axial_dim_2 = params["axial_pos_emb"]

        axial_dim = mtf.Dimension('axial_dim', axial_dim_1 * axial_dim_2)
        dim_axials = [mtf.Dimension('axial_dim_{}'.format(i), t) for i, t in enumerate((axial_dim_1, axial_dim_2))]

        axial_wpe_1 = mtf.get_variable(mesh, 'axial_wpe_1', mtf.Shape([dim_axials[0], embd_dim]),  # Position encoding
                                        initializer=tf.random_normal_initializer(stddev=0.01),
                                        master_dtype=variable_dtype.master_dtype,
                                        slice_dtype=variable_dtype.slice_dtype,
                                        activation_dtype=variable_dtype.activation_dtype)

        axial_wpe_2 = mtf.get_variable(mesh, 'axial_wpe_2', mtf.Shape([dim_axials[1], embd_dim]),  # Position encoding
                                        initializer=tf.random_normal_initializer(stddev=0.01),
                                        master_dtype=variable_dtype.master_dtype,
                                        slice_dtype=variable_dtype.slice_dtype,
                                        activation_dtype=variable_dtype.activation_dtype)

        axial_wpe_1, axial_wpe_2 = map(lambda t: mtf.broadcast(t, [dim_axials[0], dim_axials[1], embd_dim]),
                                       (axial_wpe_1, axial_wpe_2))
        wpe = (axial_wpe_1 + axial_wpe_2) / 2

        wpe = mtf.reshape(wpe, [axial_dim, embd_dim])

    wte = mtf.get_variable(mesh, 'wte', mtf.Shape([vocab_dim, embd_dim]),  # Text encoding
                            initializer=tf.random_normal_initializer(stddev=0.02),
                            master_dtype=variable_dtype.master_dtype,
                            slice_dtype=variable_dtype.slice_dtype,
                            activation_dtype=variable_dtype.activation_dtype)

    if params["embed_dropout"] > 0 and params["mode"] == "train":
        wpe = mtf.dropout(wpe, rate=params["embed_dropout"], name="wpe_dropout")
        wte = mtf.dropout(wte, rate=params["embed_dropout"], name="wte_dropout")

    with tf.variable_scope('token_embd'):
        # text embedding
        h = mtf.gather(wte, x, vocab_dim)
    with tf.variable_scope('pos_embd'):
        # positional embedding
        h += mtf.gather(wpe, mtf.range(mesh, sequence_dim, tf.int64), wpe.shape[0])

    # Transformer
    pasts = [None] * params["n_layer"]

    # gradient checkpointing 
    aux_losses = mtf.get_variable(mesh, 
                            name="aux_losses", 
                            shape=mtf.Shape([]), # loss must be a scalar
                            initializer=tf.constant_initializer(0), 
                            trainable=False,  
                            dtype=variable_dtype.slice_dtype)

    for layer, past in enumerate(pasts):
        # attn blocks
        block_scope = 'h%s' % (str(layer) if not share_parameters else '')

        block_fn = block(params=params, scope=block_scope, past=past, layer_num=layer,
                        bias=other_features["attn_bias"],
                        memory_length_dim=other_features["memory_length_dim"],
                        variable_dtype=variable_dtype,
                        context=context)

        h, loss = block_fn(h) if not recompute_grad else mtf.recompute_grad(block_fn, [h])
        aux_losses += loss

    results['present'] = None  # mtf.stack(presents, dim_name=dim_name, axis=1)

    if no_weight_tie_emb:
        with tf.variable_scope('wte_final_linear'):
            logits = linear(h, 'linear_out', vocab_dim, variable_dtype=variable_dtype, params=params)
    else:
        # layer normalize & affine transform
        h = layer_norm(h, 'ln_f', variable_dtype=variable_dtype, params=params)
        with tf.variable_scope('wte_final_einsum'):
            # equivalent to tf.matmul
            logits = mtf.einsum([h, wte], output_shape=[batch_dim, sequence_dim, vocab_dim])

    vdim = logits.shape[2]  # get vocab dimension
    if params["mode"] is not "predict":
        labels = mtf_features["labels"]
        z_loss = params.get('z_loss', 1e-4)
        # go to full precision for the logits 
        logits = mtf.cast(logits, tf.float32)
        with tf.variable_scope('xentropy_final'):
            loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=logits, targets=labels, vocab_dim=vdim, z_loss=z_loss)
        with tf.variable_scope('reduce_mean_final'):
            loss = mtf.reduce_mean(loss_batch)
        loss += aux_losses  # add on auxiliary losses (currently only used for moe)
        loss /= params["num_microbatches"]
    else:
        loss = None
        loss_batch = None
    if loss:
        # convert to train dimension 
        loss = mtf.cast(loss, variable_dtype.slice_dtype)

    # cast back to checkpoint dtype
    logits = mtf.cast(logits, variable_dtype.master_dtype)

    return logits, loss, loss_batch
