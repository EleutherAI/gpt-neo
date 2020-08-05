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


def rezero(x, scope):
    with tf.variable_scope(scope):
        dt = tf.float32
        g = mtf.get_variable(x.mesh, 'g', [], initializer=tf.constant_initializer(0, dtype=dt), dtype=dt)
        return x * g

def norm(x, axis, epsilon=1e-5):
    u = mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
    s = mtf.reduce_mean(mtf.square(x - u), reduced_dim=axis, name="norm_reduce_mean_s")

    u = mtf.broadcast(u, x.shape)
    s = mtf.broadcast(s, x.shape)

    return (x - u) * mtf.rsqrt(s + epsilon)

def scale_norm(x, scope, *, axis=sentinel, epsilon=1e-5, params=None):
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        n_state = x.shape[-1]

        dt = tf.float32

        g = mtf.get_variable(x.mesh, 'g', [], initializer=tf.constant_initializer(1, dtype=dt), dtype=dt)

        x = norm(x, axis, epsilon)
        x = x * g
        return x

def layer_norm(x, scope, *, axis=sentinel, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        n_state = x.shape[-1]

        # assuming we never do fp16 training, only bf16 or fp32. change if we someday do GPU training
        # dt = tf.bfloat16 if params["precision"] == "bfloat16" else tf.float32
        dt = tf.float32

        g = mtf.get_variable(x.mesh, 'g', [n_state], initializer=tf.constant_initializer(1, dtype=dt), dtype=dt)
        b = mtf.get_variable(x.mesh, 'b', [n_state], initializer=tf.constant_initializer(0, dtype=dt), dtype=dt)

        x = norm(x, axis, epsilon)
        x = x * g + b
        return x


# TODO: this isnt actually a convolution, rename it to something more appropriate
def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    # nf = number of features
    if params["scale_by_depth"] and scale:  # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]:  # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].size))  # Dimension is a namedtuple of (name, size)

    # assuming we never do fp16 training, only bf16 or fp32. change if we someday do GPU training
    # dt = tf.bfloat16 if params["precision"] == "bfloat16" else tf.float32
    dt = tf.float32

    # not in the variable_scope because mtf already has a variable_scope in it
    with tf.variable_scope('conv1d_main'):
        if not params["activation_function"] == "selu":
            c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True,
                                kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dt))
        else:
            c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True,
                                kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'))

        return c

def attn(x, scope, n_state, *, layer_num, past, params, bias, memory_length_dim, train=False):
    # n_state is the same as config['n_embd'], which is also the same as dim_embd.
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state.size % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    # TODO: implement proper past cache. in the meantime, don't pass a past if implementing local attention!!!
    # update: *shouldn't* be a problem anymore not that we've switched to meshtf, but tpus don't support pasts apparently (?? - ask Daj).
    # we can remove this assert once we're sure we didnt break anything
    # assert not (params["local"] and past is not None)
    assert past is None

    # x :: [batch, seq, n_embd]
    x_shape = x.shape
    dim_batch = x_shape[0]
    dim_seq = x_shape[1]
    dim_embd = x_shape[2]

    dim_heads = mtf.Dimension("heads", params['n_head'])

    # TODO: what are these comments referring to lol, do we need to keep these in?
    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    # no longer needed in mtf because TPUs cant handle pasts anyways, apparently
    # inp_len = dim_seq + (tf.shape(past)[3] if past is not None else 0)


    with tf.variable_scope(scope):

        # compute attention inputs
        dim_kv = mtf.Dimension("features_per_head", params['n_embd'] // params['n_head'])
        mtfparams = mtf.transformer.attention.attention_params_simple(
            x.mesh,
            io_dim=dim_embd,
            kv_dim=dim_kv,
            heads_dim=dim_heads,
            variable_dtype=mtf.VariableDType() # TODO: set dtype here
        )
        q = mtfparams.compute_q(x)
        k = mtfparams.compute_k(x)
        v = mtfparams.compute_v(x)

        # this is the "2" dim in pasts. probably presents are not needed until we get the pasts stuff working.
        # present = mtf.stack([mtf.reshape(x, k.shape.rename_dimension(features_per_head_key_name,
        # features_per_head_value_name)), v], "kv", axis=1, name="stack_presents_attn")
        present = None

        # if past is not None:
        #    # TODO: convert this code to mtf. Not neccessary until we start optimizing sampling.
        #    pk, pv = tf.unstack(past, axis=1)
        #    k = tf.concat([pk, k], axis=-2)
        #    v = tf.concat([pv, v], axis=-2)

        with tf.variable_scope('attention'):
            if params["attention_types"][layer_num] == "local":
                # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
                a = mtf_transformer.attention.local_attention_1d(
                    q, k, v,
                    length_dim=dim_seq, #TODO: should this be memory length? lol
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    length_dim_num_splits=1,
                    attention_kwargs={}
                    # mtf argument here should be **kwargs but is just kwargs! so we have to actually give a dict
                    # TODO: we might need to split along length dimension at some point, when we do we'll need to wire this up as a param
                )
            elif params["attention_types"][layer_num] == "global":

                # TODO: the only use of context within attention is in _maybe_reshape...
                #   in that fn, context just needs to contain mesh / layout details:
                #   mesh_shape = mtf.convert_to_shape(context.model.mesh_shape)
                #   layout_rules = mtf.convert_to_layout_rules(context.model.layout)
                #   we should create a fake context, and pass to attention for the efficiency

                # `attention` DOES NOT implement masking so we need to pass in `bias` on our own!

                # broadcast mask bias across batch and heads
                broadcasted_bias = mtf.broadcast(bias, [dim_batch, dim_heads, bias.shape[-2], bias.shape[-1]])

                # rename sequence dim of k, v because otherwise the einsum calculating QK^T won't keep both sequence dims. 
                #
                # the reason they rename memory_length (k and v) instead of q, which we originally were going to do 
                # because renaming less seems better, is because q's length dim is the one left at the end.
                #
                # QK^T (logits, in the `attention` code) has shape [batch, heads, sequence, memory_length]
                # V has shape [batch, heads, sequence, memory_length]
                # s(QK^T)V eliminates memory_length and we're left with sequence again
                k = mtf.rename_dimension(k, "sequence", "memory_length")
                v = mtf.rename_dimension(v, "sequence", "memory_length")

                # TODO: i think passing in dim_seq as memory length dim might have been a problem? I honestly don't know
                #   I (sid) have changed it to memory length dim, we'll see what happens.
                a = mtf_transformer.attention.attention(
                    q, k, v,
                    memory_length_dim=memory_length_dim,
                    key_dim=dim_kv,
                    value_dim=dim_kv,
                    bias=broadcasted_bias,
                    dropout_rate=0
                )
            else:
                raise NotImplementedError("Unknown attention type {}!".format(params["attention_types"][layer_num]))

        with tf.variable_scope('compute_output'):
            a = mtfparams.compute_output(a, x_shape)
        
        with tf.variable_scope('compute_output_bias'):
            # TODO: bfloat16 should work here
            b = mtf.get_variable(x.mesh, 'o_b', [dim_embd], initializer=tf.constant_initializer(0, dtype=tf.float32), dtype=tf.float32)
            a += b

        if not params["activation_function"] == "selu":
            a = mtf.dropout(a, params["res_dropout"], name="attn_dropout")
        else:
            a = alpha_dropout(a, params["res_dropout"], name="attn_dropout")

        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        if params["activation_function"] == "gelu":
            h = mtf.gelu(conv1d(x, 'c_fc', n_state, params=params))
        elif params["activation_function"] == "selu":
            h = mtf.selu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        if not params["activation_function"] == "selu":
            h2 = mtf.dropout(h2, params["res_dropout"], name="mlp_dropout")
        else:
            h2 = alpha_dropout(h2, params["res_dropout"], name="mlp_dropout")
        return h2


def mlp_glu(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = conv1d(x, 'c_fc', n_state, params=params)

        h, gate = mtf.split(h, h.shape[-1], 2)
        h *= mtf.gelu(gate)

        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = mtf.dropout(h2, params["res_dropout"], name="mlp_dropout")
        return h2


def alpha_dropout(x, keep_prob=None, rate=None, noise_shape=None, name=None):
    # alpha dropout - used for SELU activation
    if (keep_prob is None) == (rate is None):
        raise ValueError("exactly one of keep_prob and rate should be set")
    if keep_prob is None:
        keep_prob = 1.0 - rate
    noise_shape = mtf.ops.convert_to_shape(noise_shape)
    if noise_shape is None:
        noise_shape = x.shape

    with tf.variable_scope(name, default_name="alpha_dropout"):
        if keep_prob == 1.0:
            return x

        alpha = -1.7580993408473766

        noise = mtf.ops.cast(mtf.ops.less(mtf.ops.random_uniform(
                x.mesh, noise_shape,
                dtype=(x.dtype if x.dtype.is_floating else tf.float32)),
                            keep_prob), x.dtype)

        # Mask
        x = x * noise + alpha * (1 - noise)

        # Affine transformation parameters
        a = (keep_prob + keep_prob * (1 - keep_prob) * alpha ** 2) ** -0.5
        b = -a * alpha * (1 - keep_prob)

        # Affine transformation
        return a * x + b


# append dim = str to append onto all dim name to allow splitting i.e even / odd
def block(params, scope, past, layer_num, bias, memory_length_dim, train=False):
    # train param doesnt seem to do anything?
    use_selu = params["activation_function"] == "selu"
    use_rezero = params["rezero"] == True
    use_mlp_glu = params["mlp_glu"] == True
    use_scale_norm = params["scalenorm"] == True
    use_norm = not use_selu and not use_rezero

    def fn(x):
        with tf.variable_scope(scope):
            nx = x.shape[-1] # grab last dimension from input

            # if we are using selu activation, forgo layer norm
            if not use_norm:
                prenorm = identity
            elif use_scale_norm:
                prenorm = scale_norm
            else:
                prenorm = layer_norm

            preresidual = rezero if use_rezero else identity

            a, present = attn(prenorm(x, 'ln_1', params=params), 'attn', nx, layer_num=layer_num, past=past,
                              params=params, bias=bias, memory_length_dim=memory_length_dim)
            a = preresidual(a)
            x = x + a

            mlp_fn = mlp_glu if use_mlp_glu else mlp
            intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)

            # define intermediate layer of mlp - to split
            dim_intermediate_expanded = mtf.Dimension('intermediate_expanded', intermediate_size)

            m = mlp_fn(prenorm(x, 'ln_2', params=params), 'mlp', dim_intermediate_expanded, params=params, train=train)
            m = preresidual(m)
            x = x + m
            return x
    return fn

# --------------------------------------------------------------------------------
# MODEL:


def model(mtf_features, other_features, params, mesh, past=None):
    """A GPT style model implemented in mesh tensorflow."""

    results = {}

    # parse inputs and labels from the mtf_features / other_features input dicts
    # all dimensions are defined inside model_fn for efficiency
    x = mtf_features["inputs"]
    labels = mtf_features["labels"]
    batch_dim = x.shape[0]
    sequence_dim = x.shape[1]  # define seq length dim
    embd_dim = other_features["embd_dim"]
    vocab_dim = other_features["vocab_dim"]
    embed_sequence_dim = other_features["embed_sequence_dim"]

    encoding_dt = tf.float32 # TODO: bfloat should apply here?
    wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([embed_sequence_dim, embd_dim]),  # Position encoding
                           initializer=tf.random_normal_initializer(stddev=0.01), dtype=encoding_dt)
    wte = mtf.get_variable(mesh, 'wte', mtf.Shape([vocab_dim, embd_dim]),  # Text encoding
                           initializer=tf.random_normal_initializer(stddev=0.02), dtype=encoding_dt)
    past_length = 0 if past is None else mtf.Shape(past)[-2]
    if params["embed_dropout"] > 0:
        wpe = mtf.dropout(wpe, params["embed_dropout"], name="wpe_dropout")
        wte = mtf.dropout(wte, params["embed_dropout"], name="wte_dropout")
    with tf.variable_scope('token_embd'):
        # text embedding
        h = mtf.gather(wte, x, vocab_dim)
    with tf.variable_scope('pos_embd'):
        # positional embedding
        h += mtf.gather(wpe, positions_for(x, past_length, batch_dim), embed_sequence_dim)

    # TODO: we will need this code for sampling
    # singleton = mtf.Dimension('singleton', 1)
    # pasts = mtf.unstack(past, dim=singleton) if past is not None else [None] * params["n_layer"]
    # assert len(pasts) == params["n_layer"]

    # Transformer
    pasts = [None] * params["n_layer"]
    presents = []
    for layer, past in enumerate(pasts):
        # attn blocks
        # TODO: make recompute grad optional, since it's slower for models that can fit in memory
        h = mtf.recompute_grad(block(params=params, scope='h%d' % layer, past=past, layer_num=layer,
                                     bias=other_features["attn_bias"], memory_length_dim=other_features["memory_length_dim"]), [h])
        # presents.append(present)

    results['present'] = None # mtf.stack(presents, dim_name=dim_name, axis=1)

    # layer normalize & affine transform
    if not params["activation_function"] == "selu":
        h = layer_norm(h, 'ln_f', params=params)

    with tf.variable_scope('wte_final_einsum'):
        # equivalent to tf.matmul
        logits = mtf.einsum([h, wte], output_shape=[batch_dim, sequence_dim, vocab_dim])

    vdim = logits.shape[2] # get vocab dimension

    with tf.variable_scope('xentropy_final'):
        loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=logits, targets=labels, vocab_dim=vdim)
    with tf.variable_scope('reduce_mean_final'):
        # TODO: divide loss by loss_denominator if necessary (think it's only necessary for batch_norm)
        loss = mtf.reduce_mean(loss_batch)
    return logits, loss, loss_batch
