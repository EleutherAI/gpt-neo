"""GPT-like model in Mesh-Tensorflow"""
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import math

import mesh_tensorflow.transformer as mtf_transformer


# --------------------------------------------------------------------------------
# LAYERS:

sentinel = object()


def expand_tile(value, newdim):
    """Add a new axis of given size."""
    return mtf.broadcast(value,
                         [newdim] + value.shape.dims)  # shape.dims gets us a list which we need in order to concat


def positions_for(tokens: mtf.Tensor, past_length: int, batch_dim: mtf.Dimension):
    nsteps = tokens.shape[1]
    return expand_tile(past_length + mtf.range(tokens.mesh, nsteps, dtype=tf.int32), batch_dim)


def norm(x, scope, *, axis=sentinel, epsilon=1e-5, params=None):
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

        u = mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
        s = mtf.reduce_mean(mtf.square(x - u), reduced_dim=axis, name="norm_reduce_mean_s")

        u = mtf.broadcast(u, x.shape)
        s = mtf.broadcast(s, x.shape)

        x = (x - u) * mtf.rsqrt(s + epsilon)
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
    # TODO: verify that this is actually right

    with tf.variable_scope('tmp_channels_reshape'):
        # rename the channels dim so we dont get a collision
        # note: after change to dense, this is no longer techincally necessary, 
        # because dense does the rename for you anyways. but this way we kee pmore control over the name of the temporary dim.
        x = mtf.reshape(x, x.shape.rename_dimension(x.shape[-1].name, 'tmp_channels'))

    # not in the variable_scope because mtf already has a variable_scope in it
    with tf.variable_scope('conv1d_main'):
        if not params["activation_function"] == "selu":
            c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True
                                kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dt))
        else:
            c = mtf.layers.dense(x, new_dims=[nf], reduced_dims=[x.shape[-1]], name=scope, use_bias=True
                                kernel_initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_in'))

        return c


def visible_pos(mesh, nd, ns):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    UPDATE: modified for mtf
    """
    i = mtf.range(mesh, nd, tf.int32)[:, None]
    j = mtf.range(mesh, ns, tf.int32)
    m = i >= j - ns + nd
    return m


# append dim = str to append onto all dim name to allow splitting i.e even / odd
def attn(x, scope, n_state, *, past, params, append_dim, train=False):
    # to understand a little better what's going on here:
    # https://medium.com/analytics-vidhya/understanding-the-gpt-2-source-code-part-4-a5fbb89e5038
    # n_state is the same as config['n_embd'], which is also the same as dim_embd.
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state.size % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    # TODO: implement proper past cache. in the meantime, don't pass a past if implementing local attention!!!
    # update: *shouldn't* be a problem anymore not that we've switched to meshtf, but tpus don't support pasts apparantly (?? - ask Daj).
    # we can remove this assert once we're sure we didnt break anything
    # assert not (params["local"] and past is not None)
    assert past is None

    # x :: [batch, seq, n_embd]
    x_shape = x.shape
    dim_batch = x_shape[0]
    dim_seq = x_shape[1]
    dim_embd = x_shape[2]

    # appending odd / even to out dimension name to avoid collison
    # attn_out_dim_name = "attn_out"
    # attn_out_dim = mtf.Dimension(attn_out_dim_name, params["n_embd"]) # this is the same as n_embd

    dim_heads = mtf.Dimension("heads", params['n_head'])

    # TODO: should append odd / even here
    features_per_head_key_name = "features_per_head_key"
    features_per_head_value_name = "features_per_head_value"
    dim_features_per_head_key = mtf.Dimension(features_per_head_key_name, params['n_embd'] // params['n_head'])
    dim_features_per_head_value = mtf.Dimension(features_per_head_value_name, params['n_embd'] // params['n_head'])

    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    # no longer needed in mtf because TPUs cant handle pasts anyways, apparently
    # inp_len = dim_seq + (tf.shape(past)[3] if past is not None else 0)

    def split_heads(x, last_dim):
        with tf.variable_scope('split_heads'):
            # From [batch, sequence, features] to [batch, heads, sequence, features_per_head]
            # heads is split out of features!
            x = mtf.reshape(x, [dim_batch, dim_seq, dim_heads, last_dim], name="split_heads_reshape")
            x = mtf.transpose(x, [dim_batch, dim_heads, dim_seq, last_dim], name="split_heads_transpose")
        return x

    def merge_heads(x, merge_dim=None):
        with tf.variable_scope('merge_heads'):
            # Reverse of split_heads
            # from [batch, heads, sequence, features_per_head] to [batch, sequence, features_per_head]
            x = mtf.transpose(x, [dim_batch, dim_seq, dim_heads, dim_features_per_head_value], name="merge_heads_transpose")
            x = mtf.reshape(x, [dim_batch, dim_seq, dim_embd], name="merge_heads_reshape")
        return x

    # the old mask_attn_weights applied directly to the QK; this returns a bias that the attention code from mtf adds to the attention matrix.
    def biasmask_attn_weights(mesh, dtype):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        # n_src and n_dest are both the same, i.e equal to sequence length
        ns = dim_seq
        nd = dim_seq

        vis = visible_pos(mesh, nd, ns)

        # TODO: am I doing this right? trying to get to [1, 1, nd, ns]. not sure if a singleton dimension object is the right way.
        # and I'm assuming it gets broadcasted from there to [batch, heads, seq, seq]?
        vis = mtf.broadcast(vis, [dim_batch, dim_heads, nd, ns])
        return mtf_transformer.attention.visibility_mask_to_attention_bias(vis, dtype)

    with tf.variable_scope(scope):

        #TODO: should append odd / even here
        dim_qkv_name = "qkv"
        # n_state is multiplied by 3 here as it will later be split into three parts (q,k,v) by mtf.split()
        dim_qkv = mtf.Dimension(dim_qkv_name, n_state.size * 3)
        c = conv1d(x, 'c_attn', dim_qkv, params=params)

        conv_output_channels = c.shape[2]  # should be equal to dim_qkv
        q, k, v = mtf.split(c, conv_output_channels, 3)
        q, k, v = split_heads(q, dim_features_per_head_key), split_heads(k, dim_features_per_head_key), split_heads(v, dim_features_per_head_value)

        # this is the "2" dim in pasts. probably presents are not needed until we get the pasts stuff working.
        present = mtf.stack([mtf.reshape(x, k.shape.rename_dimension(features_per_head_key_name, features_per_head_value_name)), v], "kv", axis=1, name="stack_presents_attn")

        if past is not None:
            # TODO: convert this code to mtf. Not neccessary until we start optimizing sampling.
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)

        with tf.variable_scope('attention'):
            # TODO: control whether layer is local on a layer-by-layer basis, not as a global.
            if params["local"]:
                # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
                a = mtf_transformer.attention.local_attention_1d(
                    q, k, v,
                    length_dim=dim_seq,
                    key_dim=dim_features_per_head_key,
                    value_dim=dim_features_per_head_value,
                    length_dim_num_splits=1,
                    attention_kwargs={}
                    # mtf argument here should be **kwargs but is just kwargs! so we have to actually give a dict
                    # TODO: we might need to split along length dimension at some point, when we do we'll need to wire this up as a param
                )

            else:
                print('qkv shape', q.shape, k.shape, v.shape)

                # HOWEVER, `attention` DOES NOT implement masking so we need to pass in `bias` on our own!
                a = mtf_transformer.attention.attention(
                    q, k, v,
                    memory_length_dim=dim_seq,
                    key_dim=dim_features_per_head_key,
                    value_dim=dim_features_per_head_value,
                    bias=biasmask_attn_weights(q.mesh, q.dtype)
                )

        a = merge_heads(a)

        # TODO: should append odd / even here
        a = conv1d(a, 'c_proj', dim_embd, params=params)
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

def alpha_dropout(x, keep_prob=None, rate=None, noise_shape=None, name=None):
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
def block(x, scope, *, past, params, append_dim, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1] # grab last dimension from input
        if not params["activation_function"] == "selu":
            a, present = attn(norm(x, 'ln_1', params=params), 'attn', nx, append_dim=append_dim, past=past, params=params,)
        else:
            a, present = attn(x, 'attn', nx, append_dim=append_dim, past=past, params=params,)
        x = x + a

        # define intermediate layer of mlp - to split
        dim_intermediate_expanded = mtf.Dimension('intermediate_expanded', nx.size * 4)
        if not params["activation_function"] == "selu":
            m = mlp(norm(x, 'ln_2', params=params), 'mlp', dim_intermediate_expanded, params=params, train=train)
        else:
            m = mlp(x, 'mlp', dim_intermediate_expanded, params=params, train=train)
        x = x + m
        return x, present


def model(features, labels, params, mesh, past=None):
    """A GPT style model implemented in mesh tensorlfow."""
    results = {}

    # define mtf dims
    batch_dim = mtf.Dimension('batch', params["train_batch_size"])
    sequence_dim = mtf.Dimension('sequence', params["n_ctx"])

    # we need this because gathering when both the args have the same dimension in them it breaks stuff.
    # this dim is specifically for the weights
    # this prevents the "Einsum has lhs dimension without corresponding rhs or output dimension." error.
    embed_sequence_dim = mtf.Dimension('embed_sequence', params["n_ctx"])
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])

    # convert input tensor to mtf tensor
    x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, sequence_dim]))

    # encoding_dt = tf.bfloat16 if params["precision"] == "bfloat16" else tf.float32
    encoding_dt = tf.float32

    wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([embed_sequence_dim, embd_dim]),  # Position encoding
                           initializer=tf.random_normal_initializer(stddev=0.01), dtype=encoding_dt)
    wte = mtf.get_variable(mesh, 'wte', mtf.Shape([vocab_dim, embd_dim]),  # Text encoding
                           initializer=tf.random_normal_initializer(stddev=0.02), dtype=encoding_dt)

    past_length = 0 if past is None else mtf.Shape(past)[-2]

    if params["embed_dropout"] > 0:
        wpe = mtf.dropout(wpe, params["embed_dropout"], name="wpe_dropout")
        wte = mtf.dropout(wte, params["embed_dropout"], name="wte_dropout")
    with tf.variable_scope('token_embd'):
        h = mtf.gather(wte, x, vocab_dim)
    with tf.variable_scope('pos_embd'):
        h += mtf.gather(wpe, positions_for(x, past_length, batch_dim), embed_sequence_dim)
    # # Transformer
    presents = []

    # TODO: we will need this code for sampling
    # singleton = mtf.Dimension('singleton', 1)
    # pasts = mtf.unstack(past, dim=singleton) if past is not None else [None] * params["n_layer"]
    # assert len(pasts) == params["n_layer"]
    pasts = [None] * params["n_layer"]

    # attn blocks
    # TODO: implement odd / even here
    # pass in even / odd then append to all relevant dimension names
    lnum = 1
    for layer, past in enumerate(pasts):
        if lnum % 2 == 0:
            append_dim = '_even'
        else:
            append_dim = '_odd'
        h, present = block(h, 'h%d' % layer, append_dim=append_dim, past=past, params=params)
        presents.append(present)
        lnum += 1

    dim_name = "results"
    results['present'] = mtf.stack(presents, dim_name=dim_name, axis=1)

    # normalize & affine transform
    if not params["activation_function"] == "selu":
        h = norm(h, 'ln_f', params=params)


    with tf.variable_scope('wte_final_einsum'):
        # equivalent to tf.matmul
        logits = mtf.einsum([h, wte], output_shape=[batch_dim, sequence_dim, vocab_dim])
    results['logits'] = logits

    vdim = results["logits"].shape[2] # get vocab dimension

    # In this case, labels are simply input shifted one token to the right
    # this op is done in the input_fn
    labels = mtf.import_tf_tensor(mesh, labels, mtf.Shape([batch_dim, sequence_dim]))

    with tf.variable_scope('xentropy_final'):
        loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=results["logits"], targets=labels, vocab_dim=vdim)
    with tf.variable_scope('reduce_mean_final'):
        loss = mtf.reduce_mean(loss_batch)
    return logits, loss, loss_batch
