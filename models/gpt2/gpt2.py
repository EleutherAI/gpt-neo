import math

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf
import os

#TODO: file needs porting to mtf

# TODO: standardize which parameters are int and which are Dimension, and add type annotations.
# we probably want to turn things into Dimensions at the beginning of the code, and pass those around for the rest of the code

def shape_list(x):
    # TODO: can this be used with mtf? tensor shapes are different
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    # x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    # ex = tf.exp(x)
    # return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
    return mtf.softmax(x, axis)

def gelu(x):
    # return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
    return mtf.gelu(x)

def norm(x, scope, *, axis=-1, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    # TODO: convert to mtf code
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        if params["precision"] == "bfloat16":
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    # TODO: convert to mtf code
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    # TODO: convert to mtf code
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])


# TODO: this isnt actually a convolution, rename it to something more appropriate
def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    # nf = number of features

    if params["scale_by_depth"] and scale: # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]: # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].value))

    # assuming we never do fp16 training, only bf16 or fp32. change if we someday do GPU training
    dt = tf.bfloat16 if params["precision"] == "bfloat16" else tf.float32

    # TODO: verify that this is actually right

    # not in the variable_scope because mtf already has a variable_scope in it
    c = mtf.layers.conv1d(x, nf, name=scope, filter_size=1, stride=1, filter_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dt))
    with tf.variable_scope(scope):
        singleton = mtf.Dimension('singleton', 1)

        b = mtf.get_variable(x.mesh, 'b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=dt)
        # NWC
        b = mtf.reshape(b, [singleton, singleton, nf])

        c += b
        return c


def visible_pos(mesh, nd, ns):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    UPDATE: modified for mtf
    """
    i = mtf.range(mesh, nd, tf.int32)[:,None]
    j = mtf.range(mesh, ns, tf.int32)
    m = i >= j - ns + nd
    return m


def attn(x, scope, n_state, *, past, params, block_offset=0, train=False):
    # n_state is the same as config['n_embd'], which is also the same as sh_embd.
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    # TODO: implement proper past cache. in the meantime, don't pass a past if implementing local attention!!!
    # update: *shouldn't* be a problem anymore not that we've switched to meshtf, but tpus don't support pasts apparantly (?? - ask Daj). 
    # we can remove this assert once we're sure we didnt break anything
    #assert not (params["local"] and past is not None)
    assert past is None

    # x :: [batch, seq, n_embd]
    x_shape = x.shape
    sh_batch = x_shape[0]
    sh_seq = x_shape[1]
    sh_embd = x_shape[2]

    dim_heads = mtf.Dimension("heads", params['n_head'])

    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    # no longer needed in mtf because TPUs cant handle pasts anyways, apparently
    # inp_len = sh_seq + (tf.shape(past)[3] if past is not None else 0)

    def split_heads(x):
        # TODO: convert to mtf code
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, params["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        # TODO: convert to mtf code
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    # the old mask_attn_weights applied directly to the QK; this returns a bias that the attention code from mtf adds to the attention matrix.
    def biasmask_attn_weights(mesh, dtype):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        
        # n_src and n_dest are both the same, i.e equal to sequence length
        ns = sh_seq
        nd = sh_seq

        vis = visible_pos(mesh, nd, ns)
        # TODO: am I doing this right? trying to get to [1, 1, nd, ns]. not sure if a singleton dimension object is the right way.
        # and I'm assuming it gets broadcasted from there to [batch, heads, seq, seq]?
        singleton = mtf.Dimension('singleton', 1)
        vis = mtf.reshape(vis, [singleton, singleton, nd, ns])
        return mtf.transformer.attention.visibility_mask_to_attention_bias(vis, dtype)

    with tf.variable_scope(scope):
        dim_qkv = mtf.Dimension("qkv", n_state*3)
        c = conv1d(x, 'c_attn', dim_qkv, params=params)

        conv_output_channels = c.shape[2] # should be equal to dim_qkv
        q, k, v = map(split_heads, mtf.split(c, conv_output_channels, 3))

        # this is the "2" dim in pasts. probably presents are not needed until we get the pasts stuff working. 
        present = mtf.stack([k, v], "kv", axis=1)

        if past is not None:
            # TODO: convert this code to mtf. Not neccessary until we start optimizing sampling.
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)

        # TODO: control whether layer is local on a layer-by-layer basis, not as a global. 
        if params["local"]:
            # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
            a = mtf.transformer.attention.local_attention_1d(
                q, k, v,
                length_dim=sh_seq,
                key_dim=sh_embd,
                value_dim=sh_embd,
                length_dim_num_splits=1, # TODO: we might need to split along length dimension at some point, when we do we'll need to wire this up as a param
            )
        else:
            # HOWEVER, `attention` DOES NOT implement masking so we need to pass in `bias` on our own!
            a = mtf.transformer.attention.attention(
                q, k, v,
                memory_length_dim=sh_seq,
                key_dim=sh_embd,
                value_dim=sh_embd,
                bias=biasmask_attn_weights(q.mesh, q.dtype)
            )

        a = merge_heads(a)
        a = conv1d(a, 'c_proj', sh_embd, params=params)
        a = dropout(a, params["res_dropout"], train)

        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        # TODO: nx will probably be the only thing that needs changing here
        # TODO: also n_state needs to be a Dimension. probably best if we standardize and make whatever calls this provide a Dimension in the first place.
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = dropout(h2, params["res_dropout"], train)
        return h2


def block(x, scope, *, past, params, train=False, block_offset=0):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1', params=params), 'attn', nx, past=past, params=params, block_offset=block_offset)
        x = x + a
        m = mlp(norm(x, 'ln_2', params=params), 'mlp', nx*4, params=params, train=train)
        x = x + m
        return x, present

def past_shape(*, params, batch_size=None, sequence=None):
    # TODO: think this should be converted to mtf.Shape( return ), but not sure
    # return [batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]]
    return mtf.Shape([batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]])

def mtf_squeeze(x, dim_name, name=None):
    """tf squeeze for mtf tensors"""
    # TODO: assert the dimension appears in x with size 1?
    return mtf.reduce_sum(x, reduced_dim=mtf.Dimension(dim_name, 1), name=name)

def mtf_expand_dims(x, dim_name, axis, name=None):
    """tf expand_dims for mtf tensors"""

    new_dims = list(x.shape[:])
    if axis == -1:
        new_dims.append(mtf.Dimension(dim_name, 1))
    elif axis < 0:
        new_dims.insert(axis + 1, mtf.Dimension(dim_name, 1))
    else:
        new_dims.insert(axis, mtf.Dimension(dim_name, 1))
    return mtf.reshape(x, mtf.Shape(new_dims), name=name)


def expand_tile(value, size):
    """Add a new axis of given size."""
    # TODO: convert to mtf code ?
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(mtf.expand_dims(value, axis=0), [size] + [1]*ndims) #TODO: not sure if tile works in mtf

def positions_for(tokens, past_length):
    batch_size = tokens.shape[0]
    nsteps = tokens.shape[1]
    return expand_tile(past_length + mtf.mtf_range(nsteps), batch_size)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        # x = tf.nn.dropout(x, rate=pdrop)
        x = mtf.dropout(x, rate=pdrop)
    return x

def _assert_float_dtype(dtype):
    # no usages
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype


def model(X, params, mesh, labels=None, past=None, scope='model', reuse=False, train=False):
    with tf.variable_scope(scope, reuse=reuse):
        if os.environ.get('DEBUG', 0):
            print('INPUT SHAPE:')
            print(X.shape)
        results = {}
        # batch, sequence = shape_list(X)

        # define mtf shapes and names
        batch_size = params["train_batch_size"]
        sequence_size = params["n_ctx"]
        features_len = params["n_embd"]
        assert batch_size > 0
        batch_dim = mtf.Dimension("batch", batch_size)
        sequence_dim = mtf.Dimension("sequence", sequence_size)
        vocab_dim = mtf.Dimension("vocab", params["n_vocab"])
        embd_dim = mtf.Dimension("embd", params["n_embd"])

        X = mtf.import_tf_tensor(mesh, X, mtf.Shape([batch_dim, sequence_dim, features_len])) # convert input tensor to mtf tensor

        if params["precision"] == "bfloat16":
            wpe = mtf.get_variable('wpe', mtf.Shape([params["n_ctx"], params["n_embd"]]), # Position encoding
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.bfloat16), dtype=tf.bfloat16)
            wte = mtf.get_variable('wte', mtf.Shape([params["n_vocab"], params["n_embd"]]), # Text encoding
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.bfloat16), dtype=tf.bfloat16)

        else:
            wpe = mtf.get_variable('wpe', mtf.Shape([params["n_ctx"], params["n_embd"]]), # Position encoding
                                initializer=tf.random_normal_initializer(stddev=0.01))
            wte = mtf.get_variable('wte', mtf.Shape([params["n_vocab"], params["n_embd"]]), # Text encoding
                                initializer=tf.random_normal_initializer(stddev=0.02))

        past_length = 0 if past is None else mtf.Shape(past)[-2]

        wpe = dropout(wpe, params["embed_dropout"], train)
        wte = dropout(wte, params["embed_dropout"], train)

        # TODO: convert positions_for to mtf code
        # below code gets the positional encodings for each of the tokens
        # wpe has shape [ctx, embd]
        # h has shape [batch, seq, embd]
        h = mtf.gather(wte, X, 0) + mtf.gather(wpe, positions_for(X, past_length), 0)

        # Transformer
        presents = []
        pasts = mtf.unstack(past, dim=1) if past is not None else [None] * params["n_layer"]
        assert len(pasts) == params["n_layer"]
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, params=params, block_offset=(layer * params["layer_offset"]) % params["fixed_attn_block_size"])
            presents.append(present)
        dim_name = "results"
        results['present'] = mtf.stack(presents, dim_name=dim_name, axis=1)

        #TODO: convert norm to mtf code
        h = norm(h, 'ln_f', params=params)

        # TODO: optimization suggestion from bmk:
        # optimize by putting lots of sparse layers next to each other to reduce reshapes,
        # and only reshape between sparse and regular layers instead of resizing every time for drop in compatibility
        # (I don't think we can easily do this with the mtf code.)

        h_flat = mtf.reshape(h, [batch_size*sequence_size, embd_dim])

        # h_flat :: [batch*seq, embd]
        # wte :: [vocab, embd]
        logits = mtf.einsum([h_flat, wte], output_shape=[batch_size*sequence_size, vocab_dim])
        logits = mtf.reshape(logits, [batch_size, sequence_size, vocab_dim])
        results['logits'] = logits
        # logits :: [batch, seq, vocab]
        return results
