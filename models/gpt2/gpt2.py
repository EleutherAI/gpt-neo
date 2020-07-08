import math

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf
import os

#TODO: file needs porting to mtf

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

def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    # TODO: convert to mtf code
    #  (mtf.layers.conv1d is a thing but i think we want to keep scale_by_depth / scale_by_in opts)

    if params["scale_by_depth"] and scale: # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]: # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].value))

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        if params["precision"] == "bfloat16":
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    # TODO: convert to mtf code
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, params, block_offset=0, train=False):
    # TODO: convert to mtf code. There are some impls of local attention but not sure if they do exactly the same thing
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        ## LOCAL ATTENTION

    # TODO: implement proper past cache. in the meantime, don't pass a past if implementing local attention!!!
    assert not (params["local"] and past is not None)

    x_shape = tf.shape(x)
    sh_batch = x_shape[0]
    sh_seq = x_shape[1]

    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    inp_len = sh_seq + (tf.shape(past)[3] if past is not None else 0)

    if params["local"]:
        right_pad = params["fixed_attn_block_size"] - ((block_offset + inp_len) % params["fixed_attn_block_size"])
        dont_pad_aligned = False
        padded_seq = ((inp_len + params["fixed_attn_block_size"] - (1 if dont_pad_aligned else 0)) // params["fixed_attn_block_size"]) * params["fixed_attn_block_size"]

        # blocks is 1 more than would otherwise be thanks to padding
        # there's always one padded block at the end, even if it's entirely padded
        x = tf.pad(x, tf.stack([
                tf.constant([0,0]),
                tf.stack([block_offset, right_pad], axis=0),
                tf.constant([0,0])
            ], axis=0), "CONSTANT")
        #x = tf.Print(x, [tf.shape(x)[i] for i in range(len(x.shape.as_list()))])
        #x = tf.Print(x, [inp_len, right_pad])
        #x = tf.Print(x, [sh_batch * hparams.fixed_attn_block_size, padded_seq // hparams.fixed_attn_block_size, hparams.n_embd])
        x = tf.reshape(x, [sh_batch * params["fixed_attn_block_size"], padded_seq // params["fixed_attn_block_size"], params["n_embd"]]) # should be [batch * blocks, sequence / blocks, features]

    def split_heads(x):
        # TODO: convert to mtf code
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, params["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        # TODO: convert to mtf code
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # TODO: convert to mtf code
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        #TODO: convt to mtf code - already implemented at:
        # mtf.layers.multihead_attention(query_antecedent, memory_antecedent, mask, kv_channels, heads, *))

        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)

        w = dropout(w, params["attn_dropout"], train)

        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, params=params)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, params=params)
        a = dropout(a, params["res_dropout"], train)

        # a = tf.Print(a, [tf.shape(a)[i] for i in range(3)])

        if params["local"]:
            # a :: [batch * blocks, sequence / blocks, features]
            #a = tf.Print(a, [tf.shape(present)[i] for i in range(5)])
            #a = tf.Print(a, [tf.shape(a)[i] for i in range(3)])
            a = tf.reshape(a, [sh_batch, padded_seq, params["n_embd"]])[:, block_offset:-right_pad]

            # TODO: WARNING! present is a PLACEHOLDER and *should not be used*!!!
            # when sampling, pass None for pasts!

            # present: [batch, 2, heads, 1 (seq), features]

            present = tf.zeros([sh_batch, 2, params["n_head"], 1, params["n_embd"] // params["n_head"]])

        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        # TODO: nx will probably be the only thing that needs changing here
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
    # TODO: convert to mtf.shape ?
    batch_size = mtf.Shape(tokens)[0]
    nsteps = mtf.Shape(tokens)[1]
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


def model(X, params, mesh=None, labels=None, past=None, scope='model', reuse=False, train=False):
    with tf.variable_scope(scope, reuse=reuse):
        if os.environ.get('DEBUG', 0):
            print('INPUT SHAPE:')
            print(X.shape)
        results = {}
        # batch, sequence = shape_list(X)

        # define mtf shapes and names
        batch_size = os.environ.get('BATCH_SIZE', 0)
        sequence_size = os.environ.get('SEQUENCE', 0)
        features_len = os.environ.get('FEATURES', 0)
        assert batch_size > 0
        batch_dim = mtf.Dimension("batch", batch_size)
        sequence_dim = mtf.Dimension("sequence", sequence_size)

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
        h = norm(h, 'ln_f', params=params)

        # TODO: optimization suggestion from bmk:
        # optimize by putting lots of sparse layers next to each other to reduce reshapes,
        # and only reshape between sparse and regular layers instead of resizing every time for drop in compatibility

        h_flat = mtf.reshape(h, [batch_size*sequence_size, params["n_embd"]])
        # TODO: will need to replicate transpose_b by manually transposing i guess?
        # logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = mtf.einsum([h_flat, wte], output_shape=None) #TODO: do i need to set output shape?
        logits = mtf.reshape(logits, [batch_size, sequence_size, params["n_vocab"]])
        results['logits'] = logits
        return results
