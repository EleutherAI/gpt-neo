import tensorflow as tf
import mesh_tensorflow as mtf

def entmax(x, alpha = 1.3, dim = None, n_iter = 2, ensure_sum_one = True):
    assert alpha > 1 and alpha < 2, 'alpha must be between 1 and 2'

    _gp = lambda x, alpha: x ** (alpha - 1)
    _gp_inv = lambda x, alpha: mtf.pow(x, (1 / (alpha - 1)))
    _p = lambda x, alpha: _gp_inv(mtf.relu(x), alpha)

    dim = x.shape[-1] if dim is None else dim
    d = dim.size

    x = x * (alpha - 1)

    max_val = mtf.reduce_max(x, reduced_dim = dim)

    tau_lo = max_val - _gp(1, alpha)
    tau_hi = max_val - _gp(1 / d, alpha)

    f_lo = mtf.reduce_sum(_p(x - tau_lo, alpha), reduced_dim = dim) - 1

    dm = tau_hi - tau_lo

    for _ in range(n_iter):
        dm = dm / 2
        tau_m = tau_lo + dm
        p_m = _p(x - tau_m, alpha)
        f_m = mtf.reduce_sum(p_m, reduced_dim = dim) - 1

        mask = mtf.greater_equal((f_m * f_lo), 0)
        tau_lo = mtf.where(mask, tau_m, tau_lo)

    if ensure_sum_one:
        p_m = p_m / mtf.reduce_sum(p_m, reduced_dim = dim)

    return p_m

def biasmask_attn_weights(mesh, nd, ns, variable_dtype):
    # The old mask_attn_weights applied directly to the QK;
    # this returns a bias that the attention code from mtf adds to the attention matrix.
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    # n_src and n_dest are both the same, i.e equal to sequence length
    # We rename ns because we want bias to have shape [batch, heads, memory_length, sequence] to match up with QK^T
    # Information flows from k and v (memory_length) to q (sequence)
    i = mtf.range(mesh, nd, tf.int32) + ns.size - nd.size
    j = mtf.range(mesh, ns, tf.int32)
    i, j = map(lambda t: mtf.broadcast(t, [nd, ns]), (i, j))
    dtype = variable_dtype.activation_dtype
    return mtf.cast(mtf.less(i, j), dtype) * -1e10

def parse_inputs(mtf_features, other_features):
    # Parse inputs and labels from the mtf_features / other_features input dicts
    # All dimensions are defined inside model_fn for efficiency
    x = mtf_features["inputs"]

    batch_dim = x.shape[0]
    sequence_dim = x.shape[1]
    embd_dim = other_features["embd_dim"]
    vocab_dim = other_features["vocab_dim"]
    embed_sequence_dim = other_features["embed_sequence_dim"]

    return x, batch_dim, sequence_dim, embd_dim, vocab_dim, embed_sequence_dim