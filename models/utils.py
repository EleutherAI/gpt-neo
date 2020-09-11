import tensorflow as tf
import mesh_tensorflow as mtf

def expand_tile(value, newdim, axis=0):
    """Add a new axis of given size."""
    if axis == 0:
        return mtf.broadcast(value,
                             [newdim] + value.shape.dims)  # shape.dims gets us a list which we need in order to concat
    if axis == 1:
        return mtf.broadcast(value, value.shape.dims + [newdim])

def biasmask_attn_weights(mesh, nd, ns, variable_dtype):
    # the old mask_attn_weights applied directly to the QK;
    # this returns a bias that the attention code from mtf adds to the attention matrix.
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    # n_src and n_dest are both the same, i.e equal to sequence length
    # we rename ns because we want bias to have shape [batch, heads, memory_length, sequence] to match up with QK^T
    # information flows from k and v (memory_length) to q (sequence)
    i = mtf.range(mesh, nd, tf.int32) + ns.size - nd.size
    j = mtf.range(mesh, ns, tf.int32)
    i, j = map(lambda t: mtf.broadcast(t, [nd, ns]), (i, j))
    dtype = variable_dtype.activation_dtype
    return mtf.cast(mtf.less(i, j), dtype) * -1e10
