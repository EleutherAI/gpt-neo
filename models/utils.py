import tensorflow as tf
import mesh_tensorflow as mtf

def expand_tile(value, newdim, axis=0):
    """Add a new axis of given size."""
    if axis == 0:
        return mtf.broadcast(value,
                             [newdim] + value.shape.dims)  # shape.dims gets us a list which we need in order to concat
    if axis == 1:
        return mtf.broadcast(value, value.shape.dims + [newdim])

def visible_pos(mesh, nd, ns):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    UPDATE: modified for mtf
    """
    # TODO: I'm sure this is a maximally inefficient way of doing this, also these values could probably be hardcoded
    i = mtf.range(mesh, nd, tf.int32)
    i = expand_tile(i, ns, axis=1)
    j = mtf.range(mesh, ns, tf.int32)
    j = expand_tile(j, nd, axis=0)
    m = mtf.greater_equal(i, j)
    return m

def biasmask_attn_weights(mesh, nd, ns, dtype):
    # the old mask_attn_weights applied directly to the QK;
    # this returns a bias that the attention code from mtf adds to the attention matrix.
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    # n_src and n_dest are both the same, i.e equal to sequence length
    # we rename ns because we want bias to have shape [batch, heads, memory_length, sequence] to match up with QK^T
    # information flows from k and v (memory_length) to q (sequence)
    vis = visible_pos(mesh, nd, ns)
    return mtf.cast(mtf.logical_not(vis), dtype) * -1e9
