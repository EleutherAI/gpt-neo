import tensorflow as tf
import mesh_tensorflow as mtf

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