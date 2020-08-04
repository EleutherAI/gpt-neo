import pytest
import traceback
import logging
from contextlib import contextmanager

# helper functions

@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        logging.error(traceback.format_exc())
        raise pytest.fail("DID RAISE {0}".format(exception))

# imports

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import mesh_tensorflow as mtf
from mesh_tensorflow import placement_mesh_impl

from models.gpt2 import gpt2
from models.utils import biasmask_attn_weights

# fixtures

params = {
    "n_head": 1,
    "n_ctx": 3,
    "n_embd": 1,
    "n_vocab": 256,
    "embed_dropout": 0.,
    "n_layer": 1,
    "num_microbatches": 1,
    "train_batch_size": 1,
    "attention_types": ['global'],
    "res_dropout": 0.1,
    "activation_function": "gelu"
}

# tests

def test_model():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    batch_dim = 1
    sequence_dim = params["n_ctx"]

    features = tf.ones((batch_dim, sequence_dim), tf.int32)
    labels = tf.ones((batch_dim, sequence_dim), tf.int32)

    # create mask

    nd = mtf.Dimension('sequence', sequence_dim)
    ns = mtf.Dimension('memory_length', sequence_dim)
    bias = biasmask_attn_weights(mesh, nd, ns, tf.float32)

    with not_raises(Exception):
        logits, _, _ = gpt2.model(features, labels, params, mesh, bias)

        mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        logits = lowering.export_to_tf_tensor(logits)
