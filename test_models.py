import pytest
import traceback
import logging
from collections import defaultdict
from contextlib import contextmanager

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import mesh_tensorflow as mtf
from mesh_tensorflow import placement_mesh_impl

from inputs import mlm_sample_text
from models.gpt2 import gpt2
from models.utils import biasmask_attn_weights, entmax, sample_categorical

from sample import sample_autoregressive

# helper functions

@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        logging.error(traceback.format_exc())
        raise pytest.fail("DID RAISE {0}".format(exception))

# fixtures

params = defaultdict(lambda: None, {
    "n_head": 1,
    "n_ctx": 4,
    "n_embd": 2,
    "n_vocab": 256,
    "embed_dropout": 0.,
    "n_layer": 2,
    "num_microbatches": 1,
    "train_batch_size": 1,
    "causal": True,
    "attention_types": ['global', 'local'],
    "res_dropout": 0.1,
    "rotary_emb": True,
    "activation_function": "gelu",
    "moe_layers": (1,),
    "num_mem_kv": 16,
    "no_weight_tie": True,
    "moe_params": {
        'moe_dropout_rate': 0.0
    },
    "mesh_shape": [],
    "layout": {},
    "local_attention_radius": 128,
    "share_parameters": True,
    "rezero": True
})

# tests

def test_model():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    seq_len = params["n_ctx"]

    batch_dim = mtf.Dimension("batch", 1)
    sequence_dim = mtf.Dimension("sequence", seq_len)

    features = {
        'inputs': mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32),
        'labels': mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)
    }    

    # create mask

    num_mem_kv = params.get('num_mem_kv', 0)
    length_dim = mtf.Dimension('sequence', seq_len)
    memory_length_dim = mtf.Dimension('memory_length', seq_len + num_mem_kv)
    embed_sequence_dim = mtf.Dimension('embed_sequence', seq_len)
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])

    other_features = {}
    variable_dtype = mtf.VariableDType(tf.float32, tf.float32, tf.float32)

    other_features["attn_bias"] = biasmask_attn_weights(mesh, length_dim, memory_length_dim, variable_dtype)
    other_features["embd_dim"] = embd_dim
    other_features["vocab_dim"] = vocab_dim
    other_features["embed_sequence_dim"] = embed_sequence_dim
    other_features["memory_length_dim"] = memory_length_dim

    with not_raises(Exception):
        logits, _, _ = gpt2.model(features, other_features, params, mesh, variable_dtype=variable_dtype)

        mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        logits = lowering.export_to_tf_tensor(logits)


def test_sampling():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    batch_dim = mtf.Dimension("batch", 1)
    sequence_dim = mtf.Dimension("sequence", 1)

    inputs = mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)
    inputs = mtf.pad(inputs, [0, 3], sequence_dim.name)

    # create mask

    seq_len = params["n_ctx"]
    num_mem_kv = params.get('num_mem_kv', 0)
    length_dim = mtf.Dimension('sequence', seq_len)
    memory_length_dim = mtf.Dimension('memory_length', seq_len + num_mem_kv)
    embed_sequence_dim = mtf.Dimension('embed_sequence', seq_len)
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])

    other_features = {}

    other_features["attn_bias"] = biasmask_attn_weights(mesh, length_dim, memory_length_dim, mtf.VariableDType(tf.float32))
    other_features["embd_dim"] = embd_dim
    other_features["vocab_dim"] = vocab_dim
    other_features["embed_sequence_dim"] = embed_sequence_dim
    other_features["memory_length_dim"] = memory_length_dim

    params["mode"] = "predict"

    with not_raises(Exception):
        samples = sample_autoregressive(
            inputs, other_features=other_features, params=params, variable_dtype=mtf.VariableDType(),
            remove_partial_sequences=params["remove_partial_sequences"], stop_at_token=params["eos_id"], sampling_use_entmax=True)

        mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        samples = lowering.export_to_tf_tensor(samples)

# mlm

mlm_params = defaultdict(lambda: None, {
    "n_head": 1,
    "n_ctx": 4,
    "n_embd": 1,
    "n_vocab": 256,
    "embed_dropout": 0.,
    "n_layer": 2,
    "num_microbatches": 1,
    "train_batch_size": 1,
    "attention_types": ['global', 'local'],
    "res_dropout": 0.1,
    "mesh_shape": [],
    "layout": {},
    "share_parameters": True,
    "mlm_training": True,
    "mlm_mask_id": 3,
    "mlm_cls_token_id": 4,
    "mlm_random_token_prob": 0.1
})

def test_mlm_sample_text():
    document = tf.random.normal((16,))
    with not_raises(Exception):
        features, labels = mlm_sample_text(mlm_params, document, random_documents = True)
        assert features.shape == (mlm_params['n_ctx'],)

# entmax

def test_entmax():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    length = mtf.Dimension("tensor_length", 8)
    tensor = mtf.range(mesh, length, tf.float32)
    output = entmax(tensor)
    grad = mtf.gradients([output], [tensor])[0]
    sample = sample_categorical(output, length)

    mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    sample = lowering.export_to_tf_tensor(sample)
    grad = lowering.export_to_tf_tensor(grad)
