import argparse
import os
import numpy as np 

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--lm_head', action='store_true', default=False, help='Store LM head (just the embeddings transposed')
parser.add_argument('--config_file', type=str, help='HF config file')
parser.add_argument('--gpt_meshtf_path', type=str, default='./chkpt', help='Path to meshtf model checkpoint')
parser.add_argument('--output_path', type=str, default='./', help='Dir to store pytorch model')


def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """Load tf checkpoints in a Huggingface transformers pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    qkv = {}
    for name, shape in init_vars:
        if 'global_step' not in name and 'adam' not in name:
            print("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            array = tf.dtypes.cast(array.squeeze(), tf.float32).numpy()
            if any('attn/' + n in name for n in ['q', 'k', 'v']):
                qkv[name] = array
            else:
                name = name.replace('attn/o', 'attn/c_proj/w')
                name = name.replace('norm_1', 'ln_1')
                name = name.replace('norm_2', 'ln_2')
                name = name.replace('attn/compute_output_bias/o_b', 'attn/c_proj/b')
                name = name.replace('conv1d_main/c_fc/kernel', 'c_fc/w')
                name = name.replace('conv1d_main/c_fc/bias', 'c_fc/b')
                name = name.replace('conv1d_main/c_proj/kernel', 'c_proj/w')
                name = name.replace('conv1d_main/c_proj/bias', 'c_proj/b')
                names.append(name)
                arrays.append(array)

    # Combine query, key and value weight into one
    for i in range(len(qkv.keys()) // 3):
        # Scale query weight to offset scale in HF code
        qkv[f'gpt2/h{i}/attn/q'] = qkv[f'gpt2/h{i}/attn/q'] * float(qkv[f'gpt2/h{i}/attn/q'].shape[0] / 32)**0.5
        qkv_weight = np.concatenate((qkv[f'gpt2/h{i}/attn/q'], qkv[f'gpt2/h{i}/attn/k']), axis=1)
        qkv_weight = np.concatenate((qkv_weight, qkv[f'gpt2/h{i}/attn/v']), axis=1)
        names.append(f'gpt2/h{i}/attn/c_attn/w')
        arrays.append(qkv_weight)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "gpt2/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    return model


args = parser.parse_args()
config = GPT2Config.from_pretrained(args.config_file)
if args.lm_head:
    model = GPT2LMHeadModel(config=config)
    load_tf_weights_in_gpt2(model.transformer, args.gpt_meshtf_path)
    embs = model.transformer.wte.weight
    lin = nn.Linear(embs.size()[1], embs.size()[0], bias=False)
    lin.weight = embs
    model.set_output_embeddings(lin)
else:
    model = GPT2Model(config=config)
    load_tf_weights_in_gpt2(model, args.gpt_meshtf_path)

model.save_pretrained(args.output_path)
