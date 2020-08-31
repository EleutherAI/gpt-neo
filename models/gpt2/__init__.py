import functools
from typing import Any, Dict, List, Union

import mesh_tensorflow as mtf
import tensorflow as tf
from pydantic import AnyUrl, BaseModel
from pydantic.dataclasses import dataclass

import model_fns

from . import gpt2


class GPT2Config(BaseModel):
   scale_by_depth: bool
   scale_by_in: bool
   mesh_shape: str
   layout: str
   attention_types: List[Any]
   n_ctx: int 
   n_embd: int
   n_head: int
   n_vocab: int
   n_layer: int
   activation_function: str = "gelu"
   auto_layout: bool = False
   auto_layout_and_mesh_shape: bool = False


def expand_attention_types_params(params_list):
    newlist = []
    for item in params_list:
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist

class GPT2:

    def __init__(self, config: GPT2Config):
        self.config = config
        self.mesh_shape = mtf.convert_to_shape(config.mesh_shape)

    def set_shape(self, shape):
        self._shape = shape

    def __call__(self, features: [Dict[str, tf.Tensor]], labels, mode, params:Dict[str,Union[int, str]]):
        params.update(dict(self.config))
        params.update(dict(
            model='GPT2',
            # use_tpu=False,
            eval=False,
            mode='train',
            embed_dropout=0.1,
            attn_dropout=0.1,
            res_dropout=0.1,
            attention_types = expand_attention_types_params(self.config.attention_types)
        ))
        return model_fns.model_fn(gpt2, features, labels, mode, params)


def from_config(config: Dict):
    params = GPT2Config(**config)
    model = GPT2(params)
    return model
