from typing import Dict
from . import tpu

def from_config(config: Dict):
    # only tpu supported
    return tpu.TPU(tpu.TPUConfig(**config))