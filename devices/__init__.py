from typing import Dict
from . import tpu

def from_config(config: Dict):
    # only tpu supported
    if config.get('kind', 'cpu') == 'tpu':
        config.pop('kind')
    return tpu.TPU(tpu.TPUConfig(**config))