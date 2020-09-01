from typing import Dict, Union, Optional
from . import tpu
from pydantic.dataclasses import dataclass

import multiprocessing

@dataclass
class CPUDeviceSpec:
    num_cores: int = multiprocessing.cpu_count()

@dataclass
class TPUDeviceSpec:
    address: str
    num_cores: int
    model: Optional[str] = None

class CPU(tpu.TPU):
    # HACK
    def __init__(self, config: CPUDeviceSpec):
        self.config = config
        self._cluster = None

    def resolve(self):
        pass

DeviceSpec = Union[CPUDeviceSpec, TPUDeviceSpec]

def from_config(config: Dict):
    # only tpu supported
    if config['kind'] == 'tpu':
        config.pop('kind')
        return tpu.TPU(TPUDeviceSpec(**config))
    return CPU(config)