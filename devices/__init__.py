from typing import Dict, Union
from . import tpu
from pydantic.dataclasses import dataclass

import multiprocessing

@dataclass
class CPUDeviceSpec:
    num_cores: int = multiprocessing.cpu_count()

@dataclass
class TPUDeviceSpec:
    model: str
    address: str

class CPU(tpu.TPU):
    # HACK
    def __init__(self, config: CPUDeviceSpec):
        self.config = config
        self._cluster = None

    def resolve(self):
        pass

DeviceSpec = Union[CPUDeviceSpec, TPUDeviceSpec]

def from_config(config: DeviceSpec):
    # only tpu supported
    if type(config) is TPUDeviceSpec:
        return tpu.TPU(config)
    return CPU(config)