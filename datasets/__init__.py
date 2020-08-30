from typing import Dict
from pydantic.dataclasses import dataclass

@dataclass
class DatasetConfig:
    src: str

def from_config(config: Dict):
    pass