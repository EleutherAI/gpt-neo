from typing import Dict, List, Pattern
from pydantic.dataclasses import dataclass

@dataclass
class DatasetConfig:
    kind: str
    sources: List[Pattern]

def from_config(config: Dict):
    return config