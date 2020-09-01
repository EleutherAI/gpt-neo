from typing import Dict, List, Pattern, Optional
from pydantic.dataclasses import dataclass
from pydantic import BaseConfig

class DatasetConfig(BaseConfig):
    kind: str
    spec: str
    sources: List[Pattern]

@dataclass
class TFRecordDataset(DatasetConfig):
    vocab_size: int
    format: str
    n_samples: int 
    context_length: int
    has_eos: Optional[bool] = False
    keys = ["content", "target"]

def from_config(config: Dict):
    if config['kind'] == 'datasets.Seq2SeqDataset':
        return seq2seq.Seq2SeqDataset(**config)

    print(config) 
    if config['kind'] == 'datasets.TFRecordDataset':
        config.pop('kind')
        return TFRecordDataset(**config)

    raise ValueError('unknown dataset kind %s' % config['kind'])