from tokenizers import Tokenizer
from transformers import GPT2Tokenizer
from pydantic.dataclasses import dataclass
from pydantic import AnyUrl

@dataclass
class EncoderConfig:
    is_pretrained: bool 
    location: AnyUrl

def fetch_encoder(config: EncoderConfig):
    if config.is_pretrained:
        return GPT2Tokenizer.from_pretrained(config.location)

    return Tokenizer.from_file(config.location)

# GPT2Tokenizer and Tokenizer has different ways of fetching token ids
def encode(encoder, text):
    result = encoder.encode(text)
    if isinstance(result, list):
        return result
    return result.ids
