from tokenizers import Tokenizer
from transformers import GPT2Tokenizer

DEFAULT_TOKENIZER_PATH = "datasets/openwebtext/byte-level-bpe.tokenizer.json"

def fetch_encoder(params):
    tokenizer_path = params.get('tokenizer_path', None)
    if tokenizer_path is not None:
        return Tokenizer.from_file(tokenizer_path)

    # if "tokenizer_path" is not supplied in the config, default to usual logic
    if params["n_vocab"] > 50257:
        return GPT2Tokenizer.from_pretrained('gpt2')

    enc = Tokenizer.from_file(DEFAULT_TOKENIZER_PATH)
    return enc

# GPT2Tokenizer and Tokenizer has different ways of fetching token ids
def encode(encoder, text):
    result = encoder.encode(text)
    if isinstance(result, GPT2Tokenizer):
        return result
    return result.ids
