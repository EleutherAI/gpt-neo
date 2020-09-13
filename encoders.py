from tokenizers import Tokenizer
from transformers import GPT2Tokenizer

DEFAULT_TOKENIZER_PATH = "datasets/openwebtext/byte-level-bpe.tokenizer.json"


def fetch_encoder(params):
    no_dataset = params.get('no_dataset', False)
    if no_dataset:
        return None

    dataset = next(iter(params['dataset_configs'].values()))
    path = dataset['tokenizer_path']
    is_pretrained = dataset.get('tokenizer_is_pretrained', False)

    if is_pretrained:
        tok = GPT2Tokenizer.from_pretrained(path)

        # will add a padding token id of 50257 at run-time
        tok.add_special_tokens({'pad_token': ' '})
        return tok

    return Tokenizer.from_file(path)


# GPT2Tokenizer and Tokenizer has different ways of fetching token ids
def encode(encoder, text):
    result = encoder.encode(text)
    if isinstance(result, list):
        return result
    return result.ids
