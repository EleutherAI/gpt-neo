from tokenizers import Tokenizer
from transformers import GPT2Tokenizer

def fetch_encoder(params):
    if params.get('datasets') is None:
      return

    dataset = params['datasets'][0]
    if 'tokenizer_path' not in dataset:
      path = 'gpt2'
      is_pretrained = True
    else:
      path = dataset['tokenizer_path']
      is_pretrained = dataset.get('tokenizer_is_pretrained', False)

    if is_pretrained:
        return GPT2Tokenizer.from_pretrained(path)

    return Tokenizer.from_file(path)

# GPT2Tokenizer and Tokenizer has different ways of fetching token ids
def encode(encoder, text):
    result = encoder.encode(text)
    if isinstance(result, list):
        return result
    return result.ids
