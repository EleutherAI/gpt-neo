import os
import random
import argparse
import shutil
from glob import glob
from pathlib import Path

from lm_dataformat import Reader
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from tqdm import auto as tqdm
from absl import app, logging
from absl.flags import argparse_flags

def parse_flags(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Location of the dataset files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)")
    parser.add_argument("--output", type=str, required=True, help="Location to write the generated tokenizer configuration")
    parser.add_argument("--vocab_size", type=int, help="Size of vocabulary", required = True)
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    args = parser.parse_args(argv[1:])
    return args

def listfiles(location):
    txt_files = list(p for p in glob(location) if not os.path.isdir(p))

    # try with general glob 
    if not txt_files:
        txt_files = list(glob(os.path.join(location, '*.*')))

    txt_files = list(p for p in txt_files if not os.path.isdir(p))
    return txt_files

def main(args):

    random.seed(args.random_seed)

    txt_files = listfiles(args.input)  
    if not txt_files:
        logging.error('no data files found')
        return

    os.makedirs(args.output, exist_ok=True)

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # And then train
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    tokenizer.train(trainer, txt_files)

    # And Save it
    tokenizer_path = os.path.join(args.output, "byte-level-bpe.tokenizer.json")
    tokenizer.save(tokenizer_path, pretty=True)
    encoded_gold = tokenizer.encode("I can feel the magic, can you?")
    logging.info('tokenizer saved at %s', tokenizer_path)

    # Test it by loading it back 
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode("I can feel the magic, can you?")

    if not all(a == b for a,b in zip(encoded.ids, encoded_gold.ids)):
        logging.error("saved tokenizer and trained tokenizer do not match")

if __name__ == "__main__":
    app.run(main, flags_parser=parse_flags)