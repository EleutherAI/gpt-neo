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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Location of the dataset files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)")
    parser.add_argument("--output", type=str, help="Location to write the generated tokenizer configuration")
    # parser.add_argument("--file_type", type=str, choices=["xz", "txt"], default="xz", help="Extension of file to parse")
    parser.add_argument("--vocab_size", type=int, help="Size of vocabulary", required = True)
    parser.add_argument("--random_seed", type=int, --seed=1337, help="seed")
    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()

    random.seed(args.random_seed)

    archives = list(glob(args.input))
    if not len(archives):
        archives = list(glob(os.path.join(args.input, '*.*')))
    
    if not len(archives):
        logging.error('no files found at location %s', args.input)
        return 
    
    data_files = random.sample(data_files, int(0.2 * len(data_files)))

    assert len(data_files) > 0, 'No data files found'

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # And then train
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    tokenizer.train(trainer, data_files)

    # And Save it
    tokenizer_path = out_path / "byte-level-bpe.tokenizer.json"
    tokenizer.save(tokenizer_path, pretty=True)
    logging.info('tokenizer saved at %s', tokenizer_path)

if __name__ == "__main__":
    app.run(main)