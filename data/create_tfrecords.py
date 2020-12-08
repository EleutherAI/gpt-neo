import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import ftfy
import numpy as np
import tensorflow as tf
from lm_dataformat import Reader
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tqdm import tqdm
from encoders import encode
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["documents"], default="documents",
                    help="Whether a tfrecord example is a constant sized chunk or a full document")
parser.add_argument("--base_dir", type=str, help="Path to where your files are located. Files ending in .zst are treated as \
                    archives, all others as raw text.")
parser.add_argument("--files_per", type=int, default=200, help="Text files per tfrecord")
parser.add_argument("--name", type=str, default="openwebtext",
                    help="Name of output files will be name_i.tfrecords where i is the number of the file")
parser.add_argument("--output_dir", type=str, default="tfrecords", help="Where to put tfrecords")
parser.add_argument("--log_dir", type=str, default="logs", help="Where to put logs")
parser.add_argument("--processes", type=int, default=8,
                    help="How many subprocesses to spawn. Should be ~number of cores")
parser.add_argument("--encoder_path", type=str, default="byte-level-bpe.tokenizer.json", help="Path to encoder files")
parser.add_argument("--use_gpt2_tokenizer", action="store_true", help="Use GPT2 tokenizer as encoder")
parser.add_argument("--minimum_size", type=int, default=100, help="Minimum size a document has to be to be included")
parser.add_argument("--no_ftfy", action="store_true", help="If set skips unicode normalization with ftfy")
parser.add_argument("--separator", type=str, default="[0]", help="separator to place between files in chunk mode")
parser.add_argument("--chunk_size", type=int, default=2048, help="How big a chunk should be in chunk mode")
parser.add_argument("--write_dataset_config", action="store_true", help="Write the dataset config file on completion")
args = parser.parse_args()


# Helper functions and classes


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def chunks(l, n):
    # Divides a list into chunks
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out


def fetch_special_token_id(enc, special_token):
    ids = enc.encode(special_token).ids
    assert len(ids) == 1, f'Special token {special_token} is not assigned a unique id for the tokenizer'
    return ids[0]


def write_dataset_config():
    write_file_path = f'./configs/dataset_configs/{args.name}.json'
    with open(write_file_path, 'w') as f:
        output_path = Path(args.output_dir)
        dataset_config = {
            "path": str(output_path / f"{args.name}_*.tfrecords"),
            "eval_path": "",
        }
        if args.use_gpt2_tokenizer:
            dataset_config.update(**{
                "n_vocab": 50256,
                "tokenizer_is_pretrained": True,
                "tokenizer_path": "gpt2",
                "eos_id": 50256,
                "padding_id": 50257
            })
        else:
            dataset_config.update(**{
                "n_vocab": enc.get_vocab_size(),
                "tokenizer_path": str(args.encoder_path),
                "eos_id": fetch_special_token_id(enc, "<|endoftext|>"),
                "padding_id": fetch_special_token_id(enc, "<|padding|>")
            })
        f.write(json.dumps(dataset_config, indent=2))
    print(f'Dataset config written to {write_file_path}!')
    logging.getLogger("transformers").setLevel(logging.WARNING)


class BufferedEncodedStream(object):
    # Loads a file into memory, optionally fixes unicode, encodes it and adds the separator to the beginning
    # If set to text_mode the input is assumed to not be a file but the direct string data
    def __init__(self, inp, encoder, separator=None, fix=False, minimum_size=0, text_mode=False):
        if text_mode:
            d = inp
        else:
            with open(inp, "r") as f:
                d = f.read()

        if fix:
            d = ftfy.fix_text(d, normalization='NFKC')

        if args.use_gpt2_tokenizer:
            self.data = encode(encoder, d, gpt=True)
        else:
            self.data = encode(encoder, d)

        if len(self.data) < minimum_size or all([x == 0 for x in self.data]):  # Sanity check
            self.data = []  # Don't return file contents if it doesn't pass the sanity check
        elif separator is not None:  # Only add separator if sanity check didn't failt
            self.data = separator + self.data  # separator should be [tokens]

        self.idx = 0
        self.n = len(self.data)

    def read(self, size=None):
        if self.idx < self.n:
            if size is None or size < 0:
                chunk = self.data[self.idx:]
                self.idx = self.n
            else:
                chunk = self.data[self.idx:self.idx + size]
                self.idx += len(chunk)
            return chunk
        else:
            return []


def create_file(params):
    idx, fns = params
    s = args.name + "_" + str(idx) + ".tfrecords"
    fp = os.path.join(args.log_dir, s)
    if os.path.exists(fp):  # Hack-y, if file of same name is in log dir, sign that the file is complete, so skip
        logging.warning(f"{fp} already exists according to log dir - skipping file")
        return 0
    if os.path.exists(os.path.join(args.output_dir, s)):  # Unfinished file, remove
        logging.warning(f"{fp} already exists but is unfinished - removing file")
        os.remove(os.path.join(args.output_dir, s))

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, s)) as writer:
        def _write_to_file(data):
            # Helper function to avoid code duplication, writes the data as an example to the file and increments i
            feature = {
                "text": _int64_feature(data)
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())

        if args.mode == "documents":
            def _archive_to_files(f):
                # Generator that yields the contents of the files in an archive
                g = Reader(f).stream_data(threaded=False)
                for s in g:
                    yield BufferedEncodedStream(s, enc, [], not args.no_ftfy, args.minimum_size, text_mode=True).read()

            succesful_files = 0
            for fn in fns:
                if fn.endswith(".zst") or fn.endswith(".xz") or fn.endswith("tar.gz"):
                    data = _archive_to_files(fn)
                else:
                    data = [BufferedEncodedStream(fn, enc, [], not args.no_ftfy, args.minimum_size).read()]

                for d in data:
                    if d != []:
                        _write_to_file(d)
                        succesful_files += 1

    # File complete
    if args.mode == "documents":
        with open(os.path.join(args.log_dir, s), "w") as f:  # Create mark that file is finished in logdir
            f.write("{} / {}".format(succesful_files, len(fns)))  # How many files were good
        with open(os.path.join(args.log_dir, "good_files.log"), "a") as f:
            f.write("{}: {} / {}".format(idx, succesful_files, len(fns)))

    return succesful_files


if __name__ == "__main__":
    Path(args.log_dir).mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(exist_ok=True)

    if args.use_gpt2_tokenizer:
        enc = GPT2TokenizerFast.from_pretrained('gpt2')
    else:
        enc = Tokenizer.from_file(args.encoder_path)

    args.separator = json.loads(args.separator)  # Encode the separator to list
    files = glob.glob(os.path.join(args.base_dir, "*"))  # TODO make this more flexible maybe?
    files = [f for f in files if not os.path.isdir(f)]
    file_chunks = chunks(files, args.files_per)  # Assign files_per file to a tfrecord file each
    args.chunk_size = args.chunk_size + 1  # Chunks need to be 1 token longer so there's a target for the last token

    print("Got {} files, divided into {} chunks.".format(str(len(files)), str(len(file_chunks))))

    # do tokenization
    start = time.time()
    pool = Pool(processes=args.processes)
    total_successful = 0
    for n_successful in tqdm(pool.imap(create_file, enumerate(file_chunks)), total=len(file_chunks)):
        total_successful += n_successful
    end = time.time()

    print("Done! In {:.2f}s, {} / {} good files.".format(end - start, total_successful, len(files)))

    if args.write_dataset_config:
        write_dataset_config()
