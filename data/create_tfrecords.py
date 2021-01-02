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
import sys, traceback
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["chunks", "documents"], default="documents",
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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def chunks(l, n):
    # Divides a list into chunks
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out


def read_in_chunks(stream, chunk_size=2048):
    # Read a stream in chunk_size sized chunks
    while True:
        data = stream.read(chunk_size)
        if len(data) == 0:
            break
        yield data


def fetch_special_token_id(enc, special_token):
    ids = enc.encode(special_token).ids
    assert len(ids) == 1, f'Special token {special_token} is not assigned a unique id for the tokenizer'
    return ids[0]


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


class EncodedCompressedReader:
    # Loads, encodes and concatenates the texts within a zst archive
    # Pass a archive, returns a stream of its concatenated contents
    def __init__(self, f, encoder, separator=None, fix=False, minimum_size=0):
        def _gen():
            # Generator yielding the files inside the archive as encoded streams
            g = Reader(f).stream_data()
            for s in g:
                yield BufferedEncodedStream(s, encoder, separator, fix, minimum_size, text_mode=True)

        self.g = _gen()

        try:
            self.current = next(
                self.g)  # Current is, whenever possible, pointing to the currently opened stream in the archive
        except StopIteration:
            self.current = None

    def read(self, size=None):
        if size < 0:
            size = None
        remaining = size
        data = []

        while self.current and (remaining > 0 or remaining is None):
            data_read = self.current.read(remaining or -1)
            if len(data_read) < remaining or remaining is None:  # Exhausted file
                try:
                    self.current = next(self.g)
                except StopIteration:
                    self.current = None
            if not remaining is None:
                remaining -= len(data_read)
            data.extend(data_read)

        return data


class EncodedConcatenatedFiles(object):
    # Stitches list of file names into a stream of properly encoded and seperated tokens
    # Pass in a list of files and it outputs a stream of their contents stitched together
    def __init__(self, fns, encoder, separator=None, fix=False, minimum_size=0):
        self.fs = list(reversed(fns))  # reversed because read() reads from the last element first
        self.enc = encoder
        self.separator = separator  # separator should be [tokens]
        self.fix = fix
        self.minimum_size = minimum_size

    def read(self, size=None):
        if size < 0:
            size = None
        remaining = size
        data = []

        while self.fs and (remaining > 0 or remaining is None):
            if isinstance(self.fs[-1], str):  # If the last element in the list is a string, it's an unopened file
                if self.fs[-1].endswith(".zst") or self.fs[-1].endswith(
                        ".xz") or self.fs[-1].endswith(
                        "tar.gz"):  # If it's an archive, we use this reader
                    self.fs[-1] = EncodedCompressedReader(self.fs[-1], self.enc, self.separator, self.fix,
                                                          self.minimum_size)
                else:  # Otherwise we assume it's a normal text file
                    self.fs[-1] = BufferedEncodedStream(self.fs[-1], self.enc, self.separator, self.fix,
                                                        self.minimum_size)

            data_read = self.fs[-1].read(remaining or -1)
            if len(
                    data_read) < remaining or remaining is None:  # If we exhaust the file we're reading, pop it off the list
                self.fs.pop()

            if not remaining is None:
                remaining -= len(data_read)
            data.extend(data_read)

        return np.array(data, np.int32)


def create_file(params):
    idx, fns = params
    s = args.name + "_" + str(idx) + ".tfrecords"
    if os.path.exists(os.path.join(args.log_dir,
                                   s)):  # Hack-y, if file of same name is in log dir, sign that the file is complete, so skip
        return 0
    if os.path.exists(os.path.join(args.output_dir, s)):  # Unfinished file, remove
        os.remove(os.path.join(args.output_dir, s))

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, s)) as writer:
        def _write_to_file(data, i):
            # Helper function to avoid code duplication, writes the data as an example to the file and increments i
            # hash = fn.split("/")[-1].split(".")[0]
            feature = {
                # "hash": _bytes_feature(hash.encode()),
                "text": _int64_feature(data)
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            i += 1

        i = 0  # In document mode: Good files, in chunk mode: Number of chunks
        if args.mode == "documents":
            def _archive_to_files(f):
                # Generator that yields the contents of the files in an archive
                g = Reader(f).stream_data()
                for s in g:
                    yield BufferedEncodedStream(s, enc, [], not args.no_ftfy, args.minimum_size, text_mode=True).read()

            for fn in fns:
                if fn.endswith(".zst") or fn.endswith(".xz") or fn.endswith("tar.gz"):
                    data = _archive_to_files(fn)
                else:
                    data = [BufferedEncodedStream(fn, enc, args.separator, not args.no_ftfy, args.minimum_size).read()]

                for d in data:
                    _write_to_file(d, i)

        elif args.mode == "chunks":
            data_stream = EncodedConcatenatedFiles(fns, enc, separator=args.separator, fix=not args.no_ftfy,
                                                   minimum_size=args.minimum_size)
            data_stream = read_in_chunks(data_stream, args.chunk_size)
            for chunk in data_stream:
                if not chunk.shape[0] == args.chunk_size:  # Additional sanity check
                    continue
                _write_to_file(chunk, i)

    # File complete
    if args.mode == "documents":
        with open(os.path.join(args.log_dir, s), "w") as f:  # Create mark that file is finished in logdir
            f.write("{} / {}".format(i, len(fns)))  # How many files were good
        with open(os.path.join(args.log_dir, "good_files.log"), "a") as f:
            f.write("{}: {} / {}".format(idx, i, len(fns)))

    elif args.mode == "chunks":
        with open(os.path.join(args.log_dir, s), "w") as f:  # Create mark that file is finished in logdir
            f.write("{}".format(i))  # How many chunks
        with open(os.path.join(args.log_dir, "chunks.log"), "a") as f:
            f.write("{}: {}".format(idx, i))

    return i


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

    start = time.time()
    pool = Pool(processes=args.processes)
    ret = 0
    for i in tqdm(pool.imap(create_file, enumerate(file_chunks)), total=len(file_chunks)):
        ret += i
    end = time.time()

    if args.mode == "documents":
        print("Done! In {:.2f}s, {} / {} good files.".format(end - start, ret, len(files)))
    elif args.mode == "chunks":
        print("Done! In {:.2f}s, {} chunks.".format(end - start, ret))

    if args.write_dataset_config:
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

