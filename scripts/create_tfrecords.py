import argparse
import json
import os
import time
import random
from multiprocessing import Pool, cpu_count
from glob import glob

import ftfy
import numpy as np
import tensorflow as tf
from lm_dataformat import Reader
from tokenizers import Tokenizer
from tqdm import auto as tqdm
from absl import app, logging
from absl.flags import argparse_flags
import farmhash

from datasets import pipeline

def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["chunks", "documents"], default="documents", help="Whether a tfrecord example is a constant sized chunk or a full document")
    parser.add_argument("--input", type=str, default="/home/GPTNeo/LLMD-CommonCrawl/openwebtext", help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.")
    parser.add_argument("--files_per", type=int, default=200, help="Number of text files per tfrecord")
    parser.add_argument("--name", type=str, default="openwebtext", help="Name of output files will be {name}_%05d.tfrecord where i is the number of the file")
    parser.add_argument("--staging", type=str, default="staging", help="Where to write tfrecords being built")
    parser.add_argument("--output", type=str, default="output", help="Where to write tfrecords")
    parser.add_argument("--summaries", type=str, default="summaries", help="Where to put logs")
    # parser.add_argument("--processes", type=int, default=8, help="How many subprocesses to spawn. Should be ~number of cores")
    parser.add_argument("--tokenizer", type=str, default="byte-level-bpe.tokenizer.json", help="Name or path of a tokenizer spec")
    parser.add_argument("--min_seq_len", type=int, default=100, help="Minimum size a document has to be to be included")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="max seq length to feed to the transformer. should be the same as the input dimension")
    parser.add_argument("--fix_unicode", action="store_true", help="If set fix unicode normalization with ftfy")
    parser.add_argument("--separator", type=str, default="[0]", help="Seperator to place between files in chunk mode")
    parser.add_argument("--chunk_size", type=int, default=1024, help="How big a chunk should be in chunk mode")
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    args = parser.parse_args(argv[1:])
    return args

# Helper functions and classes
def chunks(l, n):
    # Divides a list into chunks
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out

def read_in_chunks(stream, chunk_size=1024):
    # Read a stream in chunk_size sized chunks
    while True:
        data = stream.read(chunk_size)
        if len(data) == 0:
            break
        yield data

def chunk_list(l, n):
    # Divides a list l into n chunks
    return [ l[i:i + n] for i in range(0, len(l), n) ]

def readlines_txt(src):
    with open(src) as fd:
        return fd.readlines()

LINE_READER = {
    '.txt': readlines_txt,
    '.tsv': readlines_txt,
    # '.ztx':
}
def readlines(src):
    _, ext = os.path.splitext(src)
    f = LINE_READER.get(ext, None)
    if f is None:
        logging.warning('no readlines for file %s', src)
        return
    return f(src)

END_OF_TEXT_TOKEN_ID = 0
def pad_sequence(buff, missing_elements):
    buff.extend([END_OF_TEXT_TOKEN_ID] * missing_elements)
    return buff

def token_generator(tokenizer, sources, max_seq_len):
    buff = []
    for src in sources:
        for line in readlines(src):
            enctext = tokenizer.encode(line)
            if not buff:
                # skip the line if is too long
                # most likely it does not make sense
                if len(enctext) > max_seq_len:
                    continue
                else:
                    buff = enctext.ids
                    continue

            if len(buff) + len(enctext.ids) > max_seq_len:
                padded = pad_sequence(buff, max_seq_len - len(buff))
                yield padded
                buff = enctext.ids
            else:
                buff.extend(enctext.ids)
        # flush buffer when finish one file
        if buff:
            padded = pad_sequence(buff, max_seq_len - len(buff))
            yield padded
            buff = []
    if buff:
        padded = pad_sequence(buff, max_seq_len - len(buff))
        yield padded
        buff = []


def transform_many_and_write_one_tfrecord(job):
    tokenizer, max_seq_len, sources, dst = job
    with tf.io.TFRecordWriter(dst) as w:
        for example_tokens in token_generator(tokenizer, sources, max_seq_len):
            text = tokenizer.decode(example_tokens)
            eid = farmhash.fingerprint64(text)
            example = pipeline.create_example(eid, example_tokens)
            w.write(example.SerializeToString())
    return len(sources)

def parallel(src_dst_list, total):
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    ret = 0
    for i in tqdm.tqdm(pool.imap(transform_many_and_write_one_tfrecord, src_dst_list), total=total):
        ret += i
    return ret

def load_tokenizer(location):
    return Tokenizer.from_file(location)

def listfiles(location):
    txt_files = list(p for p in glob(location) if not os.path.isdir(p))

    # try with general glob 
    if not txt_files:
        txt_files = list(glob(os.path.join(location, '*.*')))

    txt_files = list(p for p in txt_files if not os.path.isdir(p))
    return txt_files

def main(args):
    random.seed(args.random_seed)
    tf.random.set_random_seed(args.random_seed)

    txt_files = listfiles(args.input)  
    if not txt_files:
        logging.error('no data files found')
        return

    os.makedirs(args.summaries, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    args.separator = json.loads(args.separator) # Encode the separator to list

    file_chunks = chunks(txt_files, args.files_per) # Assign files_per file to a tfrecord file each
    args.chunk_size = args.chunk_size + 1 # Chunks need to be 1 token longer so there's a target for the last token

    logging.info("Got %d files, divided into %d chunks.", len(txt_files), len(file_chunks))

    def getdst(name, idx, total):
        return os.path.join(args.output, "%s_%05d_%05d.tfrecord" % (name, idx, total))

    tokenizer.enable_truncation(max_length=1024)
   
    jobs = ( (tokenizer, 
                args.max_seq_len,
                chunks, 
                getdst(args.name, idx, len(file_chunks))) for idx, chunks in enumerate(file_chunks) )

    #print(list(jobs))

    start = time.time()
    ret = parallel(jobs, total=len(txt_files))
    end = time.time()

    logging.info("Done! In %.2fs, %d / %d good files.", end-start, ret, len(txt_files))


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)