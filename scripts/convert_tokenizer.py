#!/usr/bin/env python
import os

import tokenizers

from absl import app, logging
from absl.flags import argparse_flags

import tensorflow as tf

def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("input", type=str, help="Name or path of a tokenizer FILE spec")
    parser.add_argument("output", type=str, help="Path of a tokenizer spec")
    args = parser.parse_args(argv[1:])
    return args

def convert_tokenizer(src, dst):
    if not tf.io.gfile.exists(src):
        raise ValueError('input file does not exists')
    if not tf.io.gfile.exists(dst):
        tf.io.gfile.mkdir(dst)
    elif tf.io.gfile.exists(dst) and tf.io.gfile.listdir(dst) != 0:
        raise ValueError('dst directory is not empty')
    try:
        tok = tokenizers.Tokenizer.from_file(src)
        tok.model.save(dst)
    except Exception as exc:
        logging.error('could not load tokenizer: %r', exc)
        exit()

def main(args):

    try: 
        convert_tokenizer(args.input, args.output)
        logging.info('tokenizer converted and saved to %s', args.output)
    except ValueError as e:
        logging.error('error: %r', str(e))

if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)
