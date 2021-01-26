"""the_pile dataset"""

import argparse
import io
import os
import shutil
import time

import jsonlines
import numpy as np
import requests
import simdjson
import tensorflow as tf
import zstandard
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="text",
                    help="Name of output files will be name_i.tfrecords where i is the number of the file")
parser.add_argument("--output_dir", type=str, default="gs://tfrecords/a/", help="Where to put tfrecords (in a bucket)")
parser.add_argument("--int64", type=bool, default=False, help="Whether to encode as bytes or int64")
parser.add_argument("--service_account_json_path", type=str, default="./tfrecords", help="Service account json from"
                                                                                         " gcp")
parser.add_argument("--buffer_size", type=int, default=2 ** 28, help="This is a minimum size, not a maximum size. "
                                                                     "tfrecords will have this minimum size as well.")
parser.add_argument("--separator", nargs="+", type=int, default=4,
                    help="separator to place between files in chunk mode."
                         "Default is 0 (Null) in case of byte encodings, "
                         "50256 for tokenized texts")


def file_generator(args):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = ".tmp.download"

    def _json_parser(x):
        try:
            return parse_fn(x).as_dict()
        except ValueError:
            return x

    for i in range(splits):
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with open(tmp_name, 'rb') as f:
            for item in jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)),
                                         loads=_json_parser):
                if isinstance(item, dict):
                    item = item['text']
                if isinstance(item, list):
                    item = chr(args.separator).join(item)
                yield item.encode()


def create_tfrecords(args):
    slash_idx = args.output_dir.find('/')
    bucket_name, output_dir = args.output_dir[:slash_idx], args.output_dir[slash_idx + 1:]
    bucket = storage.Client.from_service_account_json(args.service_account_json_path).get_bucket(bucket_name)
    join = bytes([args.separator]).join
    prefix = f"{'int64' if args.int64 else 'bytes'}_{args.name}_"

    files_processed = 0
    tfrecord_count = 0
    chunk = 0
    buffer_size = 0
    tokenized_files = []

    last_write = start_time = time.time()

    for f in file_generator(args):
        buffer_size += len(f)
        tokenized_files.append(f)
        files_processed += 1

        if buffer_size > chunk * args.buffer_size // 32:
            print(f"\rBuffer: {buffer_size * 2 ** -20:.1f}MB | "
                  f"Files: {files_processed} - TFrecords: {tfrecord_count} | "
                  f"Wrote: {time.time() - last_write:.0f}s ago - Started: {time.time() - start_time:.0f}s ago",
                  end='')
            chunk += 1

        if buffer_size > args.buffer_size:
            filename = f"{prefix}{tfrecord_count}_{files_processed}_{buffer_size}.tfrecord"

            joined = join(tokenized_files)
            tokenized_files.clear()

            with tf.io.TFRecordWriter(filename) as writer:
                if args.int64:
                    joined = np.cast(np.frombuffer(joined, np.uint8).reshape(-1), np.int64)
                    feature = {"text": tf.train.Feature(int64_list=tf.train.Int64List(value=joined))}
                else:
                    feature = {"text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[joined]))}
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

            bucket.blob(f'{output_dir}{filename}').upload_from_filename(filename)

            os.remove(filename)
            chunk = 0
            buffer_size = 0
            tfrecord_count += 1

            print("")

            last_write = time.time()


def main():
    args = parser.parse_args()

    if not args.output_dir.endswith("/"):
        args.output_dir = args.output_dir + "/"
    if not args.output_dir.startswith("gs://"):
        print("Output dir isn't a cloud bucket. Exiting.")
        return
    args.output_dir = args.output_dir[len('gs://'):]
    create_tfrecords(args)


if __name__ == "__main__":
    main()
