import argparse
import os
from pathlib import Path

import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat
import re

logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="Path to where your files are located. Files ending in .zst are "
                                                  "treated as archives, all others as raw text.")
parser.add_argument("--files_per", type=int, default=100000, help="Text files per tfrecord")
parser.add_argument("--name", type=str, default="openwebtext",
                    help="Name of output files will be name_i.tfrecords where i is the number of the file")
parser.add_argument("--output_dir", type=str, default="./tfrecords", help="Where to put tfrecords")
parser.add_argument("--encoder_path", type=str,
                    help="Path to encoder files, or leave unspecified to use GPT2 tokenizer")
parser.add_argument("--minimum_size", type=int, default=100, help="Minimum size a document has to be to be included")
parser.add_argument("--ftfy", action="store_false", help="normalize with ftfy")
parser.add_argument("--wikitext-detokenize", action="store_false", help="use wikitext detokenizer")
parser.add_argument("--separator", nargs="+", type=int, default=[50256],
                    help="separator to place between files in chunk mode")
parser.add_argument("--chunk_size", type=int, default=2048, help="How big a chunk should be in chunk mode. "
                                                                 "Should equal your model's context size")
parser.add_argument("--write_dataset_config", action="store_true", help="Write the dataset config file on completion")
parser.add_argument("--processes", type=int, default=0, help="Number of processes to use. Defaults to cpu count.")

args = parser.parse_args()
if not args.output_dir.endswith("/"):
    args.output_dir = args.output_dir + "/"
if not args.input_dir.endswith("/"):
    args.input_dir = args.input_dir + "/"
assert len(args.separator) == 1


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {
        "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def get_tokenizer(args):
    if args.encoder_path is None:
        return GPT2TokenizerFast.from_pretrained('gpt2')
    else:
        return Tokenizer.from_file(args.encoder_path)


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i + n] for i in range(0, len(l), n)]


def archive_to_tokens(f, encoder, args, prefix=[]):
    # Generator that yields the contents of the files in an archive
    # if data_to_prepend is not None, prepend data_to_prepend + a EOS separator to the encoded data
    reader = Reader(f)
    for doc in reader.stream_data(threaded=False):
        if args.ftfy:  # fix text with ftfy if specified
            doc = ftfy.fix_text(doc, normalization='NFKC')
        if args.wikitext_detokenize:
            doc = wikitext_detokenizer(doc)
        doc = encoder.encode(doc) + args.separator  # read document from lmd and append separator token
        yield split_list(prefix + doc, args.chunk_size)  # split into n_ctx + 1 size chunks
        prefix = []


def write_files(files, files_per, output_dir, out_name, start_no, write_remainder=False, process_no=None):
    # writes a list of files to .tfrecords
    if files == None:
        return
    chunks = split_list(files, files_per)
    if not chunks:
        return
      
    if len(chunks[-1]) != files_per and not write_remainder:  # pop the last file if it's length != files per
        remainder = chunks.pop(-1)
    else:
        remainder = None  # assuming files = remainder from an old chunk here
        files_per = len(chunks[-1])

    for files in chunks:
        fp = f"{output_dir}/{out_name}_{start_no}"
        if process_no is not None:
            fp += f"_{process_no}"
        fp += f"_{files_per}"  # add number of files in tfrecord to end of fp
        fp += ".tfrecords"
        with tf.io.TFRecordWriter(fp) as writer:
            for f in files:
                write_to_file(writer, f)
        start_no += 1
    return start_no, remainder


def get_files(input_dir, filetypes=None):
    # gets all files of <filetypes> in input_dir
    if filetypes == None:
        filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]
    files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
    # flatten list of list -> list and stringify Paths
    flattened_list = [str(item) for sublist in files for item in sublist]
    if not flattened_list:
        raise Exception(f"""did not find any files at this path {input_dir},\
 please also ensure your files are in format {filetypes}""")
    return flattened_list


def read_checkpoint(checkpoint_path, resume_from_checkpoint=True):
    # init checkpointing
    if resume_from_checkpoint and os.path.isfile(checkpoint_path):
        try:
            resume_files_processed, tfrecord_count = [int(i) for i in open(checkpoint_path, "r").read().split(", ")]
            print(f"\nResuming from tfrecord no. {tfrecord_count} / file no. {resume_files_processed}")
            return resume_files_processed, tfrecord_count
        except:
            pass
    return 0, 0


def create_tfrecords(params, write_remainder=True, write_every_n_files=1, save_checkpoints=False,
                     resume_from_checkpoint=False, display_pbar=False):
    # iterates through files in input_dir, splitting into <args.chunk_size> chunks and saving a tfrecords file every <args.files_per> chunks.
    files, args, process_no = params
    enc = get_tokenizer(args)  # get tokenizer

    # init metadata
    discarded_files = 0
    files_processed = 0
    pbar = tqdm(desc=f"Writing TFRecord Files to {args.output_dir}. Parsed 0 input files. files_written ",
                disable=not display_pbar)
    checkpoint_path = f"{args.output_dir}/checkpoint.txt"
    resume_files_processed, tfrecord_count = read_checkpoint(checkpoint_path, resume_from_checkpoint)

    data_to_prepend = []
    tokenized_files_array = []

    for f in files:
        for tokenized_files in archive_to_tokens(f, enc, args, prefix=data_to_prepend):
            files_processed += 1
            if files_processed < resume_files_processed:
                continue  # resume from checkpoint

            # if the last chunk < chunk size, but > minimum_size, take it and append it to the beginning of the next file
            data_to_prepend = []
            n_tokens = len(tokenized_files[-1])
            if n_tokens < args.chunk_size:
                data = tokenized_files.pop(-1)
                if n_tokens >= args.minimum_size:
                    data_to_prepend = data
                else:
                    discarded_files += 1

            # add tokenized files > chunk size to main array
            tokenized_files_array.extend(tokenized_files)

            if len(tokenized_files_array) >= args.files_per * write_every_n_files:  # write every n files
                _tfrecord_count, remainder = write_files(tokenized_files_array, files_per=args.files_per,
                                                         output_dir=args.output_dir, out_name=args.name,
                                                         start_no=tfrecord_count, process_no=process_no)
                pbar.update(_tfrecord_count - tfrecord_count)  # update progress bar
                pbar.set_description(
                    f"Writing TFRecord Files to {args.output_dir}. Parsed {files_processed} input files. files_written ")
                tfrecord_count = _tfrecord_count
                tokenized_files_array = remainder if remainder is not None else []  # add remaining files to next chunk
                with open(checkpoint_path, "w") as checkpoint_file:
                    checkpoint_file.write(f"{files_processed}, {tfrecord_count}")

    if len(tokenized_files_array) >= args.files_per:  # also write at end
        _tfrecord_count, remainder = write_files(tokenized_files_array, files_per=args.files_per,
                                                 output_dir=args.output_dir, out_name=args.name,
                                                 start_no=tfrecord_count, process_no=process_no)
        pbar.update(_tfrecord_count - tfrecord_count)
        pbar.set_description(
            f"Writing TFRecord Files to {args.output_dir}. Parsed {files_processed} input files. files_written ")
        tfrecord_count = _tfrecord_count
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(f"{files_processed}, {tfrecord_count}")
    else:
        remainder = tokenized_files_array  # add remaining to remainder

    if write_remainder:
        # write out the remaining files even if there's less than files_per
        write_files(remainder, files_per=args.files_per, output_dir=args.output_dir, out_name=args.name,
                    start_no=tfrecord_count, write_remainder=True)

    successful_files = files_processed - discarded_files
    return {"discarded": discarded_files, "processed": files_processed, "successful": successful_files}


def create_tfrecords_mp(files, args):
    files = split_list(files, len(files) // args.processes)
    with Pool(processes=args.processes) as pool:
        pbar = tqdm(pool.imap(create_tfrecords, zip(files, repeat(args), range(len(files)))))
        meta = {"discarded": 0, "processed": 0, "successful": 0}
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v  # update metadata
        return meta


if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)  # make output dir if it doesn't exist
    files = get_files(args.input_dir)
    args.chunk_size += 1  # we shift the data by 1 to the right for targets, so increment the chunk size here

    if args.processes == 0:
        args.processes = cpu_count()
    if args.processes > 1:
        results = create_tfrecords_mp(files, args)
    else:
        results = create_tfrecords((files, args, 0), display_pbar=True)
    print(results)
