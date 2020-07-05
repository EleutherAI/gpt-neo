from __future__ import print_function
from __future__ import division

from glob import glob
import os.path as op
import argparse, time, tarfile
import multiprocessing as mpl
from hashlib import md5

import os
import tarfile
import re

import newspaper

# Adapted from https://github.com/jcpeterson/openwebtext
# Use his downloader script to get archives

parser = argparse.ArgumentParser()
parser.add_argument("--html_archive", type=str, default="openwebtext/RS_2017-04-4_data.xz")
parser.add_argument("--chunk_size", type=int, default=100)
parser.add_argument("--n_procs", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="parsed")
args = parser.parse_args()


def parse_file(file_entry):
    file_name, html = file_entry
    url_hash = md5(html).hexdigest()
    article = newspaper.Article(url=url_hash, fetch_images=False)
    article.set_html(html)
    article.parse()
    return (file_name, article.text)

def save_parsed_text(parsed_entries, out_dir):
    for fn, txt in parsed_entries:
        txt_fp = op.join(out_dir, fn)
        with open(txt_fp, "w") as handle:
            handle.write(txt)


def get_processed_files(out_dir):
    parsed = glob(op.join(out_dir, "*.txt"))
    return set([op.split(f)[-1] for f in parsed])


def parse_archive(archive_fp, out_dir, n_procs, chunk_size=100):
    processed = get_processed_files(out_dir)
    with tarfile.open(archive_fp, "r") as tf:
        files = list(set(tf.getnames()) - set(processed))
        if len(files) == 0:
            return

        if len(processed) > 0:
            print("{} files already processed.".format(len(processed)))

        # pool = mpl.Pool(n_procs)
        for ci, chunk in enumerate(chunks(files, chunk_size)):
            file_entries = [(fn, tf.extractfile(fn).read()) for fn in chunk]

            t1 = time.time()
            parsed = list(map(parse_file, file_entries)) # parsed = list(pool.imap(parse_file, file_entries, chunksize=1))

            # remove empty strings from output
            parsed = [p for p in parsed if len(p[1]) != 0]

            hit_rate = len(parsed) / len(chunk) * 100
            print("Parsing chunk {} took {} seconds".format(ci + 1, time.time() - t1))
            print(" -- {}% of chunk {}'s docs yielded text.".format(hit_rate, ci + 1))

            t1 = time.time()
            save_parsed_text(parsed, out_dir)
            print("Saving chunk {} took {} seconds".format(ci + 1, time.time() - t1))

def extract_month(url_file_name):
    month_re = r"(RS_.*2\d{3}-\d{2})"
    month = op.split(url_file_name)[-1]
    month = re.match(month_re, month).group()
    return month


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def extract_archive(archive_fp, outdir="."):
    with tarfile.open(archive_fp, "r") as tar:
        tar.extractall(outdir)
    return outdir


def mkdir(fp):
    if not op.exists(fp):
        os.makedirs(fp)
    return fp

if __name__ == "__main__":
    month = extract_month(args.html_archive)
    out_dir = mkdir(op.join(args.output_dir, month))
    parse_archive(args.html_archive, out_dir, args.n_procs, args.chunk_size)
    print("Done!")