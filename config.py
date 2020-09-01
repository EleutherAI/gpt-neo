import os
import sys
import json

import tensorflow as tf
import _jsonnet
from absl import app
from absl.flags import argparse_flags
from absl import logging

"""
utility module to deal with resolving the system configuration
"""

def load(resource:str):
    if tf.io.gfile.isdir(resource):
        resource = os.path.join(resource, 'dataset.info.json')
        if not tf.io.gfile.exists(resource):
            raise ValueError('directory provided but directory does not contain a dataset.info.json file')
    path, ext = os.path.splitext(resource)
    if ext in ('.json', ):
        with tf.io.gfile.GFile(resource) as fd:
            params = json.load(fd)
    elif ext in ('.jsonnet', ):
        try:
            json_str = _jsonnet.evaluate_file(
                resource,
                ext_vars={'MODEL_PATH': 'Bob'},
                #import_callback=import_callback,
            )
            params = json.loads(json_str)
        except RuntimeError as e:
            logging.error(e)
            sys.exit(-1)
    else:
        raise ValueError
    return params