import json
from typing import Any, Dict, Optional, Union

import cmd
import mesh_tensorflow as mtf
import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import app, logging
from absl.flags import argparse_flags
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

import config
import models
import inputs
from devices.tpu import TPUJobSpec, TPUInfeedSpec
import devices


@dataclass
class ScheduleSpec:
    steps: int

@dataclass
class EvalConfig:
    infeed: Dict
    model: Dict
    model_path: str
    schedule: ScheduleSpec
    device: devices.DeviceSpec = devices.CPUDeviceSpec()

    @classmethod
    def from_config(cls, location):
        config_dict = config.load(location)
        return cls(**config_dict)

class Evaluator:
    def __init__(self, config:EvalConfig):
        self.config = config
        self.model = None
        self.infeed = None
        self.device = None

    def load_model(self):
        if not (self.model is None):
            return self.model
        self.model = models.from_config(self.config.model)
        return self.model

    def load_infeed(self):
        if not (self.infeed is None):
            return self.infeed
        self.infeed = inputs.from_config(self.config.infeed)
        return self.infeed

    def create_jobspec(self):
        model = self.load_model()
        infeed = self.load_infeed()
        return TPUJobSpec(
            function=self.model,
            params={ 
                # patch legacy config
                'eval_steps': self.config.schedule.steps,
                'model_path': self.config.model_path,
                'steps_per_iteration': self.config.schedule.steps,
                'steps_per_checkpoint': self.config.schedule.steps,
            },
            max_steps=self.config.schedule.steps,
            use_tpu=type(self.config.device) is devices.TPUDeviceSpec,
            model_path=self.config.model_path,
            # steps_per_iteration=self.config.schedule.steps_per_iteration,
            # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
            infeed=TPUInfeedSpec(
                batch_size=infeed.config.batch_size, 
                function=infeed,
                params={}
            ),
        )

    def execute(self, jobspec):
        if self.device is None:
            self.device = devices.from_config(self.config.device)
        return self.device.execute(jobspec)

class InteractiveTask(cmd.Cmd):
    intro = 'Welcome to the gptneo shell. Type help or ? to list commands.\n'
    prompt = '(gptneo) '
    file = None

    def init(self, args):
        """
        Loads saved model and run inference on TPU.
        Args:
            inputs: dataset to convert
            saved_model_dir: The directory SavedModel being exported to.
        Returns:
            A dict of resulting tensors.
        """

        feature_spec = {
            "uid": tf.io.FixedLenFeature([], tf.string),
            #"inputs": tf.io.FixedLenFeature([1024], tf.int64),
            "content": tf.io.FixedLenFeature([], tf.string), # raw string 
            # "tokens": tf.io.VarLenFeature(tf.int64) # raw tokens (not splitted to 128)
        } 

        graph = tf.Graph()

        self._sess = sess = tf.InteractiveSession(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))

        meta_graph = loader.load(sess, [tag_constants.SERVING], args.saved_model)

        # describe_graph(meta_graph.graph_def)

        key_prediction = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        print(meta_graph.signature_def[key_prediction].inputs)
        
        input_placeholder_name = 'input'
        # meta graph inputs
        tensor_name_input = (
            meta_graph.signature_def[key_prediction].inputs[input_placeholder_name]
            .name)
        print(tensor_name_input)

        # meta graph outputs
        tensor_name_output = {
            k: v.name
            for k, v in (meta_graph.signature_def[key_prediction].outputs.items())
        }

        print(tensor_name_output)

        def transform(value):
            return sess.run( feed_dict={tensor_name_input: value} , fetches={ 'outputs': 'logits:0'})

        self.transform = transform
        return self
        # it = tf.compat.v1.data.make_initializable_iterator(ds)
        # init_op = it.initializer
        # example = it.get_next()
        # sess.run(init_op)
    def do_sum(self, arg):
        import numpy as np
        pad = 0 # pad token
        eos = 1 # end of sentence token
        bos = 2 # begin of sentence token
        SHIFT = 3
        oin = arg.split(' ')
        v = np.concatenate([ [bos], [int(v) + SHIFT for v in oin], [eos], [pad] * 8], axis=0)
        io = self.transform([v[:8]])
        max_value = np.argmax(io['outputs'], axis=-1)
        values = max_value - SHIFT #[:, 1:len(oin)] - SHIFT)
        print(' '.join(str(v) for v in values[values > 0]))

    def do_bye(self, arg):
        print('Thank you for using neogpt')
        return True
    def do_q(self, arg):
        print('Thank you for using neogpt')
        return True
    def do_quit(self, arg):
        print('Thank you for using neogpt')
        return True

def parse_args(args, parser=None):
    # Parse command line arguments
    parser.add_argument("saved_model", type=str, help="location of the task saved model")

def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])

def main(args):
    logging.info("started evaluation process")

    InteractiveTask().init(args).cmdloop()


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
