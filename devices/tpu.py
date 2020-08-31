import tensorflow as tf
from typing import Callable, Any, Optional, Dict, Union
from pydantic import AnyUrl
from pydantic.dataclasses import dataclass
import ipaddress

from absl import logging
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
import functools

"""
TPU Configuration Module
"""

class Device:
    pass

@dataclass
class TPUConfig:
    address: Optional[Union[str, ipaddress.IPv4Address]] = None
    num_cores: int = 8

@dataclass
class TPUJobSpec:
    steps_per_iteration: int 
    steps_per_checkpoint: int 
    max_steps: int
    model_path: str
    function: Callable[[Dict[str,Any]], Any]
    params: Dict
    batch_size: int
    train: bool = False
    test: bool = False
    predict: bool = False

class TPU:
    def __init__(self, config: TPUConfig):
        self.config = config
        self._cluster = None 

    def resolve(self):
        if not self.config.address: return

        if not self._cluster:
            self._cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=self.config.address
            ) 
        return self._cluster

    def check_connection(self):
        pass

    def execute(self, job: TPUJobSpec):
        cluster = self.resolve()

        run_config = tpu_config.RunConfig(
            cluster=self._cluster,
            model_dir=job.model_path,
            save_checkpoints_steps=None,  # Disable the default saver
            save_checkpoints_secs=None,  # Disable the default saver
            log_step_count_steps=job.steps_per_iteration,
            save_summary_steps=job.steps_per_checkpoint,
            tpu_config=tpu_config.TPUConfig(
                num_shards=job.function.mesh_shape.size,
                iterations_per_loop=job.steps_per_iteration,
                num_cores_per_replica=1,
                per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

        estimator = tpu_estimator.TPUEstimator(
            use_tpu=True,
            model_fn=job.function,
            config=run_config,
            train_batch_size=job.batch_size,
            eval_batch_size=None,
            predict_batch_size=None,
            params=job.params)

        if job.train:
            if tf.io.gfile.exists(job.model_path):
                current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(job.model_path))
            else: 
                current_step = 0

            fn = functools.partial(job.function, eval=False)

            while current_step < job.max_steps:
                # Else, don't stop and restart
                estimator.train(input_fn=fn, max_steps=job.max_steps)
                current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(job.model_path))
                logging.info('step {}', current_step)

        raise ValueError