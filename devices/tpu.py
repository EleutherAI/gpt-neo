from typing import Callable, Any, Optional, Dict

from pydantic import AnyUrl
from pydantic.dataclasses import dataclass

from absl import logging

class Device:
    pass

@dataclass
class TPUConfig:
    name: Optional[str] = None
    address: Optional[AnyUrl] = None
    num_cores: int = 8

@dataclass
class JobSpec:
    iterations: int # after how many internal iterations should the control come back
    model_path: AnyUrl
    function: Callable[[Dict[str,Any]], Any]
    args: Dict
    staging_location: AnyUrl
    batch_size: int

class TPU:
    def __init__(self, config: TPUConfig):
        self.config = config
        self._cluster = None 

    def resolve(self):
        if not self._cluster:
            self._cluster = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu) 

    def check_connection(self):
        pass

    def execute(self, job: JobSpec):
        
        run_config = tpu_config.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=jobspec.model_path,
            save_checkpoints_steps=None,  # Disable the default saver
            save_checkpoints_secs=None,  # Disable the default saver
            log_step_count_steps=job.iterations,
            save_summary_steps=job.iterations,
            tpu_config=tpu_config.TPUConfig(
                num_shards=job.mesh_shape.size,
                iterations_per_loop=job.iterations,
                num_cores_per_replica=1,
                per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

        estimator = tpu_estimator.TPUEstimator(
            use_tpu=True,
            model_fn=job.function,
            config=run_config,
            train_batch_size=infeed.batch_size,
            eval_batch_size=None,
            predict_batch_size=None,
            params=job.params)

        while current_step < job.steps:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(input_fn, eval=False), max_steps=job.steps)
            current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(job.staging_location))
            logging.info('step {}', current_step)