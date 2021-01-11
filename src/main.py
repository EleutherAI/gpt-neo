import argparse
from functools import partial

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib

from .inputs import dataset, gpt_neo_input
from .train import model_fn
from .dataclass import ModelParameter
import json

def main(args: argparse.Namespace):
    # Setup logging
    model_path = args.model if args.model.endswith(".json") else f"configs/{args.model}.json"
    with open(model_path) as f:
        params = json.load(f)
    params = ModelParameter(params)
    # Read params of model

    # Fetch appropriate input functions

    if params.model_mode == 'jannet':
        input_fn = dataset(params, 0)
    elif params.model_mode == 'gpt':
        input_fn = gpt_neo_input(params, 0, eval=False)
    else:
        raise ValueError("model_mode need to be 'jannet' or 'gpt' {}, "
                         "is a not supported option.".format(params.model_mode))

    # Add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    params.num_cores = mesh_shape.size
    params.use_tpu = True if not args.tpu is None else False
    params.gpu_ids = args.gpu_ids
    # Expand attention types param
    params.predict = args.predict

    # Sample quality of MoE models suffers when using the faster sampling method, so default to slow_sampling if
    # moe layers are present
    params.slow_sampling = False

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    config = tpu_config.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=params.model_path,
            save_checkpoints_steps=None,  # Disable the default saver
            save_checkpoints_secs=None,  # Disable the default saver
            log_step_count_steps=params.iterations,
            save_summary_steps=params.iterations,
            tpu_config=tpu_config.TPUConfig(
                    num_shards=mesh_shape.size,
                    iterations_per_loop=params.iterations,
                    num_cores_per_replica=1,
                    per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

    estimator = tpu_estimator.TPUEstimator(
            use_tpu=params.use_tpu,
            model_fn=model_fn,
            config=config,
            train_batch_size=params.train_batch_size,
            predict_batch_size=1,
            params=params.dict())

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))

    while current_step < params.train_steps:
        # Else, don't stop and restart
        estimator.train(input_fn=partial(input_fn), max_steps=params.train_steps)
