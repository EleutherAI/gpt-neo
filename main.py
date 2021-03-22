"""GPT-like model in Mesh-Tensorflow"""

from functools import partial
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from utils import save_config, expand_attention_types_params, yes_or_no, remove_gs_or_filepath, setup_logging, \
    check_dataset
from inputs import sequential_input, pred_input, handle_pred_output, mlm_sample_text, generic_text
from export import export_model
from model_fns import model_fn
from data.encoders import fetch_encoder
from configs import fetch_model_params
from tasks import task_descriptors
import argparse
import json
import numpy


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.")
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=["device:GPU:0"],
                        help="If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=5000, help="Save a model checkpoint every X steps.")
    parser.add_argument("--auto_layout", action="store_true", help="If set, generates and prints the most memory "
                                                                   "efficient layout according to MTF auto layout.")
    parser.add_argument("--auto_layout_and_mesh_shape", action="store_true",
                        help="If set, generates and prints the most memory efficient layout and mesh shape according to"
                             " MTF auto layout.")
    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")
    parser.add_argument("--predict", action="store_true", help="If set, uses the model to predict rather than train.")
    parser.add_argument("--eval", action="store_true", help="If set, run model in evaluation mode.")
    parser.add_argument("--prompt", type=str, help="path to .txt file containing a prompt for prediction. If empty, "
                                                   "defaults to unicorns.",
                        default="")
    parser.add_argument("--check_dataset", action="store_true",
                        help="If set, outputs sample from the dataset and quits.")
    parser.add_argument("--sacred_id", type=str, default="nosacred", help="Sacred run id.")
    parser.add_argument("--entmax_sampling", action="store_true", help="(experimental) use entmax sampling")
    parser.add_argument("--export", action="store_true", help="If set, will export the model.")
    args = parser.parse_args()
    assert args.model is not None, "Model must be set"
    return args


def main(args):
    # Setup logging
    logger = setup_logging(args)

    # Read params of model
    params = fetch_model_params(args.model)

    # Fetch appropriate input functions
    input_fn = params.get("input_fn", "sequential_input")
    if input_fn == "sequential_input":
        input_fn = sequential_input
    elif input_fn == "generic_text":
        input_fn = generic_text
    pred_input_fn = pred_input
    handle_pred_output_fn = handle_pred_output

    # get current step
    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info(f"Current step {current_step}")

    if params["mlm_training"]:
        mlm_sample_text_fn = partial(mlm_sample_text, params)
        input_fn = partial(generic_text, sample_text_fn=mlm_sample_text_fn)
        if args.check_dataset:
            check_dataset(input_fn, params)


    # Fetch encoder per params
    encoder = fetch_encoder(params)

    pred_input_fn = partial(pred_input_fn, path_to_prompt=args.prompt, logger=logger, enc=encoder)

    # Sample from Dataset if check dataset flag is on
    if args.check_dataset:
        check_dataset(input_fn, params, global_step=current_step)

    # Confirm deletion of checkpoint files if --new flag is set
    if args.new:
        if yes_or_no(f"Are you sure you want to remove '{params['model_path']}' to start afresh?"):
            remove_gs_or_filepath(params["model_path"])
        else:
            exit()

    # Save config to logdir for experiment management
    save_config(params, params["model_path"])

    # Add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    params["num_cores"] = mesh_shape.size
    params["auto_layout"] = args.auto_layout
    params["auto_layout_and_mesh_shape"] = args.auto_layout_and_mesh_shape
    params["use_tpu"] = True if not args.tpu is None else False
    params["gpu_ids"] = args.gpu_ids
    params["steps_per_checkpoint"] = args.steps_per_checkpoint
    # Expand attention types param
    params["attention_types"] = expand_attention_types_params(params["attention_types"])
    assert len(params["attention_types"]) == params["n_layer"]  # Assert that the length of expanded list = num layers
    params["predict_batch_size"] = params.get("predict_batch_size", 1)  # Default to 1
    params["predict"] = args.predict
    params['model'] = params.get("model", "GPT") # Default model selection to GPT since it's the only option for now
    params["export"] = args.export
    # Set sampling parameters
    params["sampling_use_entmax"] = args.entmax_sampling

    # Sample quality of MoE models suffers when using the faster sampling method, so default to slow_sampling if
    # moe layers are present
    params["slow_sampling"] = True if params["moe_layers"] is not None else False

    logger.info(f"params = {params}")

    # Get eval tasks from params
    eval_tasks = params.get("eval_tasks", [])
    has_predict_or_eval_steps_or_eval_tasks = params["predict_steps"] > 0 or params["eval_steps"] > 0 or len(
        eval_tasks) > 0

    for t in eval_tasks:
        assert t in task_descriptors, f"Eval task '{t}' is not known"
        task_descriptors[t]["init_fn"](params)

    # Set up TPUs and Estimator
    if args.tpu == "colab":
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() if params["use_tpu"] else None
    else:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu) if params["use_tpu"] else None

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params["model_path"],
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=params["iterations"],
        save_summary_steps=params["iterations"],
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=params["iterations"],
            num_cores_per_replica=1,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

    estimator = tpu_estimator.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=model_fn,
        config=config,
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["train_batch_size"],
        predict_batch_size=params["predict_batch_size"],
        params=params)

    def _make_task_estimator(task):
        task_params = params.copy()
        task_params["eval_task"] = task
        return tpu_estimator.TPUEstimator(
            use_tpu=params["use_tpu"],
            model_fn=model_fn,
            config=config,
            train_batch_size=params["train_batch_size"],
            eval_batch_size=params["eval_batch_size"],
            predict_batch_size=params["predict_batch_size"],
            params=task_params)

    eval_task_estimators = {
        task: _make_task_estimator(task)
        for task in eval_tasks
    }

    if args.export:
        export_model(estimator, "export", params)
        return

    if args.predict:
        # Predict
        predictions = estimator.predict(input_fn=pred_input_fn)
        logger.info("Predictions generated")
        enc = fetch_encoder(params)
        handle_pred_output_fn(predictions, logger, enc, params, out_name=f"predictions_{args.sacred_id}_{current_step}")
        return

    def save_eval_results(task, eval_results):
        def as_python(x):
            if isinstance(x, numpy.generic):
                return x.item()
            return x
        eval_results = {k: as_python(v) for k, v in eval_results.items()}
        with open(f'eval_{args.sacred_id}.jsonl', 'a') as fh:
            json.dump({'task': task, 'current_step': current_step, **eval_results}, fh)
            fh.write('\n')

    def run_eval():
        logger.info("Running evaluation...")
        eval_results = estimator.evaluate(
                input_fn=partial(input_fn, eval=True),
                steps=params["eval_steps"])
        logger.info(f"Eval results: {eval_results}")
        save_eval_results('validation', eval_results)

    def run_eval_tasks():
        for task in eval_tasks:
            logger.info(f"Starting evaluation task '{task}'")
            task_info = task_descriptors[task]["get_task_info_fn"](params)
            task_estimator = eval_task_estimators[task]
            task_input_fn = task_descriptors[task]["input_fn"]
            eval_results = task_estimator.evaluate(
                input_fn=task_input_fn,
                steps=task_info["n_steps"],
                name=task)
            logger.info(f"Eval task '{task}' results: {eval_results}")
            save_eval_results(task, eval_results)
    
    if args.eval:
        run_eval_tasks()
        if params["eval_steps"] > 0:
            run_eval()
        return


    elif has_predict_or_eval_steps_or_eval_tasks:
        # Eval and train - stop and predict and/or eval every checkpoint
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])

            estimator.train(input_fn=partial(input_fn, global_step=current_step, eval=False), max_steps=next_checkpoint)
            current_step = next_checkpoint

            if params["predict_steps"] > 0:
                logger.info("Running prediction...")
                predictions = estimator.predict(input_fn=pred_input_fn)
                enc = fetch_encoder(params)
                handle_pred_output_fn(predictions, logger, enc, params, out_name=f"predictions_{args.sacred_id}_{current_step}")

            if params["eval_steps"] > 0:
                run_eval()

            if eval_tasks:
                run_eval_tasks()
                
        return
    else:
        # Else, just train
        while current_step < params["train_steps"]:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(input_fn, global_step=current_step, eval=False), max_steps=params["train_steps"])


if __name__ == "__main__":
    tf.disable_v2_behavior()
    args = parse_args()
    main(args)
