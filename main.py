"""GPT-like model in Mesh-Tensorflow"""

from functools import partial
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from utils import save_config, expand_attention_types_params, yes_or_no, remove_gs_or_filepath, setup_logging, check_dataset
from inputs import generic_text, pred_input, test_generic_text, test_pred_input, handle_pred_output, test_handle_pred_output, mlm_sample_text
from model_fns import model_fn
from encoders import fetch_encoder
from configs import fetch_model_params
from tasks import task_descriptors

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.") 
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0], help=" If training on GPU, can specify which GPU ids.")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=5000, help="Save a model checkpoint every X steps.")
    parser.add_argument("--auto_layout", action="store_true", help="If set, generates a MTF auto layout.")
    parser.add_argument("--auto_layout_and_mesh_shape", action="store_true", help="If set, generates a MTF auto layout and auto mesh shape.")
    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true", help="If set, uses the model to predict rather than train.")
    parser.add_argument("--check_dataset", action="store_true", help="If set, outputs sample from the dataset and quits.")
    args = parser.parse_args()
    
    # Rewire to use testing related functions if --test is set
    if args.test:
        args.model = "test"
    assert args.model is not None, "Model must be set"

    return args

def main(args):
    # Setup logging
    logger = setup_logging(args)

    # Read params of model
    params = fetch_model_params(args.model)

    # Fetch appropriate input functions
    input_fn = generic_text if not args.test else test_generic_text
    pred_input_fn = pred_input if not args.test else test_pred_input
    handle_pred_output_fn = handle_pred_output if not args.test else test_handle_pred_output

    if params["mlm_training"]:
        mlm_sample_text_fn = partial(mlm_sample_text, params)
        input_fn = partial(generic_text, sample_text_fn=mlm_sample_text_fn)

    # Fetch encoder per params
    encoder = fetch_encoder(params)

    pred_input_fn = partial(pred_input_fn, logger=logger, enc=encoder)

    # Sample from Dataset if check dataset flag is on
    if args.check_dataset:
        check_dataset(input_fn)
        
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
    params["predict_batch_size"] = params.get("predict_batch_size", 1) # Default to 1
    params["predict"] = args.predict

    # Sample quality of MoE models suffers when using the faster sampling method, so default to slow_sampling if
    # moe layers are present
    params["slow_sampling"] = True if params["moe_layers"] is not None else False

    logger.info(f"params = {params}")

    # Get eval tasks from params
    eval_tasks = params.get("eval_tasks", [])
    has_predict_or_eval_steps_or_eval_tasks = params["predict_steps"] > 0 or params["eval_steps"] > 0 or len(eval_tasks) > 0

    for t in eval_tasks:
        assert t in task_descriptors, f"Eval task '{t}' is not known"
        task_descriptors[t]["init_fn"](params)

    # Set up TPUs and Estimator
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
            eval_batch_size=params["train_batch_size"],
            predict_batch_size=params["predict_batch_size"],
            params=task_params)

    eval_task_estimators = {
        task: _make_task_estimator(task)
        for task in eval_tasks
    }

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info(f"Current step {current_step}")

    if args.predict:
        # Predict
        predictions = estimator.predict(input_fn=pred_input_fn)
        logger.info("Predictions generated")
        enc = fetch_encoder(params)
        handle_pred_output_fn(predictions, logger, enc, params, out_name=f"predictions_{current_step}")
        return

    elif has_predict_or_eval_steps_or_eval_tasks:
        # Eval and train - stop and predict and/or eval every checkpoint
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + args.steps_per_checkpoint,
                                  params["train_steps"])

            estimator.train(input_fn=partial(input_fn, eval=False, current_step=current_step), max_steps=next_checkpoint)
            current_step = next_checkpoint

            if params["predict_steps"] > 0:
                logger.info("Running prediction...")
                predictions = estimator.predict(input_fn=pred_input_fn)
                enc = fetch_encoder(params)
                handle_pred_output_fn(predictions, logger, enc, params, out_name=f"predictions_{current_step}")

            if params["eval_steps"] > 0:
                logger.info("Running evaluation...")
                eval_results = estimator.evaluate(
                    input_fn=partial(input_fn, eval=True),
                    steps=params["eval_steps"])
                logger.info(f"Eval results: {eval_results}")

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
        return
    else:
        # Else, just train
        while current_step < params["train_steps"]:
            estimator.train(input_fn=partial(input_fn, eval=False, current_step=current_step), max_steps=params["train_steps"])
        return

if __name__ == "__main__":
    tf.disable_v2_behavior()
    args = parse_args()
    main(args)
