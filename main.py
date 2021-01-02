import argparse

import tensorflow as tf

from src.main import main

if __name__ == "__main__":
    tf.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.")
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=["device:GPU:0"],
                        help=" If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
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
    parser.add_argument("--prompt", type=str, help="path to .txt file containing a prompt for prediction. If empty, "
                                                   "defaults to unicorns.",
                        default="")
    parser.add_argument("--check_dataset", action="store_true",
                        help="If set, outputs sample from the dataset and quits.")
    args = parser.parse_args()
    main(args)
