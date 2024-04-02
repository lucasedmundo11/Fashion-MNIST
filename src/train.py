import os
import random
import logging
import argparse

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from model.fmnist import input_fn

from utils import Params
from utils import set_logger
from utils import save_dict_to_json
from model.cnn import model_fn
from model.trainer import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Exp dir containing params.json")
parser.add_argument('--data_dir', default='data/fashion_mnist',
                    help="Dir containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "test")

    (train_images, train_labels), (eval_images, eval_labels) = fashion_mnist.load_data()

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_images)
    params.eval_size = len(eval_images)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_images, train_labels, params)
    eval_inputs = input_fn(False, eval_images, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec,
                       args.model_dir, params, args.restore_from)