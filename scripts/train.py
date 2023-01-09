"""
This script will be used for training our model. The only input argument it
should receive is the path to our configuration file in which we define all
the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.
"""
import argparse
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from models import video_classification_model
from utils import utils
from utils import data_aug
from scripts.video_data_generator import VideoDataGenerator

# Prevent tensorflow to allocate the entire GPU
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Supported optimizer algorithms
OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
}


# Supported callbacks
CALLBACKS = {
    "model_checkpoint": keras.callbacks.ModelCheckpoint,
    "tensor_board": keras.callbacks.TensorBoard,
    "reduce_lr" : keras.callbacks.ReduceLROnPlateau,
    "early_stopping": keras.callbacks.EarlyStopping
}


def parse_args():
    """
    Use argparse to get the input parameters for training the model.
    """
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Full path to experiment configuration file.",
    )

    args = parser.parse_args()

    return args


def parse_optimizer(config):
    """
    Get experiment settings for optimizer algorithm.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    opt_name, opt_params = list(config["compile"]["optimizer"].items())[0]
    optimizer = OPTIMIZERS[opt_name](**opt_params)

    del config["compile"]["optimizer"]

    return optimizer


def parse_callbacks(config):
    """
    Add Keras callbacks based on experiment settings.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    callbacks = []
    if "callbacks" in config["fit"]:
        for callbk_name, callbk_params in config["fit"]["callbacks"].items():
            callbacks.append(CALLBACKS[callbk_name](**callbk_params))

        del config["fit"]["callbacks"]

    return callbacks

def freeze_layers(config,model):
    """
    Freezes layers accourding to setting file

    Parameters
    ----------
    config : str
        Experiment settings.
    """

    if 'trainable' in config:
        for layer_name, freeze in config['trainable'].items():
             model.get_layer(name=layer_name).trainable = freeze
    return model


def main(config_file):
    """
    Code for the training logic.

    Parameters
    ----------
    config_file : str
        Full path to experiment configuration file.
    """
    # Load configuration file, use utils.load_config()
    config = utils.load_config(config_file)

    # Build data augmentator, use data_aug.data_augmentator()
    augmentation_seq = data_aug.data_augmentator(config)

    # Load training  and validation dataset
    train_df = pd.read_csv(config['data']['train_csv_directory'])
    val_df = pd.read_csv(config['data']['val_csv_directory'])

    # 
    train_data_gen = VideoDataGenerator(
        df = train_df,
        file_col= config['data']['file_col'],
        batch_size = config['data']['batch_size'],
        shuffle = True,
        y_col_scale= config['data']['y_col_scale'],
        y_col_movement= config['data']['y_col_movement'],
        mapping_scale= config['data']['mapping_scale'],
        mapping_movement= config['data']['mapping_movement'],
        max_frames= config['data']['max_frames'],
        img_size= config['data']['image_size'],
        augmentation_seq = augmentation_seq
    )

    val_data_gen = VideoDataGenerator(
        df = val_df,
        file_col= config['data']['file_col'],
        batch_size = config['data']['batch_size'],
        shuffle = True,
        y_col_scale= config['data']['y_col_scale'],
        y_col_movement= config['data']['y_col_movement'],
        mapping_scale= config['data']['mapping_scale'],
        mapping_movement= config['data']['mapping_movement'],
        max_frames= config['data']['max_frames'],
        img_size= config['data']['image_size'],
        augmentation_seq = augmentation_seq
    )

    # get number of classes and check if they are equal to the number of classes declared in the model configuration, use utils.get_class_number()
    scale_classes, move_classes = utils.get_class_number(config, train_df)
    if scale_classes != config['model']['scale_classes']:
        print(f'Scale clases quantity from config file dont match the database ({scale_classes})')
    
    if move_classes != config['model']['move_classes']:
        print(f'Scale clases quantity from config file dont match the database ({move_classes})')

    # Creates a Resnet50 model for finetuning
    model = video_classification_model.create_model(**config["model"])
    model = freeze_layers(config,model)
    print(model.summary())

    # Compile model, prepare for training
    optimizer = parse_optimizer(config)
    model.compile(
        optimizer=optimizer,
        **config["compile"],
    )

    # Start training!
    callbacks = parse_callbacks(config)
    model.fit(
        train_data_gen, validation_data=val_data_gen, callbacks=callbacks, **config["fit"]
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
