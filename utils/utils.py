import os
import yaml
import numpy as np



def validate_config(config):
    """
    Takes as input the experiment configuration as a dict and checks for
    minimum acceptance requirements.

    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    """
    if "data" not in config:
        raise ValueError("Missing experiment data")

    if "train_csv_directory" not in config["data"]:
        raise ValueError("Missing experiment training data")

    if "val_csv_directory" not in config["data"]:
        raise ValueError("Missing experiment validation data")
    
    if "test_csv_directory" not in config["data"]:
        raise ValueError("Missing experiment testing data")


def load_config(config_file_path):
    """
    Loads experiment settings from a YAML file into a Python dict.
    See: https://pyyaml.org/.

    Parameters
    ----------
    config_file_path : str
        Full path to experiment configuration file.
        E.g: `/home/app/src/experiments/exp_001/config.yml`

    Returns
    -------
    config : dict
        Experiment settings as a Python dict.
    """
    # TODO
    # Load config here and assign to `config` variable
    with open(config_file_path, 'rt', encoding='utf8') as f:
            config = yaml.safe_load(f)
    # Don't remove this as will help you doing some basic checks on config
    # content
    validate_config(config)
    return config


def get_class_number(config, df):
    """
    Takes as input the experiment configuration as a dict and a dataframe and get the number of classes

    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    
    df : dataframe
        Dataframe containing the data columns with the scale and movement classes
    """
    scale_class_num = len(df[config['data']['y_col_scale']].unique())
    movement_class_num = len(df[config['data']['y_col_movement']].unique())

    return scale_class_num, movement_class_num


def predict_from_folder(model, test_data_gen, scale_names,move_names):
    """
    Gets predictions based on a data_generator based on a model
 
    Parameters
    ----------
    model : str
        Full path to .h5 model file.
        E.g: `/home/app/src/experiments/exp_003/model.09-0.5953.h5`

    Returns
    -------
    prediction : array
        Returns most likely scale and movement label, as well as their likelyhood
    """
    scale_preds = np.array([])
    move_preds = np.array([])
    scale_tests = np.array([])
    move_tests = np.array([])

    for ims, batch_labels in test_data_gen:
        
        scale_pred, move_pred = model.predict(ims)
        scale_preds = np.concatenate([scale_preds, np.argmax(scale_pred,axis = -1)]).astype('int')
        move_preds = np.concatenate([move_preds, np.argmax(move_pred,axis = -1)]).astype('int')
        
        scale_test, move_test = batch_labels
        scale_tests = np.append(scale_tests,np.argmax(scale_test, axis=-1)).astype('int')
        move_tests = np.append(move_tests,np.argmax(move_test, axis=-1)).astype('int')

    scale_preds = [scale_names[id] for id in scale_preds]
    scale_tests = [scale_names[id] for id in scale_tests]
    move_preds = [move_names[id] for id in move_preds]
    move_tests = [move_names[id] for id in move_tests]

    return scale_preds,scale_tests, move_preds, move_tests
