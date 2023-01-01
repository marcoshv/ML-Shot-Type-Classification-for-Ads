import tensorflow as tf
def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    # Parse config and create layers
    # You can use as a guide on how to pass config parameters to keras
    # looking at the code in `scripts/train.py`
    # TODO
    # Append the data augmentation layers on this list
    data_augmentation = tf.keras.Sequential()
    if data_aug_layer is not None:
        if "random_flip" in data_aug_layer:
            random_flip = tf.keras.layers.RandomFlip(**data_aug_layer['random_flip'])
            data_augmentation.add(random_flip)
        if "random_rotation" in data_aug_layer:
            random_rotation = tf.keras.layers.RandomRotation(**data_aug_layer["random_rotation"])
            data_augmentation.add(random_rotation)
        if "random_zoom" in data_aug_layer:
            random_zoom = tf.keras.layers.RandomZoom(**data_aug_layer["random_zoom"])
            data_augmentation.add(random_zoom)
    
    # Return a keras.Sequential model having the the new layers created
    # Assign to `data_augmentation` variable
    # TODO
    
    return data_augmentation
