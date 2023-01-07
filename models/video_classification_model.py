from tensorflow.keras.applications.resnet import ResNet50
import tensorflow as tf
from tensorflow.keras import regularizers

def get_rnn(rnn_type,rnn_units,name):
    RNN_TYPE = {
        "gru": tf.keras.layers.GRU(rnn_units, return_sequences=False, name= name),
        "lstm": tf.keras.layers.LSTM(rnn_units, return_sequences=False, name= name)
    }
    return RNN_TYPE[rnn_type]


def create_model(
    weights:str = "imagenet",
    input_shape:tuple = (16,224,224,3),
    X3_dropout_rate:float = 0.0,
    rnn_type:'int' = "gru",
    rnn_units:int = 128,
    X4_dropout_rate:float = 0.0,
    scale_classes:int = None,
    move_classes:int = None
):
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (frames,height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size can not be smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    X3_dropout_rate : float
        Value used for Dropout layer affecting teh CNN to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    rnn_type: int 
        String, "gru" for a GRU CNN or "lstm" for a LSTM one

    rnn_units: int
        Quantity of neurons on the RNN

    X4_dropout_rate: float
        Value used for Dropout layer affecting teh RNN to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    scale_classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    move_classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """

    # Create the model to be used for finetuning here!
    if weights == "imagenet":
        input_l = tf.keras.layers.Input(shape=input_shape,dtype='float32', name='input_l')

        mask_input = tf.keras.layers.Input((input_shape[0],), dtype="bool", name='mask_input')

        X1_preprocess = tf.keras.applications.resnet50.preprocess_input(input_l) # Change pixels interval from [0, 255] to [0, 1]

        # Create the corresponding core CNN model
        model_cnn = ResNet50(
                weights = 'imagenet',  # Load weights pre-trained on ImageNet.
                input_shape = input_shape[1:],
                include_top=False,
                pooling='avg',
                )

        # freeze the base model and create a new model.
        model_cnn.trainable = False #Prevent coeficients from changing in first iteration.
        # This wrapper allows to apply a layer to every temporal slice of an input,  tf.keras.layers.TimeDistributed()() 
        X2_cnn_tdist = tf.keras.layers.TimeDistributed(model_cnn, name='X2_cnn_tdist')(X1_preprocess,mask=mask_input) 

        # SCALE
        # Adds a first dropout layer to the TimeDistributed() layer
        X3_drop_scale = tf.keras.layers.Dropout(X3_dropout_rate, name='X3_drop_scale')(X2_cnn_tdist)

        #Adds recurrent layer
        X4_model_rnn_scale = get_rnn(rnn_type,rnn_units,'X4_model_rnn_scale')(X3_drop_scale)

        # Adds a second dropout layer for regularization,
        X5_drop2_scale = tf.keras.layers.Dropout(X4_dropout_rate, name='X5_drop2_scale')(X4_model_rnn_scale)


        # MOVEMENT
        # Adds a first dropout layer to the TimeDistributed() layer
        X3_drop_move = tf.keras.layers.Dropout(X3_dropout_rate, name='X3_drop_move')(X2_cnn_tdist)
        #Adds recurrent layer
        X4_model_rnn_move = get_rnn(rnn_type,rnn_units,'X4_model_rnn_move')(X3_drop_move)
        # Adds a second dropout layer for regularization.
        X5_drop2_move = tf.keras.layers.Dropout(X4_dropout_rate, name='X5_drop2_move')(X4_model_rnn_move)


        # Add the classification layer here, use keras.layers.Dense() and `classes` parameter for scale and movement.
        # Assign it to `outputs` variable
        #outputs = tf.keras.layers.Dense(classes, kernel_regularizer='l2',activation='softmax')(x) 
        outputs_scale = tf.keras.layers.Dense(scale_classes, 
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5), 
                activation='softmax',
                name='outputs_scale')(X5_drop2_scale)

        outputs_move = tf.keras.layers.Dense(move_classes, 
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5), 
                activation='softmax',
                name='outputs_move')(X5_drop2_move)

        model = tf.keras.Model([input_l, mask_input], [outputs_scale,outputs_move])
    
    else: 
        #load our already defined and finetuned model
        # Assign it to `model` variable
        model = tf.keras.load_model(weights)
        model.trainable = True

    return model
