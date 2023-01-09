import cv2
import numpy as np

def load_video(path,max_frames, resize):
    frames = []
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_list = np.linspace(0,frame_count,int(min(frame_count,max_frames)),dtype=int) # int(min(frame_count,max_frames)) work as step
    for fn in frame_list:
        cv2.CAP_PROP_POS_FRAMES = fn
        success, frame = cap.read()
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if not success:
            break
    cap.release()
    
    mask = np.zeros((max_frames,), dtype=bool)
    mask[:len(frames)] = 1
    
    return np.concatenate((np.array(frames), np.zeros((max_frames-len(frames), *resize, 3)))), mask
"""--------------------------------------------------------------------------------------------------"""
import numpy as np
from tensorflow import keras
#from load_video import load_video

class VideoDataGenerator(keras.utils.Sequence):
    """
    A custom video data generator to read videos and create batches of samples
    for the Keras' model.fit_generator interface.
    Attributes
    ----------
    df : pandas.DataFrame
        a dataframe with video pathname and corresponding label
    path : str
        root folder of all the video
    batch_size : int, optional
        number of samples in each batch
    shuffle : bool, optional
        whether to shuffle the data at the end of each epoch
    file_col : str, optional
        column name of the video pathname column
    y_col : str, optional
        column name of the label column
    mapping : dict, optional
        custom mapping for the label
    max_frames : int, optional
        max number of captured frames in each video
    resize : int, optional
        resolution of converted video
    step : int, optional
        skip how many frames before capturing 1 frame
    """

    def __init__(self, df, file_col=None, batch_size=4, shuffle=True, y_col_scale=None, y_col_movement= None,  
                mapping_scale=None, mapping_movement=None, max_frames=120, img_size=224):
        self.df = df.copy()
        self.file_col = file_col
        self.indices = self.df.index.to_list()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.y_col_scale = y_col_scale
        self.y_col_movement = y_col_movement
        self.mapping_scale = mapping_scale
        self.mapping_movement = mapping_movement
        self.max_frames = max_frames
        self.resize = (img_size, img_size)

    def __len__(self):
        """Return the number of batches in the data"""
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        """Return a batch of data"""
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__generate_data(indices)

    def on_epoch_end(self):
        """this is called by Keras at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __generate_data(self, indices):
        """helper method for generating a batch of data"""
        X, mask = [], []
        for i in indices:
            v, m = load_video(self.df.loc[i, self.file_col], self.max_frames, self.resize)
            X.append(v)
            mask.append(m)
        y_scale = self.df.loc[indices, self.y_col_scale]
        y_movement = self.df.loc[indices, self.y_col_movement]
        if self.mapping_scale:
            scale = y_scale.map(self.mapping_scale)
        if self.mapping_movement:
            movement = y_movement.map(self.mapping_movement)

        return (np.array(X), np.array(mask)), (np.array(scale), np.array(movement))
"""--------------------------------------------------------------------------------------"""

import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#from video_data_generator import VideoDataGenerator

# Define hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 5
MAX_SEQ_LENGTH = 16
NUM_FEATURES = 128
FILE_COL = 'scene_fullpath'
NUM_CLASSES_SCALE = 5
NUM_CLASSES_MOVEMENT = 5
Y_COL_SCALE = 'scene_scale_label'
Y_COL_MOVEMENT = 'scene_movement_label'
CLASS_MAPPING_SCALE = {'CS':1, 'LS':4, 'ECS':0, 'MS':2 ,'FS':3}
CLASS_MAPPING_MOVEMENT =  {'Static':4,'Motion':0,'Multi_movement':3,'Push':1,'Pull':2}

# Data preparation
train_df = pd.read_csv('data_movies/csv_files/train.csv')
val_df = pd.read_csv('data_movies/csv_files/val.csv')
test_df = pd.read_csv('data_movies/csv_files/test.csv')

print(f"Total number of videos for training: {len(train_df)}")
print(f"Total number of videos for validation: {len(val_df)}")
print(f"Total number of videos for testing: {len(test_df)}")

train_data_gen = VideoDataGenerator(
    train_df,
    FILE_COL,
    BATCH_SIZE,
    y_col_scale= Y_COL_SCALE,
    y_col_movement= Y_COL_MOVEMENT,
    mapping_scale=CLASS_MAPPING_SCALE,
    mapping_movement=CLASS_MAPPING_MOVEMENT,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
)

val_data_gen = VideoDataGenerator(
    val_df,
    FILE_COL,
    BATCH_SIZE,
    y_col_scale= Y_COL_SCALE,
    y_col_movement= Y_COL_MOVEMENT,
    mapping_scale=CLASS_MAPPING_SCALE,
    mapping_movement=CLASS_MAPPING_MOVEMENT,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
)

test_data_gen = VideoDataGenerator(
    test_df,
    FILE_COL,
    BATCH_SIZE,
    y_col_scale= Y_COL_SCALE,
    y_col_movement= Y_COL_MOVEMENT,
    mapping_scale=CLASS_MAPPING_SCALE,
    mapping_movement=CLASS_MAPPING_MOVEMENT,
    max_frames=MAX_SEQ_LENGTH,
    img_size=IMG_SIZE,
)

"""--------------------------------------------------------------------------------------"""
# building the model
def build_model():
    cnn = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn.trainable = False
    input_layer = keras.Input((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')
    model = keras.layers.TimeDistributed(cnn)(input_layer, mask=mask_input)
    model = keras.layers.GRU(NUM_FEATURES)(model)
    model = keras.layers.Dropout(0.2)(model)
    output_scale = keras.layers.Dense(NUM_CLASSES_SCALE, activation='softmax', name='output_scale')(model)
    output_movement = keras.layers.Dense(NUM_CLASSES_MOVEMENT, activation='softmax', name='output_movement')(model)

    model= keras.models.Model(inputs=[input_layer, mask_input], outputs=[output_scale, output_movement])
    optimizer = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                    loss={'output_scale': 'sparse_categorical_crossentropy', 'output_movement': 'sparse_categorical_crossentropy'},
                    metrics={'output_scale': 'accuracy', 'output_movement': 'accuracy'})
    return model

def run_experiment():
    checkpoint_path = 'experiments/checkpoints/'
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1, monitor='val_accuracy', mode='max'
    )
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    model = build_model()
    model.summary()
    history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=EPOCHS,
        callbacks=callbacks_list,
    )

    return history, model

if __name__ == '__main__':
    run_experiment()