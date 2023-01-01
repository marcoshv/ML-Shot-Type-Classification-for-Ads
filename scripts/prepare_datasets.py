import argparse
import pandas as pd
import json
import os
import cv2

def parse_args():
    """
    Use argparse to get the input parameters for preparing the dataset.
    """
    parser = argparse.ArgumentParser(description="Create dataset.csv files.")
    parser.add_argument("path_json", type=str,help="Full path to json file.",)
    parser.add_argument("path_videos", type=str,help="Full path to video.",)

    args = parser.parse_args()

    return args


def json_to_dataframe(folder_path, split_label= None, scale='scale', movement='movement', label='label', value='value'):
    """
    Takes a given path, and a split label (optional), loads a Json file and return a dictionary, splitted if using the split label. Then
    takes the given dictonary, and features (scale, movement, label, value), and returns a dataframe ready for EDA.

    Parameters
    ----------
    path : str --> Path to the file directory
    split_label : str --> Split label in the Json file
    scale : str --> Scale of the scene
    movement : str --> Type of movement in the scene
    label : str --> Scale`s and movement`s label of the scene
    value : str --> Scale`s and movement`s value of the scene
    Returns
    -------
    dataframe: pd.Dataframe --> Pandas Dataframe."""
    with open(folder_path) as json_file:
        dictionary = json.load(json_file)
    if split_label is not None:
        dict = dictionary[split_label]
    else:
        dict = dictionary

    movie_id, scene_id, scale_label, scale_value, movement_label, movement_value, data_split_type = [], [], [], [], [], [], []
    for movie in dict.items():
        for movie_scene in movie[1].items():
            movie_id.append(movie[0])
            scene_id.append(movie_scene[0])
            scale_label.append(movie_scene[1][scale][label])
            scale_value.append(movie_scene[1][scale][value])
            movement_label.append(movie_scene[1][movement][label])
            movement_value.append(movie_scene[1][movement][value])
            if split_label is not None:
                data_split_type.append(split_label) 
    if split_label is not None:  
        dataframe = pd.DataFrame(list(zip(movie_id, scene_id, scale_label, scale_value, movement_label, movement_value, data_split_type)),
                                        columns=['movie_id', 'movie_scene_id', 'scene_scale_label', 'scene_scale_value',
                                         'scene_movement_label', 'scene_movement_value', 'data_split_type'])
    else:
        dataframe = pd.DataFrame(list(zip(movie_id, scene_id, scale_label, scale_value, movement_label, movement_value)),
                                        columns=['movie_id', 'movie_scene_id', 'scene_scale_label', 'scene_scale_value',
                                         'scene_movement_label', 'scene_movement_value', ])
    return dataframe



def get_video_parameters(folder_path):
    """
    Takes a given path filled with videos, and loops through the folder, captures the folder`s full path, movie`s id, scene`s id, scene`s frames, scene`s width,
    scene`s height, scene`s fps and scene`s duration, append those values to different lists and creates a dataframe.
    

    Parameters
    ----------
    folder_path : str --> Path to the file directory
    Returns
    -------
    dataframe: pd.Dataframe --> Pandas Dataframe."""
    scene_frames, scene_width, scene_height, scene_fps, scene_duration, scene_fullpath, movie_id_list,  scene_id = [], [], [], [], [], [], [], []
    for dirpath, _, files in os.walk(folder_path):
        for filename in files:
            fullpath= os.path.join(dirpath + '/',filename)
            pre_label = os.path.dirname(fullpath)
            movie_id = os.path.basename(pre_label)
            scene_capture = cv2.VideoCapture(fullpath)
            if scene_capture.isOpened() == True:
                scene_fullpath.append(fullpath)
                movie_id_list.append(movie_id)
                scene_id.append(filename.split('.')[0].split('_')[1])
                scene_frames.append(scene_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                scene_width.append(scene_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                scene_height.append(scene_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                scene_fps.append(round(scene_capture.get(cv2.CAP_PROP_FPS), 2))
                scene_duration.append(round(scene_capture.get(cv2.CAP_PROP_FRAME_COUNT)/scene_capture.get(cv2.CAP_PROP_FPS), 2))
            else:
                print(f'Error while reading {folder_path}')
                scene_frames.append('NA')
                scene_width.append('NA')
                scene_height.append('NA')
                scene_fps.append('NA')
                scene_duration.append('NA')
    dataframe_video_parameters = pd.DataFrame(list(zip(movie_id_list, scene_id, scene_frames, scene_width, scene_height, 
                                                scene_fps, scene_duration, scene_fullpath)),
                                                columns=['movie_id',  'movie_scene_id','scene_frames', 'scene_width', 'scene_height',
                                                 'scene_fps', 'scene_duration', 'scene_fullpath'])
    
    return dataframe_video_parameters

def create_dataframes(path_json, path_videos, train='train', test='test', validation='val'):
    """
    Takes a given path filled with videos, and splitting labels, then calls 'json_to_dataframe' and 'get_video_parameters' function,
    to create train_df, test_df, val_df, video_parameter_df and a complete_dataframe.

    Parameters
    ----------
    path_json : str --> Path to the json file
    path_json : str --> Path to folder containing videos
    Returns
    -------
    dataframe: pd.Dataframe --> Pandas Dataframe."""

    pre_train_df = json_to_dataframe(path_json, train)
    pre_test_df = json_to_dataframe(path_json, test)
    pre_val_df = json_to_dataframe(path_json, validation)
    pre_train_test_val_df = pd.concat([pre_train_df , pre_test_df , pre_val_df]).reset_index(drop=True)

    video_parameter_df = get_video_parameters(path_videos)

    complete_dataset_df = pd.merge(pre_train_test_val_df, video_parameter_df, on=['movie_id', 'movie_scene_id'])
    train_df = complete_dataset_df[complete_dataset_df['data_split_type'] == 'train']
    test_df = complete_dataset_df[complete_dataset_df['data_split_type'] == 'test']
    val_df = complete_dataset_df[complete_dataset_df['data_split_type'] == 'val']

    complete_dataset_df.to_csv('datasetv1_movie_shot_trailers.csv', index=False)
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    val_df.to_csv('val.csv', index=False)

if __name__ == "__main__":
    args = parse_args()
    create_dataframes(args.path_json,args.path_videos)