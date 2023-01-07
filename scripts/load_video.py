import cv2
import numpy as np

def load_video(path,max_frames, resize):
    """
    Load a video and return the numpy array along with the mask.
    Capture 1 frame every `step` steps.
    If the number of captured frames is less than `max_frames`, pad the numpy array with zeros.
    It the number of captrued frames is more than `max_frames`, ignore the rest.
    Parameters
    ----------
    path : str
        Path of the video
    max_frames : int, optional
        Maximum number of captured frames
    resize : int, optional
        Resolution of the returned video. Ex (224,224)
    Returns
    -------
    tuple
        a tuple of the converted video and the mask
    """
    frames = []
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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