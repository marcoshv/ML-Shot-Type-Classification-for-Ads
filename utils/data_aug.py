from vidaug import augmentors as va
def data_augmentator(config):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    Parameters
    ----------
    config : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    seq: arrays
        augmentated frames.
    """
    try:
        st_video = lambda aug: va.Sometimes(config['data_augmentation']['video_prob'],aug)
        st_flip = lambda aug: va.Sometimes(config['data_augmentation']['flip_prob'], aug) #float as probability
        st_pepp_salt = lambda aug: va.Sometimes(config['data_augmentation']['pep_sal_prob'], aug) 
        st_dist = lambda aug: va.Sometimes(config['data_augmentation']['distortion_prob'], aug)
        st_geo = lambda aug: va.Sometimes(config['data_augmentation']['geometric_prob'], aug)

        pepper_salt_seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            va.Salt(50),  #number of real pixels
            va.Pepper(50)
        ])
        distorsion_seq = va.OneOf([
            va.ElasticTransformation(2,.5,0,mode='wrap'), # horizontally flip the video with 100% probability
            va.PiecewiseAffineTransform(50,7,0.2)    #???
        ])
        geometric_seq = va.OneOf([
            va.RandomTranslate(25,25), #moves image x, y pixels
            va.RandomShear(.2,.2) #perspective
        ])

        seq = st_video(va.Sequential([
            st_flip(va.HorizontalFlip()),
            st_pepp_salt(pepper_salt_seq),
            st_dist(distorsion_seq),
            st_geo(geometric_seq)]
            ))
    except:
        seq=False
    return seq
