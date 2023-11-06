import pathlib
import pickle

def load_all_flms(flm_path:pathlib.Path): 
    '''
    Load the first level models from a specified path.

    Args
        flm_path: path to the first level models

    Returns
        fl_models: list of first level models (objects)
    '''

    # obtain all file paths
    flm_files = [file for file in flm_path.iterdir() if file.name.endswith(".pkl")]

    # sort list of file paths
    flm_files.sort()

    # initialize list for all models
    all_flms = {}

    # iterate over file names
    for file in flm_files:
        # get subject id from name
        subject_id = file.name.split("_")[1][:4]

        # load flm
        flm = pickle.load(open(file, 'rb'))

        # append to list
        all_flms[subject_id] = flm
    
    return all_flms


def load_masks(masks_object_path:pathlib.Path):
    '''
    Load saved mask objects from a specified path.

    Args
        masks_object_path: path to the mask objects (pkl)

    Returns
        masks: list of masks (objects)
    '''

    # obtain all file paths
    mask_files = [file for file in masks_object_path.iterdir() if file.name.endswith(".pkl")]

    # sort 
    mask_files.sort()

    # initialize list for all masks
    masks = {}

    # iterate over file names
    for file in mask_files:

        # get subject id from name 
        subject_id = file.name.split("_")[1][:4]

        # load mask
        mask = pickle.load(open(file, 'rb'))

        # add to to dict
        masks[subject_id] = mask
    
    return masks