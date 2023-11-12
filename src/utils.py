import pathlib
import pickle

def remove_flms(flms_dict, subject_ids=[]):
    '''
    Removes specified subjects (e.g. due to poor data) from the dictionary of first level models.  
      
    Args
        flms_dict: dictionary of all first level models (output of load_all_flms)
        subject_ids: list of subject ids to remove

    Returns
        flms_dict: dictionary of all first level models without the removed subjects
      '''
    
    for subject_id in subject_ids:
        if subject_id in flms_dict:
            del flms_dict[subject_id]
    
    return flms_dict

def load_all_flms(flm_path:pathlib.Path, exclude_subjects:list=[]): 
    '''
    Load the first level models from a specified path. Option to exclude subjects based on their ID.

    Args
        flm_path: path to the first level models
        exclude_subjects: list of subject ids to exclude

    Returns
        fl_models: list of first level models (objects)
    '''

    # obtain all file paths (except for the excluded )
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
    
    # remove the subjects should be excluded
    all_flms = remove_flms(all_flms, subject_ids=exclude_subjects)

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