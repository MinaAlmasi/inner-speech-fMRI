import pathlib

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

    # initialize list for all models
    all_flms = []

    # iterate over file names
    for file in flm_files:

        # load flm
        flm = pickle.load(open(file, 'rb'))

        # append to list
        all_flms.append(flm)
    
    return all_flms


def load_masks(masks_path):
    '''
    Load the masks from a specified path.

    Args
        masks_path: path to the masks

    Returns
        masks: list of masks (objects)
    '''

    # obtain all file paths
    mask_files = [file for file in masks_path.iterdir() if file.name.endswith(".pkl")]

    # initialize list for all masks
    masks = {}

    # iterate over file names
    for file in mask_files:

        # get subject id from name 
        subject_id = file.name.split("_")[1]

        # load mask
        mask = nib.load(file)

        # add to to dict
        masks[subject_id] = mask
    
    return masks