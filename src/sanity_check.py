'''
Perform sanity check using the first level models
'''

import pathlib
import pickle
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.glm import threshold_stats_img

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


def load_masks(mask_path):
    pass


def plot_contrasts(flm, ax, contrast = "button_press"):
    # compute the contrast
    contrast = flm.compute_contrast(contrast, output_type = "z_score")

    # make bonferroni correction
    contrast, threshold = threshold_stats_img(
            contrast, 
            alpha=0.05, 
            height_control='bonferroni')
    
    plotting.plot_glass_brain(
        contrast,
        colorbar=True,
        plot_abs=False, 
        cmap='RdBu',
        axes = ax)

def main():
    path = pathlib.Path(__file__)
    flm_path = path.parents[1] / "data" / "flm_models"
    
    flms = load_all_flms(flm_path)

    plot_contrasts(flms[0])


if __name__ == "__main__":
    main()






