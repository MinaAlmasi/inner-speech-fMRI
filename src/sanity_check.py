'''
Perform sanity check using the first level models
'''

import pathlib
import pickle
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.glm import threshold_stats_img
from utils import load_all_flms, load_masks


def plot_contrasts(subject, flm, ax, contrast = "button_press"):
    '''
    Plot contrast for a given subject against "baseline" (see notebook 13)

    Args
        subject: subject id
        flm: first level model
        ax: axis to plot on
        contrast: contrast to plot
    '''

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
    
    ax.set_title(f"Subject: {subject}")

def plot_all_subjects_contrasts(flms, save_path):
    
    # set the canvas
    fig, axes = plt.subplots(4,2,figsize=(10, 12))

    # iterate over each subject in dictionary
    for i, subject_id in enumerate(flms):
        flm = flms[subject_id]
        
        # flatten axes
        ax = axes.flatten()[i]

        # plot contrasts
        plot_contrasts(subject = subject_id, flm = flm, ax = ax, contrast = "button_press")

    # save fig 
    if save_path: 
        plt.savefig(save_path)


def main():
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    
    results_path = path.parents[1] / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    
    flms = load_all_flms(data_path / "all_flms")

    masks = load_masks(data_path / "mask_objects")

    plot_all_subjects_contrasts(flms, save_path = results_path / "sanity_check.png")


if __name__ == "__main__":
    main()






