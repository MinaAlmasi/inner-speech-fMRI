'''
Perform sanity check using the first level models
'''

import pathlib
import pickle
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.glm import threshold_stats_img
import pandas as pd

# custom packages
from utils import load_all_flms
from first_level import get_paths, get_events

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

def get_button_press_per_run(bids_path, subjects_list,):
    subject_counts = {}

    for subject in subjects_list: 
        # get paths
        fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, subject, n_runs=6)

        # get events df 
        events = get_events(event_paths, drop_RT=False)

        # get number of button presses
        BP_per_run = {}

        for i, event_df in enumerate(events): # solution from https://www.tutorialspoint.com/how-to-count-occurrences-of-specific-value-in-pandas-column#:~:text=Using%20value_counts()%20method,unique%20value%20in%20the%20column.
            # start by filtering out events where the RT is 0 
            event_df = event_df.loc[event_df['RT'] > 0] 
            
            # get counts of button img (now only in trials where RT is more than 0, i.e. a button press is recorded)
            counts = event_df["trial_type"].value_counts()["button_img"]

            # add to BP per run
            BP_per_run[i+1] = counts
        
        subject_counts[subject] = BP_per_run

    counts_df = pd.DataFrame.from_dict(subject_counts)

    return counts_df

def plot_button_press_counts(counts_df, highlight_subjects, save_path=None):
    '''
    Plotting button press counts from a counts df using pandas 
    '''
    # create subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))
    axes = axes.flatten()

    # set the plots
    for i, column in enumerate(counts_df.columns):
        ax = axes[i]
        # Color condition
        color = ['#808080' if col in highlight_subjects else '#D3D3D3' for col in counts_df.columns]
        counts_df[column].plot(kind='bar', ax=ax, color=color[i])
        ax.set_title(column)
        ax.set_ylabel('Counts')
        ax.legend().set_visible(False)

    plt.tight_layout()

    # save fig
    if save_path:
        print("Saving plot ..")
        plt.savefig(save_path)

def test(): 
    # define root dir 
    path = pathlib.Path(__file__)
    bids_path = path.parents[1] / "data" / "InSpePosNegData" / "BIDS_2023E"

    save_path = path.parents[1] / "data"
    save_path.mkdir(parents=True, exist_ok=True)
    
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    counts = get_button_press_per_run(bids_path, subjects)

    plot_button_press_counts(counts, highlight_subjects = ["0119"], save_path = "button_press_sanity_check.png")

def main():
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    
    results_path = path.parents[1] / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # plot contrasts
    flms = load_all_flms(data_path / "all_flms")
    plot_all_subjects_contrasts(flms, save_path = results_path / "contrast_sanity_check.png")

    # plot button press
    bids_path = path.parents[1] / "data" / "InSpePosNegData" / "BIDS_2023E"
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    
    counts = get_button_press_per_run(bids_path, subjects)
    plot_button_press_counts(counts, highlight_subjects = ["0119"], save_path = "button_press_sanity_check.png")

if __name__ == "__main__":
    main()






