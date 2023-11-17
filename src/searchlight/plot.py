'''
Plot outcome of searchlight analysis from train.py
'''

# utils
import pathlib, pickle
import numpy as np

# custom functions
import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2] / "src"))
from first_level import get_paths
from utils import load_masks

# 
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_img, show

def plot_searchlight_outcome(fprep_f_paths, mask_paths, searchlight_scores, results_path):
    """
    Plot searchlight results
    """
    # create an image of the searchlight scores
    searchlight_img = new_img_like(fprep_f_paths[0], searchlight_scores)

    # plot the searchlight scores
    surface_plot = plotting.plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='Negative vs Button (Acc>0.6')
    deep_plot = plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='Positive vs Negative (Acc>0.6')

    # save the plots
    surface_plot.savefig(results_path / "searchlight_surface_plot.png")
    deep_plot.savefig(results_path / "searchlight_deep_plot.png")

def find_most_important_voxels(searchlight_scores, mask_img, n_voxels=500):
    # find the percentile that makes the cutoff for x best voxels
    perc=100*(1-n_voxels/searchlight_scores.size)
    
    # find the cutoff
    cut=np.percentile(searchlight_scores,perc)

    # make copy of mask
    process_mask = mask_img.get_fdata().astype(int)

    # set all voxels below the cutoff to 0
    process_mask[searchlight_scores<cut]=0

    process_mask_img = new_img_like(mask_img, process_mask)

    return process_mask_img, cut

def plot_most_important_voxels(anat_filename, searchlight_scores, results_path, n_voxels=500):
    # find the percentile that makes the cutoff for x best voxels
    perc=100*(1-n_voxels/searchlight_scores.size)
    
    # find the cutoff
    cut=np.percentile(searchlight_scores,perc)

    # create an image of the searchlight scores
    searchlight_img = new_img_like(anat_filename, searchlight_scores)

    plot_img(searchlight_img, bg_img=anat_filename,
         title="Searchlight", display_mode="z",cut_coords=[-25,-20,-15,-10,-5,0,5],
         vmin=.40, cmap='jet', threshold=cut, black_bg=True)
    
    # save plot
    plt.savefig(results_path / "searchlight_topvoxels.png")

def main():
    subject = "0116"

    # define paths 
    path = pathlib.Path(__file__)
    bids_path = path.parents[2] / "data" / "InSpePosNegData" / "BIDS_2023E"

    data_path = path.parents[2] / "data" / "searchlight"
    results_path = path.parents[2] / "results"

    anat_filename=pathlib.Path(f'/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')

    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, subject, n_runs=6)
    
    # load the searchlight file
    with open(data_path / "searchlight_pos_neg.pkl", 'rb') as f:
        searchlight, searchlight_scores  = pickle.load(f)

    plot_searchlight_outcome(fprep_f_paths, mask_paths, searchlight_scores, results_path)

    # get mask paths, load masks
    masks = load_masks(mask_paths)
    subject_mask = masks[subject]



if __name__ == "__main__":
    main()