'''
Plot outcome of searchlight analysis from train.py
'''

# utils
import pathlib, pickle

# custom functions
import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2] / "src"))
from first_level import get_paths

# 
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn import plotting

def plot_searchlight_outcome(fprep_f_paths, mask_paths, searchlight_scores, results_path):
    """
    Plot searchlight results
    """
    # create an image of the searchlight scores
    searchlight_img = new_img_like(fprep_f_paths[0], searchlight_scores)

    # plot the searchlight scores
    surface_plot = plotting.plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='Positive vs Negative (Acc>0.6')
    deep_plot = plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='Positive vs Negative (Acc>0.6')

    # save the plots
    surface_plot.savefig(results_path / "searchlight_surface_plot.png")
    deep_plot.savefig(results_path / "searchlight_deep_plot.png")


def main():
    subject = "0116"

    # define paths 
    path = pathlib.Path(__file__)
    bids_path = path.parents[2] / "data" / "InSpePosNegData" / "BIDS_2023E"

    data_path = path.parents[2] / "data" / "searchlight"
    results_path = path.parents[2] / "results"

    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, subject, n_runs=6)
    
    # load the searchlight file
    with open(data_path / "searchlight_pos_neg.pkl", 'rb') as f:
        searchlight, searchlight_scores  = pickle.load(f)

    plot_searchlight_outcome(fprep_f_paths, mask_paths, searchlight_scores, results_path)

if __name__ == "__main__":
    main()