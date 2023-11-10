'''
Script to perform GLM analysis on the data
'''
import pathlib
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.plotting import plot_stat_map
from scipy.stats import norm

from utils import load_all_flms

def second_level(flms):
    '''
    Perform second level analysis on the data using already created first level models
    '''
    # init second level model with smoothing parameter
    second_level_mdl = SecondLevelModel(smoothing_fwhm=8.0)

    # fit second level model
    second_level_mdl = second_level_mdl.fit(flms)

    return second_level_mdl

def plot_wholebrain_contrasts(second_level_mdl, contrast = "positive_img - negative_img", pval = 0.01, save_path = None):
    '''
    Plot wholebrain contrasts for a group with a second level model

    Args
        second_level_mdl: second level model
        contrast: contrast to plot

    Returns
        surface_plot, deep_plot: wholebrain plots of the contrast        
    '''
    # setting the threshold (converts wanted p-value into critical value for said p-value)
    threshold = norm.isf(pval)
    
    # compute contrassts
    zmap_g = second_level_mdl.compute_contrast(first_level_contrast = contrast, output_type='z_score')

    # plot contrast
    surface_plot = plotting.plot_glass_brain(zmap_g, cmap='blue_red',colorbar=True, threshold=threshold,
                          plot_abs=False)
                          
    deep_plot = plot_stat_map(zmap_g, cmap='cold_hot',threshold=threshold, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False)
    
    if save_path is not None:
        surface_plot.savefig(save_path / "surface_plot.png")
        deep_plot.savefig(save_path / "deep_plot.png")
    
    return surface_plot, deep_plot

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data" / "all_flms"

    # load all first level models
    flms_dict = load_all_flms(data_path)

    # get flms from dict
    flms = [val for val in flms_dict.values()]

    # perform second level analysis
    second_level_mdl = second_level(flms)

    # save path
    results_path = path.parents[1] / "results"

    # plot wholebrain contrasts
    surface_plot, deep_plot = plot_wholebrain_contrasts(second_level_mdl,  save_path = results_path)

if __name__ == "__main__":
    main()