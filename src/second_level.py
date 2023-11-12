'''
Script to perform GLM analysis on the data
'''
import pathlib
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.plotting import plot_stat_map
from scipy.stats import norm
import atlasreader 

from utils import load_all_flms, remove_flms

def second_level(flms):
    '''
    Perform second level analysis on the data using already created first level models
    '''
    # init second level model with smoothing parameter (njobs = -2 to use all cores EXCEPT 1 for faster compute)
    second_level_mdl = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-2)

    # fit second level model
    second_level_mdl = second_level_mdl.fit(flms)

    return second_level_mdl

def plot_wholebrain_contrasts(second_level_mdl, contrast = "positive_img - negative_img", pval = 0.001, save_path = None):
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
    
    return surface_plot, deep_plot, zmap_g

def get_atlas(stat_map, save_path, pval:int=0.001): 
    '''
    Use atlas reader to extract coordinates 

    Args
        stat_map: fmri image statistical map (e.g., zmap_g)
    '''
    threshold = norm.isf(pval)

    atlasreader.create_output(stat_map, voxel_thresh=threshold, cluster_extent=10, outdir=save_path)
    

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data" / "all_flms"

    # load all first level models (excludes subject 0119 based on sanity check)
    flms_dict = load_all_flms(data_path, exclude_subjects=["0119"])

    # get flms only 
    flms = [val for val in flms_dict.values()]

    # perform second level analysis
    print("[INFO:] Making second level model ...")
    second_level_mdl = second_level(flms)

    # save path
    results_path = path.parents[1] / "results"

    # plot wholebrain contrasts
    print("[INFO:] Plotting results ...")
    surface_plot, deep_plot, zmap_g = plot_wholebrain_contrasts(second_level_mdl, pval=0.001, save_path = results_path)

    # read atlas 
    print("[INFO:] Finding clusters ...")
    atlas = get_atlas(zmap_g, results_path / "atlas_reader", pval=0.001)
    
if __name__ == "__main__":
    main()