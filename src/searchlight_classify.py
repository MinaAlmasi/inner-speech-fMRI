'''
Searchlight classification
'''
# import own functions
from utils import load_all_flms
from first_level import get_paths, get_events, get_confounds

# import packages
import pandas as pd
import nibabel
import nilearn
import numpy as np
import pathlib


def fl_matrix(events:pd.DataFrame, confounds:pd.DataFrame, fprep_f_paths):
    '''
    Create first level matrix for a single run. 

    Args:
        events: dataframe with events
        confounds: dataframe with confounds 
    
    Returns: 
        design_matrix: first level matrix 
    '''

    # calculate frame times
    TR = int(nib.load(fprep_f_paths[0]).header["pixdim"][4]) # get TR from first functional fmri path (based on https://nipy.org/nibabel/devel/biaps/biap_0006.html)
    frame_times = np.linspace(0, TR*len(confounds), len(confounds), endpoint=False)

    # make design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,  # we defined this earlier for interpolation!
        events=events,
        hrf_model='glover',
        drift_model=None,
        add_regs=confounds
    )

    return design_matrix

def fl_matrices(all_events:list, all_confounds:list, fprep_f_paths):
    '''
    Create first level matrix for all runs.

    Args
        all_events: list of dataframes with events
        confounds: dataframe with confounds
        fprep_f_paths: list of paths to preprocessed functional data

    Returns
        matrices: list of first level matrices
    '''
    matrices = []

    for event, confounds in zip(all_events, all_confounds):
        dm = first_level_matrix(event, confounds, fprep_f_paths)
        matrices.append(dm)

    return matrices

def main():
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    bids_path = data_path / "InSpePosNegData" / "BIDS_2023E"
    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, "0116", n_runs=6)
    
    # load flms, events and confounds
    flms = load_all_flms(data_path / "all_flms")
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths)

    
    print(confounds)

if __name__ == "__main__":
    main()