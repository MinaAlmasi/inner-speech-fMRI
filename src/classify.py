'''
Script to perform classification
'''
import nilearn
import pathlib

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

# custom module 
from first_level import get_paths, get_events, get_confounds, get_masks


def first_level_matrix(events:pd.DataFrame, confounds:pd.DataFrame, fprep_f_paths):
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

def extract_trial_matrices(design_matrix, events:pd.DataFrame):
    '''
    Create design matrices for each trial (LSS approach) for a single run. 

    Args
        design_matrix: first level matrix for the particular run
        events: events dataframe pertaining to the design matrix 

    Returns 

    '''

    # empty lst for design matrices
    trial_dms = []

    # iterate over cols in design_matrix (except the intercept, i.e., -1)
    for i, col in enumerate(design_matrix.columns[:-1]):
        print("Extracting matrices ...")

        # extract single trial regressor/predictor 
        trial_regressor = design_matrix.loc[:, col]

        # for regressors per condition
        other_regressors = []

        # loop across unique conditions
        for condition in np.unique(events["trial_type"]):
            # find cols that belong to the current condition
            idx = condition == np.array(events["trial_type"])

            # make sure to not include trial that is being currently estimated
            idx[i] = False
            
            # also exclude last intercept
            idx = np.append(idx, False)

            # extract all N-1 regressors
            regressors = design_matrix.loc[:, idx]

            # sum together to create a single regressor for current condition
            current_condition_regressor = regressors.sum(axis=1)

            other_regressors.append(current_condition_regressor)

        # concatenate condition regresssors
        other_regressors = pd.concat(other_regressors)

        # concatenate single trial regressor and N condition regressors 
        dm = pd.concat([trial_regressor, other_regressors])

        # add back intercept
        dm.loc[:, "intercept"] = 1

        # add sensible col names
        dm.columns = ['trial_to_estimate'] + list(set(events['trial_type'])) + ['intercept']

        trial_dms.append(dm)
    
    print(f"Sucess! {len(trial_dms)} design matrices have been created!")

    return trial_dms

def main(): 
    # define root dir 
    path = pathlib.Path(__file__)
    bids_path = path.parents[1] / "data" / "InSpePosNegData" / "BIDS_2023E"

    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, "0118", n_runs=6)

    # get events, confounds
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths)
    
    design_matrix_1 = first_level_matrix(events[0], confounds[0], fprep_f_paths)
    print(design_matrix_1)
    
    #trial_dms = extract_trial_matrices(design_matrix_1, events[0])

    #print(trial_dms)



if __name__ == "__main__":
    main()