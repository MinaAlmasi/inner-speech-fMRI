'''
Script to perform classification
'''
import nilearn
import pathlib

import nibabel as nib
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix

# custom module 
from first_level import get_paths, get_events, get_confounds, get_masks


def first_level_matrix(events, confounds):
    '''
    Create first level matrix for a single run. 
    '''

    # calculate frame times
    TR = int(nib.load(fprep_f_paths[0]).header["pixdim"][4]) # get TR from first functional fmri path (based on https://nipy.org/nibabel/devel/biaps/biap_0006.html)
    frame_times = np.linspace(0, TR*len(confounds[0]), len(confounds[0]))

    # make design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,  # we defined this earlier for interpolation!
        events=events[0],
        hrf_model='glover',
        drift_model=None,
        add_regs=confounds[0]
    )

    return design_matrix

    

def main(): 
    # define root dir 
    path = pathlib.Path(__file__)
    bids_path = path.parents[1] / "data" / "InSpePosNegData" / "BIDS_2023E"

    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, "0118", n_runs=6)

    # get events, confounds
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths)
    
    design_matrix = first_level_matrix(fprep_f_paths, event_paths, confounds_paths)
    
    print(design_matrix)



if __name__ == "__main__":
    main()