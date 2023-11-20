'''
Preparation for searchlight classification
'''

import pathlib

# import own functions
import sys 
sys.path.append(str(pathlib.Path(__file__).parents[2] / "src"))

from utils import load_all_flms
from first_level import get_paths, get_events, get_confounds

# import packages
import pandas as pd
import nibabel as nib
import pickle
import nilearn
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel

def first_level_matrix(events:list, confounds:list, fprep_f_paths): 
    '''
    Create first level matrix for a single participant
    '''
    
    # calculate frame times
    TR = int(nib.load(fprep_f_paths[0]).header["pixdim"][4]) # get TR from first functional fmri path (based on https://nipy.org/nibabel/devel/biaps/biap_0006.html)
    frame_times = np.linspace(0, TR*len(confounds[0]), len(confounds[0]), endpoint=False)

    trial_dms = []

    for idx, event_df in enumerate(events):
        N = event_df.shape[0]

        trials = pd.DataFrame(event_df, columns=['onset'])
        trials.loc[:, 'duration'] = 0.7
        trials.loc[:, 'trial_type'] = [event_df['trial_type'][i-1]+'_'+'t_'+str(i).zfill(3)  for i in range(1, N+1)]

        # lsa_dm = least squares all design matrix
        trial_dms.append(make_first_level_design_matrix(
            frame_times=frame_times,  # we defined this earlier 
            events=trials,
            add_regs=confounds[idx], #Add the confounds from fmriprep
            hrf_model='glover',
            drift_model='cosine'  
        ))

    return trial_dms


def flm_new_design_matrix(events:list, confounds:list, fprep_f_paths, trial_dms, data_path): 
    models=[]
    
    for idx, event_df in enumerate(events):
        # load functional images
        imgs = fprep_f_paths[idx]

        # ready the model
        models.append(FirstLevelModel())

        # Fit the model and append it
        print('Fitting GLM: ', idx+1)
        models[idx].fit(imgs,design_matrices=trial_dms[idx])

    # save file with all models
    f = open(data_path / "searchlight" / "all_flms.pkl", "wb")
    pickle.dump([models, trial_dms], f)
    f.close()

    return models


def create_bmaps(events, trial_dms, models, data_path): 
    b_maps = []
    conditions_label = []

    for idx, event_df in enumerate(events):
        N=event_df.shape[0]
        # make an identity matrix with N= number of trials
        contrasts=np.eye(N)

        # g ind difference between columns in design matrix and number of trials
        dif=trial_dms[idx].shape[1]-contrasts.shape[1]
        
        #Pad with zeros
        contrasts=np.pad(contrasts, ((0,0),(0,dif)),'constant')
        
        print('Making contrasts for session : ', idx+1)
        print('Number of contrasts : ', N)
        
        for i in range(N):
            # Add a beta-contrast image from each trial
            b_maps.append(models[idx].compute_contrast(contrasts[i,], output_type='effect_size'))
            
            # Make a variable with condition labels for use in later classification
            conditions_label.append(trial_dms[idx].columns[i])

    f = open(data_path / "searchlight" / "bmaps_conditions.pkl", "wb")
    pickle.dump([b_maps, conditions_label], f)
    f.close()

    return b_maps, conditions_label

def main():
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data"
    bids_path = data_path / "InSpePosNegData" / "BIDS_2023E"
    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, "0116", n_runs=6)
    
    # load flms, events and confounds
    flms = load_all_flms(data_path / "all_flms")
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths)

    # create first level matrices
    trial_dms = first_level_matrix(events, confounds, fprep_f_paths)

    # create new first_level_models
    models = flm_new_design_matrix(events, confounds, fprep_f_paths, trial_dms, data_path)
    
    # create bmaps
    bmaps, conditions_label = create_bmaps(events, trial_dms, models, data_path)

if __name__ == "__main__":
    main()