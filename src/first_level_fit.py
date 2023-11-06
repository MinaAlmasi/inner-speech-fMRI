'''
Script to fit a first level model to all participants
'''
# utils & data
import pathlib, os
import pandas as pd
import numpy as np
import pickle

# neuroimaging 
import nibabel as nib
from nilearn import masking
from nilearn.glm.first_level import FirstLevelModel

def get_paths(bids_path, subject:str, n_runs:int):
    '''
    Get all paths to files needed to fit a first level model for a particular subject.

    Args
        bids_path: path to bids directory (root)
        subject: ID of subject (e.g., "0116")
        n_runs: number of blocks in the experiment

    Returns
        fprep_f_paths: path to functional fMRIprep processed data
        event_paths: path to events (trigger) data
        confounds_paths: paths to the confound data 
        mask_paths: paths needed to fit a first level model (functional fmriPREP processed data, events, confounds, masks)
    '''
    # define space ()
    space = "MNI152NLin2009cAsym"

    # define functional paths
    raw_fdir = bids_path / f"sub-{subject}" / "func"
    fprep_fdir = bids_path / f"derivatives" / f"sub-{subject}" / "func"

    # get fprep paths 
    fprep_f_paths = [fprep_fdir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz" for run in range(1, n_runs+1)]

    # get event paths 
    event_paths = [path for path in raw_fdir.iterdir() if path.name.endswith("_events.tsv")]

    # get confounds paths
    confounds_paths = [path for path in fprep_fdir.iterdir() if path.name.endswith("_desc-confounds_timeseries.tsv")]

    # get mask paths 
    mask_paths = [path for path in fprep_fdir.iterdir() if path.name.endswith(f"_space-{space}_desc-brain_mask.nii.gz")]

    # sort lists in place (to ensure that the runs are in the right order)
    for p in [fprep_f_paths, event_paths, confounds_paths, mask_paths]:
        p.sort()
    
    return fprep_f_paths, event_paths, confounds_paths, mask_paths


def add_buttonpress_events(events_dfs):
    '''
    Add button press events to events dataframe. 
    This is done by adding a new row to the dataframe with the onset + RT of the trial and a trial type of "button_press".

    Args
        events_dfs: list of dataframes containing events data

    Returns
        modified_events_dfs: list of dataframes containing events data with button press events added
    '''
    modified_events_dfs = []
    
    # iterate over runs
    for df in events_dfs:

        # iterate over events in each run
        for i, row in df.iterrows():
            
            # get trial type, onset and reaction time
            trial_type = row["trial_type"]
            onset = row["onset"]
            rt = row["RT"]

            if trial_type == "IMG_BI" and np.isnan(rt) == False:
                # calculate new onset based on reaction
                new_onset = onset + rt

                new_trial_type = "button_press"

                # define new row
                new_row = {"onset": new_onset, "duration": 0.1, "trial_type": "button_press", "RT": 0}
                
                # append new row to df
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # sort df by onset
        df = df.sort_values(by="onset").reset_index(drop=True)

        # drop RT col
        df = df.drop(columns=["RT"])

        # append modified df to list
        modified_events_dfs.append(df)

    return modified_events_dfs


def get_events(events_paths): 
    events = []

    # read in events 
    for path in events_paths: 
        # load the raw events df
        event_df = pd.read_csv(path, sep="\t")

        # remove all cols except onset, duration, trial type, RT (RT is only used to calculate button_press triggers)
        event_df = event_df.loc[:, ["onset", "duration", "trial_type", "RT"]]
        
        # append to events
        events.append(event_df)

    # add the button press events (also removes RT)
    events = add_buttonpress_events(events)

    return events 

def get_masks(mask_paths, save_path = None):
    masks = []

    for path in mask_paths: 
        mask = nib.load(path)
        masks.append(mask)

    # get subject id (from first mask path, does not matter which as all pertain to subject)
    subject_name = mask_paths[0].name[4:8]

    # merge masks 
    mask_image = masking.intersect_masks(masks, threshold=0.8)

    # if specified, save the mask image for future use (also used in sanity_check.py)
    if save_path:
        file_path = save_path / "mask_objects"
        file_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(mask_image, open(file_path / f"{subject_name}.pkl", "wb"))
        
    return mask_image


def get_confounds(confound_paths):
    confounds = []

    for path in confound_paths:
        confounds_df = pd.read_csv(path, sep="\t")

        # select confound cols we are interested in
        confounds_df.loc[:, ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]

        # replace all nans # NB to whether this is the correct solution (0s should not do a big impact https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326/18)
        # for other solution, see maybe https://carpentries-incubator.github.io/SDC-BIDS-fMRI/05-data-cleaning-with-nilearn/index.html
        confounds_df = confounds_df.fillna(0)

        confounds.append(confounds_df)

    return confounds                
    

def first_level_fit(fprep_f_paths, event_paths, confounds_paths, mask_paths, save_path): 
    # get TR from first functional fmri path (based on https://nipy.org/nibabel/devel/biaps/biap_0006.html)
    TR = int(nib.load(fprep_f_paths[0]).header["pixdim"][4])

    # get events, confonds and mask img
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths) 
    mask_image = get_masks(mask_paths, save_path = save_path)

    # create first lvl model 
    first_level_mdl = FirstLevelModel(
        t_r=TR,
        slice_time_ref=0, # default val, ask Mikkel as notebook 13 has it set to 0.5 
        hrf_model="glover", 
        mask_img = mask_image, 
        noise_model="ols", 
        verbose=1
    )

    # fit model 
    first_level_mdl.fit(fprep_f_paths, events, confounds)

    return first_level_mdl

def all_subjects_pipeline(bids_path, subjects_list, save_path=None):
    for subject in subjects_list: 
        # get paths
        fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, subject, n_runs=6)
        
        # first level model 
        first_level_mdl = first_level_fit(fprep_f_paths, event_paths, confounds_paths, mask_paths, save_path=save_path)

        # save if savepath is 
        if save_path:
            file_name = f"flm_{subject}.pkl"
            file_path = save_path / "all_flms"
            file_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(first_level_mdl, open(file_path / file_name, "wb"))


def main():
    # set number of threads to 1 as it otherwise hogs memory to run first level models
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # define root dir 
    path = pathlib.Path(__file__)
    bids_path = path.parents[1] / "data" / "InSpePosNegData" / "BIDS_2023E"

    save_path = path.parents[1] / "data"
    save_path.mkdir(parents=True, exist_ok=True)
    
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    all_subjects_pipeline(bids_path, subjects, save_path=save_path)


if __name__ == "__main__":
    main()


