'''
Script to fit a first level model to all participants
'''
# utils & data
import pathlib, os
import pandas as pd

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

    return fprep_f_paths, event_paths, confounds_paths, mask_paths

def get_events(events_paths): 
    events = []

    for path in events_paths: 
        event_df = pd.read_csv(path, sep="\t")

        # remove all cols except onset, duration, trial type
        event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]
        
        events.append(event_df)

    return events 

def get_masks(mask_paths):
    masks = []

    for path in mask_paths: 
        mask = nib.load(path)
        masks.append(mask)

    # merge masks 
    mask_image = masking.intersect_masks(masks, threshold=0.8)

    return mask_image

def get_confounds(confound_paths):
    confounds = []

    for path in confound_paths:
        confounds_df = pd.read_csv(path, sep="\t")

        # select confound cols we are interested in
        confounds_df.loc[:, ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]

        confounds.append(confounds_df)

    return confounds

def first_level_fit(fprep_f_paths, event_paths, confounds_paths, mask_paths): 
    # get TR from first functional fmri path (based on https://nipy.org/nibabel/devel/biaps/biap_0006.html)
    TR = int(nib.load(fprep_f_paths[0]).header["pixdim"][4])

    # get events, confonds and mask img
    events = get_events(event_paths)
    confounds = get_confounds(confounds_paths) 
    mask_image = get_masks(mask_paths)

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


def main():
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # define root dir 
    path = pathlib.Path(__file__)
    bids_path = path.parents[3] / "816119" / "InSpePosNegData" / "BIDS_2023E"

    # get paths 
    fprep_f_paths, event_paths, confounds_paths, mask_paths = get_paths(bids_path, "0117", 6)
    print(fprep_f_paths)

    # get first level mdl for subject 0116
    first_level_mdl = first_level_fit(fprep_f_paths, event_paths, confounds_paths, mask_paths)


if __name__ == "__main__":
    main()


