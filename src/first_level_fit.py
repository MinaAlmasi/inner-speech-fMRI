'''
Script to fit a first level model to all participants
'''
import pathlib
import nibabel as nib

def first_level_fit(bids_path, subject:str, n_runs:int):
    '''
    Fit first level model on a particular subject

    Args
        bids_path: path to bids directory (root)
        subject: ID of subject (e.g., "0116")
        n_runs: number of blocks in the experiment

    Returns
        first_level_mdl: first level model
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
        pass


def main():
     path = pathlib.Path(__file__)

     bids_path = path.parents[3] / "816119" / "InSpePosNegData" / "BIDS_2023E"

     first_level_fit(bids_path, "0116", 6)


if __name__ == "__main__":
    main()


