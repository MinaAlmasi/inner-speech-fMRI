'''
Searchlight classification
'''
import pathlib
import pickle

import numpy as np
import pandas as pd

from nilearn.image import index_img, concat_imgs
from nilearn.image import new_img_like, load_img
from sklearn.model_selection import train_test_split

from nilearn.decoding import SearchLight
from sklearn.naive_bayes import GaussianNB

def remake_labels(conditions_label): 
    '''
    Remake labels for classification
    '''
    # get ntrials
    n_trials=len(conditions_label)

    #Find all negative and positive trials (NB differs from notebook as our trials are coded slightly differently)
    idx_neg=[int(i) for i in range(n_trials) if 'negative_img' in conditions_label[i]]
    idx_pos=[int(i) for i in range(n_trials) if 'positive_img' in conditions_label[i]]
    idx_but=[int(i) for i in range(n_trials) if 'button_img' in conditions_label[i]]
    idx_but_press=[int(i) for i in range(n_trials) if 'button_press' in conditions_label[i]]

    # recode labels to N, P, B (NB button press is not coded here!!)
    for i in range(n_trials):
        if i in idx_neg:
            conditions_label[i] = "N"
        if i in idx_pos:
            conditions_label[i] = "P"
        if i in idx_but:
            conditions_label[i] = "B"
        if i in idx_but_press:
            conditions_label[i] = "BP"

    return idx_neg, idx_pos, idx_but, idx_but_press, conditions_label


def reshape_classify(idx_cond1, idx_cond2, conditions_label, b_maps):
    '''
    Reshape for classification. Select two conditions of interest by inserting their indicies.

    Args
        idx_cond1: indicies of condition 1
        idx_cond2: indicies of condition 2 
        conditions_label: labels of conditions 
        b_maps: beta maps 
    '''
    # concatenate bmaps
    b_maps_conc=concat_imgs(b_maps)

    # select conditions (indexes of relevant cnonds)
    idx = np.concatenate((idx_cond1, idx_cond2))

    # select trials
    conditions = np.array(conditions_label)[idx]
    
    # select bmaps
    b_maps_img = index_img(b_maps_conc, idx)

    # Make an index for spliting fMRI data with same size as class labels
    idx2 = np.arange(conditions.shape[0])

    # create training and testing vars on the basis of class labels (is this the correct split?? Notebook says this)
    idx_train, idx_test, conditions_train, conditions_test = train_test_split(idx2, conditions, test_size=0.2)
    
    # reshape data
    fmri_img_train = index_img(b_maps_img, idx_train)
    fmri_img_test = index_img(b_maps_img, idx_test)

    return fmri_img_train, fmri_img_test, conditions_train, conditions_test


def run_searchlight(mask_img, fmri_img_train, conditions_train, data_path, filename):
    '''
    Run searchlight classification. Save to data path.
    '''
    # initialize searchlight
    print("Intializing searchlight ...")
    searchlight = SearchLight(
    mask_img,
    estimator=GaussianNB(),
    radius=5, n_jobs=-1,
    verbose=5, cv=3)
    
    # fit searchlight
    print("Fitting searchlight ...")
    searchlight.fit(fmri_img_train, conditions_train)
    
    # save searchlight pickle
    f = open(data_path / filename, 'wb')
    pickle.dump([searchlight, searchlight.scores_], f)
    f.close()

    return searchlight


def main(): 
    subject = "0116"
    
    # define paths 
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "searchlight"
    bids_path = path.parents[2] / "data" / "InSpePosNegData" / "BIDS_2023E"

    # load all flms, trial_dms for particular subject
    with open(data_path / "all_flms.pkl", 'rb') as f:
        all_flms, trial_dms  = pickle.load(f)

    # load all bmaps and condition labels for particular subject
    with open(data_path / "bmaps_conditions.pkl", 'rb') as f:
        b_maps, conditions_label  = pickle.load(f)

    # remake labels
    idx_neg, idx_pos, idx_but, idx_but_press, conditions_label = remake_labels(conditions_label)
    
    # reshape and split for classification on the conditions we are interested in 
    cond1, cond2 = idx_pos, idx_neg
    fmri_img_train, fmri_img_test, conditions_train, conditions_test = reshape_classify(cond1, cond2, conditions_label, b_maps)

    # get mask paths, load masks
    mask_path = bids_path / pathlib.Path(f'derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    subject_mask = load_img(mask_path)

    # run searchlight 
    filename = 'searchlight_pos_neg.pkl'
    searchlight = run_searchlight(subject_mask, fmri_img_train, conditions_train, data_path, filename)


if __name__ == "__main__":
    main()