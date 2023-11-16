'''
Searchlight classification
'''
import pathlib
import pickle

import numpy as np
import pandas as pd

from nilearn.image import new_img_like, load_img, index_img, clean_img, concat_imgs
from sklearn.model_selection import train_test_split, GroupKFold

from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from nilearn.decoding import SearchLight
from sklearn import naive_bayes, model_selection

def reshape_classify(b_maps, conditions_label): 
    '''
    Reshape data for classification
    '''
    # get ntrials
    n_trials=len(conditions_label)
    
    # concatenate bmaps
    b_maps_conc=concat_imgs(b_maps)

    #Find all negative and positive trials (NB differs from notebook!!!!!)
    idx_neg=[int(i) for i in range(len(conditions_label)) if 'negative_img' in conditions_label[i]]
    idx_pos=[int(i) for i in range(len(conditions_label)) if 'positive_img' in conditions_label[i]]
    idx_but=[int(i) for i in range(len(conditions_label)) if 'button_img' in conditions_label[i]]

    # recode labels to N, P, B (NB button press is not coded here!!)
    for i in range(len(conditions_label)):
        if i in idx_neg:
            conditions_label[i] = 'N'
        if i in idx_pos:
            conditions_label[i] = 'P'
        if i in idx_but:
            conditions_label[i] = 'B'

    # select conditions (indexes of relevant cnonds)
    idx = np.concatenate((idx_neg, idx_but))

    # select trials
    conditions = np.array(conditions_label)[idx]
    
    # select bmaps
    b_maps_img = index_img(b_maps_conc, idx)

    # Make an index for spliting fMRI data with same size as class labels
    idx2 = np.arange(conditions.shape[0])

    # create training and testing vars on the basis of class labels (is this the correct split?? Notebook says this)
    idx_train, idx_test, conditions_train,  conditions_test = train_test_split(idx2, conditions, test_size=0.2)
    
    # reshape data
    fmri_img_train = index_img(b_maps_img, idx_train)
    fmri_img_test = index_img(b_maps_img, idx_test)




def main(): 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data" / "searchlight"

    # load all flms, trial_dms for particular subject
    with open(data_path / "all_flms.pkl", 'rb') as f:
        all_flms, trial_dms  = pickle.load(f)

    # load all bmaps and condition labels for particular subject
    with open(data_path / "bmaps_conditions.pkl", 'rb') as f:
        b_maps, conditions_label  = pickle.load(f)

    # check that they match labels (a little worrying that they do not match notebook - i think maybe it is due to jupyter cells not being run in the same order)
    print(trial_dms[0].columns[0:9])
    print(conditions_label[0:9])
    print(conditions_label)

    # reshape
    reshape_classify(b_maps, conditions_label)



if __name__ == "__main__":
    main()