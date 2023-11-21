from sklearn.naive_bayes import GaussianNB
from nilearn.input_data import NiftiMasker
import pathlib
from nilearn.image import new_img_like, load_img
from sklearn.model_selection import permutation_test_score
import numpy as np
import pickle

def find_most_important_voxels(searchlight_scores, mask_wb_filename, n_voxels=500):
    """
    Find the most important voxels in the searchlight analysis.

    Args:
        searchlight_scores (numpy array): array of scores from the searchlight analysis
        anat_filename (nifti image): anatomical image of the subject
        n_voxels (int): number of voxels to select

    
    Returns:
        process_mask_img (nifti image): mask of the voxels to be used in the permutation test
        cut (float): cutoff value for the scores

    """

    # find the percentile that makes the cutoff for x best voxels
    perc=100*(1-n_voxels/searchlight_scores.size)
    
    # find the cutoff
    cut=np.percentile(searchlight_scores, perc)

    # load mask
    mask_img = load_img(mask_wb_filename)

    # make copy of mask
    process_mask = mask_img.get_fdata().astype(int)

    # set all voxels below the cutoff to 0
    process_mask[searchlight_scores<cut]=0

    process_mask_img = new_img_like(mask_img, process_mask)

    return process_mask_img, cut


def do_permutation(process_mask_img, fmri_img_test, conditions_test, data_path):
    """
    Does permutation test on the test data based on the process mask.

    Args:
        process_mask_img (nifti image): mask of the voxels to be used in the permutation test (top 500)
        fmri_img_test (nifti image): fMRI image of the test data
        conditions_test (list): conditions of the test data
        data_path (pathlib path): path to the data folder
    
    Returns:
        score_cv_test (float): classification score of the test data
        scores_perm (numpy array): array of scores from the permutation test
        pvalue (float): pvalue of the permutation test
    
    """
    # We have our fMRI data as a 4D file, we need to use a masker to convert it to a
    masker = NiftiMasker(mask_img=process_mask_img, standardize=False)

    # We use masker to retrieve a 2D array ready for machine learning with scikit-learn
    fmri_masked = masker.fit_transform(fmri_img_test)

    print(fmri_masked.shape)

    # Create the model
    score_cv_test, scores_perm, pvalue = permutation_test_score(
        GaussianNB(), fmri_masked, conditions_test, cv=3, n_permutations=1000, 
        n_jobs=-1, random_state=2502, verbose=0, scoring=None)

    # Save the results
    f = open(data_path / 'permutation_results.pkl', 'wb')
    pickle.dump([score_cv_test, scores_perm, pvalue], f)
    f.close()

    # Return the results and print them
    print("Classification score %s (pvalue : %s)" % (score_cv_test, pvalue))

    return score_cv_test, scores_perm, pvalue


def main(): 
    subject = "0117"

    # define paths 
    path = pathlib.Path(__file__)
    bids_path = path.parents[2] / "data" / "InSpePosNegData" / "BIDS_2023E"

    data_path = path.parents[2] / "data" / "searchlight"
    results_path = path.parents[2] / "results"

    mask_wb_filename = bids_path / pathlib.Path(f'derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')

    # load the searchlight file
    with open(data_path / "searchlight_pos_neg.pkl", 'rb') as f:
        searchlight, searchlight_scores  = pickle.load(f)

    # load reshaped data 
    with open(data_path / "searchlight_reshaped_data.pkl", 'rb') as f:
        fmri_img_train, fmri_img_test, conditions_train, conditions_test = pickle.load(f)

    # find top 500 voxels
    process_mask_img, cut = find_most_important_voxels(searchlight_scores, mask_wb_filename, n_voxels=500)

    # do permutation test
    score_cv_test, scores_perm, pvalue = do_permutation(process_mask_img, fmri_img_test, conditions_test, data_path)
    

if __name__ == "__main__":
    main()