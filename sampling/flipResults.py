
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

#%%
mcmcResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jana_SRI/"

lmiResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_ivan_SRI/"

lnmiOriginal = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jonas_SRI/"

cmaesPriorResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_5_nSamples-600_priorInit-False_factorSTD-13_5/"

cmaesNaiveResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_5_nSamples-600_priorInit-True_factorSTD-13_5/"

tumorSegmentationPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_PatientData_SRI/"
# %% save again flipped correctly
for i in range(100):
    try: 
        patientName = "rec" + ("0000" + str(i))[-3:] + "_pre"

        segmentation = nib.load(os.path.join(tumorSegmentationPath, patientName, "tumorFlair_flippedCorrectly.nii")).get_fdata()

        mcmcResult = nib.load(os.path.join(mcmcResultPath, patientName, "MAP.nii")).get_fdata()
        mcmcResult = np.flip(mcmcResult, axis=0)
        nib.save(nib.Nifti1Image(mcmcResult, np.eye(4)), os.path.join(mcmcResultPath, patientName, "MAP.nii").replace('.nii', '_flippedCorrectly.nii'))


        lmiResult = nib.load(os.path.join(lmiResultPath, patientName, "inferred_tumor_patientspace_mri.nii")).get_fdata()
        lmiResult = np.flip(lmiResult, axis=0)
        nib.save(nib.Nifti1Image(lmiResult, np.eye(4)), os.path.join(lmiResultPath, patientName, "inferred_tumor_patientspace_mri.nii").replace('.nii', '_flippedCorrectly.nii'))

        lnmiResult = nib.load(os.path.join(lnmiOriginal, patientName, "predictionJonas.nii")).get_fdata()
        lnmiResult = np.flip(lnmiResult, axis=0)
        nib.save(nib.Nifti1Image(lnmiResult, np.eye(4)), os.path.join(lnmiOriginal, patientName, "predictionJonas.nii").replace('.nii', '_flippedCorrectly.nii'))

        print("did for ", patientName)
    except:
        pass