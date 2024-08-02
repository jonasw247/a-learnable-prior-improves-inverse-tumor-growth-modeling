#%%
import torch
import sys
import numpy as np
import dataloader
import nibabel as nib
import os
import matplotlib.pyplot as plt
import network
import torch.nn.functional as F
from torch.utils.data import DataLoader

def dataloader(wmPath, gmPath, csfPath, flairPath, corePath, excludeCSFInSegmentation, doSaveAgain = True):
    #for i in range(len(patients)):

    flair = np.flip(nib.load(flairPath).get_fdata() , axis=0)
    core = np.flip(nib.load(corePath).get_fdata(), axis = 0)

    thr_volume = 0.66666 * core + 0.33333 * flair

    

    whiteMatter = np.flip(nib.load(wmPath).get_fdata(), axis=0)
    grayMatter = np.flip(nib.load(gmPath).get_fdata(), axis=0)
    csfMatter = np.flip(nib.load(csfPath).get_fdata(), axis=0)

    if doSaveAgain:
        nib.save(nib.Nifti1Image(thr_volume, np.eye(4)), flairPath.replace('.nii', '_flippedCorrectly.nii'))
        nib.save(nib.Nifti1Image(core, np.eye(4)), corePath.replace('.nii', '_flippedCorrectly.nii'))
        nib.save(nib.Nifti1Image(whiteMatter, np.eye(4)), wmPath.replace('.nii', '_flippedCorrectly.nii'))
        nib.save(nib.Nifti1Image(grayMatter, np.eye(4)), gmPath.replace('.nii', '_flippedCorrectly.nii'))
        nib.save(nib.Nifti1Image(csfMatter, np.eye(4)), csfPath.replace('.nii', '_flippedCorrectly.nii'))


    if excludeCSFInSegmentation:

        thr_volume[csfMatter >= whiteMatter + grayMatter] = 0


    #thrvolume_resized = np.expand_dims( np.flip(thr_volume, axis=0), -1)
    thrvolume_resized = np.expand_dims( thr_volume, -1)

    whiteMatter_resized = np.expand_dims(whiteMatter, -1)
    grayMatter_resized = np.expand_dims(grayMatter, -1)
    csfMatter_resized = np.expand_dims(csfMatter, -1)

    # change take the tumor volume twice
    #nn_input = np.concatenate((thrvolume_resized, thrvolume_resized), -1)
    nn_input = np.concatenate((thrvolume_resized, whiteMatter_resized, grayMatter_resized, csfMatter_resized), -1)

    nninput_resized = nn_input.transpose((3, 0, 1, 2))

    return torch.from_numpy(nninput_resized.astype(np.float32))[None]

def evalModelRealPat(device = 'cpu', isSRI = False, excludeCSFInSegmentation = False, modelNumber = -1):

    #-1 means the original model
    if modelNumber == -1:
        modelStatePath = "/mnt/8tb_slot8/jonas/datasets/modelWeights/jonasWeightsWithTissue/epoch49.pt"
    else:
        modelFolderPath = "/mnt/8tb_slot8/jonas/datasets/modelWeights/jonasWeightsWithTissueFinetuned/"

        logfolder = sorted(os.listdir(modelFolderPath))[modelNumber]
        modelStatePath = modelFolderPath+  logfolder + '/bestval-model.pt'

    #tissueFolder ='/mnt/Drive3/jonas/mich_rec/mich_rec_128_maskedAndCutS3Tissue' #this is the old one, repalced by the one below in sri space
    if isSRI:
        tissueFolder = '/mnt/8tb_slot8/jonas/datasets/mich_rec/mich_rec_SRI_S3_maskedAndCut_rescaled_128'
        tumorSegmentationFolder  = '/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_PatientData_SRI'
    else:
        #throw error
        print('not implemented!!!')

    checkpoint = torch.load(modelStatePath, map_location=torch.device(device=device))

    model = network.NetConstant_noBN_64_n4_l4_inplacefull(5, None, False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.eval()

    yPreds = []
    paths = []

    patients = sorted(os.listdir(tumorSegmentationFolder))

    with torch.set_grad_enabled(False):
        for i in range(len(patients)):
            x = dataloader(os.path.join(tissueFolder, patients[i], 'WM.nii'),
                           os.path.join(tissueFolder, patients[i], 'GM.nii'),
                           os.path.join(tissueFolder, patients[i], 'CSF.nii'),
                           os.path.join(tumorSegmentationFolder, patients[i], 'tumorFlair.nii'),  
                           os.path.join(tumorSegmentationFolder, patients[i], 'tumorCore.nii'), excludeCSFInSegmentation)
            x= x.to(device)
            y_predicted = model(x)
            
            yPred = y_predicted.cpu().detach().numpy()

            yPreds.append(yPred[0])
            paths.append(os.path.join(tumorSegmentationFolder, patients[i]))
            print(i, patients[i])

    return  np.array(yPreds), paths

def convert(mu1, mu2, x, y, z):
    normalization_range = [-1.0, 1.0]
    T = 100 

    mu1 = np.interp(mu1, normalization_range, [np.sqrt(0.01), np.sqrt(22.5)]) 
    mu2 = np.interp(mu2, normalization_range, [np.sqrt(0.1), np.sqrt(300)])  

    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = mu1**2 / T
    rho =  mu2**2 / T 

    Tret = T + 0*rho

    return D, rho, Tret, x, y, z , mu1, mu2

if __name__ == '__main__':

    predDws, predRhos, predTends, predIcxs, predIcys, predIczs, mu1s, mu2s = [], [], [], [], [], [], [], []
    for modelNumber in range(10):
        ypredT, paths = evalModelRealPat(isSRI = True, excludeCSFInSegmentation = False, modelNumber = modelNumber)
        ypred = ypredT.T

        predDw, predRho, predTend, predIcx, predIcy, predIcz, mu1, mu2 = convert(ypred[0], ypred[1], ypred[2], ypred[3], ypred[4])

        predDws.append(predDw)
        predRhos.append(predRho)
        predTends.append(predTend)
        predIcxs.append(predIcx)
        predIcys.append(predIcy)
        predIczs.append(predIcz)
        mu1s.append(mu1)
        mu2s.append(mu2)

    #%%
    ypredT, paths = evalModelRealPat(isSRI = True, excludeCSFInSegmentation = False, modelNumber = -1)
    ypred = ypredT.T

    predDwSingle, predRhoSingle, predTendSingle, predIcxSingle, predIcySingle, predIczSingle,  mu1Single, mu2Single = convert(ypred[0], ypred[1], ypred[2], ypred[3], ypred[4])

    #%%
    variableNames = ['D - cm/d', 'rho 1/d ', 'T', 'x', 'y', 'z' , 'mu1 - cm', 'mu2 - unitless', 'originDiff']

    allPredictions = [predDws, predRhos, predTends, predIcxs, predIcys, predIczs, mu1s, mu2s]

    allSingles = [predDwSingle, predRhoSingle, predTendSingle, predIcxSingle, predIcySingle, predIczSingle, mu1Single, mu2Single]

    patientLabels = sorted(os.listdir("/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_PatientData_SRI/"))
    patientNumberOriginal = [int(x.split('_')[0].split("c")[-1]) for x in patientLabels]
    for i in range(len(allPredictions)):

        finalArr = []
        finalArr.append(patientNumberOriginal)
        for modelIDX in range(10):
            finalArr.append(allPredictions[i][modelIDX])

        finalArr.append(allSingles[i])

        respath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/parameterResultsCSV/"
        np.savetxt(respath +variableNames[i].replace(' ','').replace('/', '-') +'.csv', np.array(finalArr).T, delimiter=',', header='patient, singleNetwork, ensemble0, ensemble1,ensemble2,ensemble3,ensemble4,ensemble5,ensemble6,ensemble7,ensemble8,ensemble9', comments='')

    # %%
    import matplotlib.pyplot as plt
    plt.plot(predDw)
    plt.plot(predRho)
# %%
