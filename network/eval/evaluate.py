#%%
import numpy as np
import torch

import os
import subprocess
import shutil
import pickle

from toolsForEvaluation import evalModel
import matplotlib.pyplot as plt

# %%
# parameters
varNames = ["mu1", "mu2", "x", "y", "z"]
T_global = 100
endTest = 10

sweepTime = True

modelState = "25kSamplesDiffBGTissueAll_epoch49"

if modelState == "25kSamplesDiffBGTissueAll_epoch49":
    modelStatePath = "/home/home/jonas/programs/learn-morph-infer/log/1403-15-48-19-v7_1-jonasTest_with25kSamplesDiffBGTissueAll/epoch49.pt"

set = 'training'
test_data_path = None
bgTissueFolder = None
if set == 'training':
    test_data_path = "/mnt/Drive3/jonas/synthetic_data/2023_3_5___21_55_34_Sampled30k/Dataset/npz_data"
    bgTissueFolder = '/mnt/Drive3/jonas/synthetic_data/backgroundTissue'

if set == 'test':
    test_data_path = "/mnt/Drive3/jonas/synthetic_data/2023_3_9___17_25_2_TestSetOnDiffBrains400_600/Dataset/npz_data"
    bgTissueFolder = '/mnt/Drive3/jonas/synthetic_data/backgroundTissue400_800_secondtry'

if set == 'RealPat':
    endTest = 10
    bgTissueFolder = '/mnt/Drive3/jonas/mich_rec/mich_rec_128_maskedAndCutS3Tissue' #This is the old one...

if set == 'RealPatSRI':
    endTest = 10
    excludeCSFInSegmentation = False
    bgTissueFolder = '/mnt/Drive3/jonas/mich_rec/mich_rec_SRI_S3_maskedAndCut' # this is th 'original' way# this works on 11_04_2023

#### end params


outputFolder =  '/mnt/Drive3/jonas/LMITestDataJonas/' + modelState + '_Tend_' + str(T_global) + '/' + set + '/'

ys, yPreds, paths, flair_thrs, t1gd_thrs = evalModel(modelStatePath, test_data_path, endTest)


# %% simulate ...
def assertrange(Dw,rho,Tend):
    if not (Dw >= 0.0002 and Dw <= 0.015):
        print("Dw - LIGHT WARNING: parameter(s) out of generated range")
    if not (rho >= 0.002 and rho <= 0.2):
        print("rho - LIGHT WARNING: parameter(s) out of generated range")
    if not (Tend >= 50 and Tend <= 1500):
        print("Tend - LIGHT WARNING: parameter(s) out of generated range")

def convert(mu1, mu2, x, y, z):
    normalization_range = [-1.0, 1.0]
    T = T_global 

    mu1 = np.interp(mu1, normalization_range, [np.sqrt(0.01), np.sqrt(22.5)]) 
    mu2 = np.interp(mu2, normalization_range, [np.sqrt(0.1), np.sqrt(300)])  

    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = mu1**2 / T
    rho =  mu2**2 / T 

    return D, rho, T, x, y, z , mu1, mu2
#%%
for i in range(endTest):
    path = paths[i]

    pred = yPreds[i]
    predDw, predRho, predTend, predIcx, predIcy, predIcz, predMu1, predMu2 = convert(pred[0], pred[1], pred[2], pred[3], pred[4])

    predConverted = convert(pred[0], pred[1], pred[2], pred[3], pred[4])
    assertrange(predDw, predRho,predTend)

    if set == 'RealPat' or set == 'RealPatSRI':
        anatomyFolder = os.path.join(bgTissueFolder, path.split('/')[-1] + '_dat/')
    else: 
        gtConvertedAfterNetwork = convert(ys[i][0], ys[i][1], ys[i][2], ys[i][3], ys[i][4])
        #read parameter_tag.pkl with numpy load
        with open(path[0]+'/parameter_tag.pkl', 'rb') as f:
            groundTruthParams = pickle.load(f)

        anatomyFolder = os.path.join(bgTissueFolder, groundTruthParams['bgTissue']+'_dat/')

    #sweep over time to find the best fitting time step,
    # the idea is that this can be easily done as it does not take much more computational time
    if sweepTime:
        dumpFreq = int(predTend / 50 )

        command = "./brain -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
        predDw) + " -rho " + str(predRho) + " -Tend " + str(int(predTend * 1.2 )) + " -dumpfreq " + str(dumpFreq) + " -icx " + str(
        predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"

        print(f"command: {command}")

        parapid = i 

        #os.makedirs("./"+addOnPath+"vtus" + str(parapid) + "/sim/", exist_ok=True)
        vtuPath = os.path.join(outputFolder, 'multipleTimepoints', "vtus" + str(parapid) +'/')
        os.makedirs(vtuPath, exist_ok=True)
        npzPath = os.path.join(outputFolder, 'multipleTimepoints', "npzs" + str(parapid) +'/')
        os.makedirs(npzPath, exist_ok=True)

        simulation = subprocess.check_call([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/

        
    else: #this  is the normal case. Only the last time step is saved and converted to npz'''
        command = "./brain -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
        predDw) + " -rho " + str(predRho) + " -Tend " + str(predTend) + " -dumpfreq " + str(0.9999 * predTend) + " -icx " + str(
        predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"  # -bDumpIC 1

        print(f"command: {command}")

        parapid = i 

        #os.makedirs("./"+addOnPath+"vtus" + str(parapid) + "/sim/", exist_ok=True)
        vtuPath = os.path.join(outputFolder, "vtus" + str(parapid) +'/')
        os.makedirs(vtuPath, exist_ok=True)
        npzPath = os.path.join(outputFolder, "npzs" + str(parapid) +'/')
        os.makedirs(npzPath, exist_ok=True)

        simulation = subprocess.check_call([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/

    vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --vtk_path " + vtuPath + " --npz_path " + npzPath ], shell=True)
    
    shutil.rmtree(vtuPath)

    saveDict = {}

    saveDict['predOriginal'] = pred
    saveDict['predConverted'] = predConverted
    if not (set == 'RealPat' or set == 'RealPatSRI'):
        saveDict['groundTruthParams'] = groundTruthParams
        saveDict['gtConvertedAfterNetwork'] = gtConvertedAfterNetwork
        saveDict['flair_thr'] = flair_thrs[i]
        saveDict['t1gd_thr'] = t1gd_thrs[i]
        saveDict['gtdataPath']= path[0]


    np.save( os.path.join(npzPath, "allParams.npy"), saveDict)
