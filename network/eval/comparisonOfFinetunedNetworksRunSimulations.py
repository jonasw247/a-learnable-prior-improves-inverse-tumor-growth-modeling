
#%%
import numpy as np
import os
import time
import subprocess
import shutil
import multiprocessing
import pickle

def getBackgroundTissueDatPathTestsetDat(patientNumberTestsetLNMI):

    bgTissueFolder = '/mnt/Drive3/jonas/synthetic_data/backgroundTissue400_800_secondtry'

    gtFolder = '/mnt/Drive3/jonas/synthetic_data/2023_3_9___17_25_2_TestSetOnDiffBrains400_600/Dataset/npz_data/'

    #this is the final numbering order
    lnmiParams = np.load('/mnt/Drive3/jonas/LMITestDataJonas/25kSamplesDiffBGTissueAll_epoch49_Tend_100/test/npzs'+str(patientNumberTestsetLNMI)+'/allParams.npy', allow_pickle=True).item()

    npzGTID = lnmiParams['gtdataPath'].split('/')[-2]

    with open(gtFolder + str(npzGTID)+'/parameter_tag.pkl', 'rb') as f:
        gt_params = pickle.load(f)

    anatomyFolder = os.path.join(bgTissueFolder, gt_params['bgTissue']+'_dat/')

    return anatomyFolder

#%%
def simulationRunOnDatFiles(convertedParameters, outputFolder, anatomyFolder = './Atlas/anatomy_dat/', patientLabel = "noLabel", pathToSimulator = "./brain"):

    vtuPath = os.path.join(outputFolder, "vtus" + str(patientLabel) + '/')
    os.makedirs(vtuPath, exist_ok=True)
    npzPath = os.path.join(outputFolder, "npzs" + str(patientLabel) +'/')
    os.makedirs(npzPath, exist_ok=True)

    predDw, predRho, predTend, predIcx, predIcy, predIcz, predMu1, predMu2 = convertedParameters

    dumpFreq = 0.9999 * predTend

    command = pathToSimulator + " -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
    predDw) + " -rho " + str(predRho) + " -Tend " + str(int(predTend )) + " -dumpfreq " + str(dumpFreq) + " -icx " + str(
    predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"

    print('run ', command)

    start = time.time()
    simulation = subprocess.check_call([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/

    vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --vtk_path " + vtuPath + " --npz_path " + npzPath ], shell=True)
    
    shutil.rmtree(vtuPath)

    end = time.time()
    saveDict = {}

    saveDict['predConverted'] = convertedParameters
    saveDict['simtime'] = start-end
    saveDict['anatomyFolder'] = anatomyFolder

    np.save( os.path.join(npzPath, "allParams.npy"), saveDict)

#%% load network predictions
loadDir = np.load( './results/multipleFinetunedModels/180_eval_savedir.npy', allow_pickle=True).item()

allYPreds = np.array(loadDir['allYPreds'])

#%% run simulation 
numberOfProcesses = 40
pool = multiprocessing.Pool( numberOfProcesses)

outputFolderAll = '/mnt/Drive3/jonas/lmni_stuff/finetunedPredictionCorrectPaths/'
for patientID in range(len(allYPreds[0,:,0])):
    for networkID in range(len(allYPreds)):
        outputFolder = outputFolderAll + str(networkID) + '/'
    
        print('---------------------- run ', patientID)  

        params  = allYPreds[networkID,patientID,:]

        anatomyFolder = getBackgroundTissueDatPathTestsetDat(patientID)

        #simulationRunOnDatFiles(params, outputFolder,  anatomyFolder, patientLabel=str(patientID))
        pool.apply_async(simulationRunOnDatFiles, args=( params, outputFolder, anatomyFolder, patientID))

        time.sleep(1)

pool.close()
pool.join()
# %%
