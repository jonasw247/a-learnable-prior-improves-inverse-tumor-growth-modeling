
#%%
import numpy as np
import os
import time
import datetime
import subprocess
import shutil
from vtutonpz2NoMultiprocessing import converter as VtuToNpz
import nibabel as nib

#%%
def simulationRunOnDatFiles(convertedParameters, outputFolder, anatomyFolder = '/home/home/jonas/programs/learn-morph-infer/torchcode/evalJonas/Atlas/anatomy_dat/', patientLabel = "noLabel", pathToSimulator = "./brain", doSave =False):
    #print time as string
    string = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + str(np.random.randint(0,1000000))

    vtuPath = os.path.join(outputFolder, "vtus" + str(patientLabel) + string + '/')
    os.makedirs(vtuPath, exist_ok=True)
    npzPath = os.path.join(outputFolder, "npzs" + str(patientLabel) + string+'/')
    os.makedirs(npzPath, exist_ok=True)

    predDw, predRho, predTend, predIcx, predIcy, predIcz, predMu1, predMu2 = convertedParameters

    dumpFreq = 0.9999 * predTend

    command = pathToSimulator + " -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
    predDw) + " -rho " + str(predRho) + " -Tend " + str(int(predTend )) + " -dumpfreq " + str(dumpFreq) + " -icx " + str(
    predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"


    print(" ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print('run ', command)

    start = time.time()
    simulation = subprocess.check_output([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<	")
    print(" ")

    converter = VtuToNpz(vtuPath, npzPath)
    array = converter.getArray()[:,:,:,0]

    shutil.rmtree(vtuPath)

    end = time.time()
    saveDict = {}

    saveDict['predConverted'] = convertedParameters
    saveDict['simtime'] = start-end
    saveDict['anatomyFolder'] = anatomyFolder

    #array = np.load( os.path.join(npzPath, "Data_0001.npz"))['data'][:,:,:,0]

    if doSave:
        np.save( os.path.join(npzPath, "allParams.npy"), saveDict)
    else:
        shutil.rmtree(npzPath)

    return array

def run(datPath, icx, icy, icz, dw, rho, Tend):
   
    allParams = [dw, rho, Tend, icx, icy, icz, 0, 0] 

    resultArray = simulationRunOnDatFiles(allParams, './tempOutput/', anatomyFolder = datPath, patientLabel = "noLabel", pathToSimulator = "/home/jonas/workspace/programs/monteCarloSolver/brain")
   
    return resultArray



#%%
if __name__ == "__main__":
    patientID=1


    datPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/mich_rec_SRI_S3_maskedAndCut/rec001_pre_dat/"

    paramsPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/parameterResultsCSV/"


    singleNetworkParams = []
    for key in ["x", "y", "z", "D-cm-d", "rho1-d"]:
        
        #read csv
        params = np.genfromtxt(paramsPath + key + ".csv", delimiter=',')
        data = np.loadtxt(paramsPath + key + ".csv", delimiter=',', skiprows=1)
        for patient in data:
            if int(patient[0]) == patientID:
                print(patient)
                singleNetworkParams.append(patient[1])
                #ensemble = patient[1]


    bpd = 16#16#6
    icx = singleNetworkParams[0]
    icy = singleNetworkParams[1]
    icz = singleNetworkParams[2]
    
    dw = singleNetworkParams[3]
    rho = singleNetworkParams[4]
    tend = 100

    lala = run(datPath, icx, icy, icz, dw, rho, tend)

    nib.save(nib.Nifti1Image(lala, np.eye(4)), "lala.nii.gz")
# %%
