#%%

"""
This to rerun all experiment with the old solver and save the results in the dataset as nifit files
"""
import numpy as np
from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats
from runJanasForwardSolver import run as  runJanasForwardSolver


def runEnsemble(patientID, fromParams = "ensemble"):
    directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
    workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
    testDataset = Dataset_SyntheticBrats(directoryPath, workdir, deleteAndReloadWorkingDir=False)
    
    datPath = testDataset.getDatPath(patientID)

    params = testDataset.getParams(patientID)

    dw = np.mean(params["D-cm-d"][fromParams])
    rho = np.mean(params["rho1-d"][fromParams])
    icx, icy, icz = [np.mean(params["x"][fromParams]) , np.mean(params["y"][fromParams]),np.mean( params["z"][fromParams])]
    Tend = 100

    datPath = testDataset.getDatPath(patientID)
    print(datPath)

    tumor = runJanasForwardSolver(datPath, icx, icy, icz, dw, rho, Tend)

    modality = "result_" + fromParams

    testDataset.addNewDataToDataset(patientID, modality, 0, "dat128JanasSolver", tumor)


def runOneOfEnsemble(patientID, ensembleNumber):
    fromParams = "ensemble"
    directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
    workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
    testDataset = Dataset_SyntheticBrats(directoryPath, workdir, deleteAndReloadWorkingDir=False)
    
    datPath = testDataset.getDatPath(patientID)

    params = testDataset.getParams(patientID)

    dw = params["D-cm-d"][fromParams][ensembleNumber]
    rho = params["rho1-d"][fromParams][ensembleNumber]
    icx, icy, icz = [params["x"][fromParams][ensembleNumber] , params["y"][fromParams][ensembleNumber], params["z"][fromParams][ensembleNumber]]
    Tend = 100

    datPath = testDataset.getDatPath(patientID)
    print(datPath)

    tumor = runJanasForwardSolver(datPath, icx, icy, icz, dw, rho, Tend)

    modality = "result_" + fromParams + "_" + str(ensembleNumber)	

    testDataset.addNewDataToDataset(patientID, modality, 0, "dat128JanasSolver", tumor)


#%% run all ensembles for 
if False:#__name__ == "__main__":

    for i in range(0, 180):
        for j in range(10):
            runOneOfEnsemble(i, j)

#%%
if  __name__ == "__main__":

    #runEnsemble(4, fromParams = "groundTruth")
    for i in [160, 161]:# range(180):
        print('patient', i ) 
        #runEnsemble(i,  fromParams = "groundTruth")
        runEnsemble(i,  fromParams = "lnmi")
        runEnsemble(i,  fromParams = "lmi")
        runEnsemble(i,  fromParams = "ensemble")
 