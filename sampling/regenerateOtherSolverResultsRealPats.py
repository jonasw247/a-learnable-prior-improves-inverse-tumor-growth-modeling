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

def runrealPatSingle(patientID, fromParams = "ensemble"):
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


#%%
if  __name__ == "__main__":

    for i in range(180):
        print('patient', i ) 
        runEnsemble(i,  fromParams = "groundTruth")
        runEnsemble(i,  fromParams = "lnmi")
        runEnsemble(i,  fromParams = "lmi")
        runEnsemble(i,  fromParams = "ensemble")
    