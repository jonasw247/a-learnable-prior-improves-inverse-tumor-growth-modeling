
#%%
from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats
from cmaesSettingsWrapper import readNii, writeNii, CmaesSolver
import numpy as np
import os
import time


from scipy import ndimage

#%%
def extendTo256(a):
    return np.repeat(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1), 2, axis=2)

def runOnSimulatedData(patientID = 0):
    print("start")

    directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
    workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
    testDataset = Dataset_SyntheticBrats(directoryPath, workdir)
    
    gt128 = testDataset.loadPatientImageEnumarated(patientID, "groundTruth", '0', "dat128JanasSolver").astype(np.float32)

    """    params = testDataset.getParams(patientID)

    dw = params["D-cm-d"]['lnmi']
    rho = params["rho1-d"]['lnmi']
    icx, icy, icz = [params["x"]['lnmi'] , params["y"]['lnmi'], params["z"]['lnmi']]
    Tend = 100"""

    datPath = testDataset.getDatPath(patientID)
    
    GM = testDataset.loadPatientImageEnumarated(patientID, "GM", '0', "dat128JanasSolver").astype(np.float32)
    WM = testDataset.loadPatientImageEnumarated(patientID, "WM", '0', "dat128JanasSolver").astype(np.float32)

    T1c = testDataset.loadPatientImageEnumarated(patientID, "seg-t1c", '0', "dat128JanasSolver")
    FLAIR = testDataset.loadPatientImageEnumarated(patientID, "seg-flair", '0', "dat128JanasSolver")


    settings = {}
    # ranges from LMI paper with T = 100
    parameterRanges = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3]] 
    settings["parameterRanges"] = parameterRanges

    settings["datPath"] = datPath

    settings["bpd"] = 16
    settings["rho0"] = 0.001#0.1
    settings["dw0"] = 0.001#0.2
    settings["origin"] = np.divide(ndimage.center_of_mass(FLAIR), np.shape(FLAIR))
    settings["workers"] = 0
    settings["sigma0"] = 0.02#0.02 
    settings["generations"] = 75
    settings["lossfunction"] ="dice"
    settings["Tend"] = 100

    settings["priorInit"] = True#
    settings["addPrior"] = 0.5 

    # the factor to increase the range for the ensemble
    settings["factorSTD"] = 13.5 
    settings["diffLikelihood"] = True


    debugMode = False

    if debugMode:
        print('---------------------------------')
        print('--------- Debug Mode ------------')
        print('---------------------------------')
        settings["generations"] = 2
        settings["workers"] = 0
        settings["Tend"] = 1
        settings["priorInit"] = True
        settings["addPrior"] = 0.05


    params = testDataset.getParams(patientID)


    settings["stdMeasured"], settings["xMeasured"] =[], []
    for key in ["x", "y", "z", "D-cm-d", "rho1-d"]:
        print(key, params[key]['ensemble'])

        settings["xMeasured"].append(np.mean(params[key]['ensemble']))
        settings["stdMeasured"].append(np.std(params[key]['ensemble']) * settings["factorSTD"])

    if settings["priorInit"]:
        settings["rho0"] = settings["xMeasured"][4]
        settings["dw0"] = settings["xMeasured"][3]
        settings["origin"] = settings["xMeasured"][:3]

    solver = CmaesSolver(settings, None, None, FLAIR, T1c)
    resultTumor, resultDict = solver.run()

    # save results
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = "./resultsSynData/"+ 'patient_' + str(patientID) + '_dtime' + datetime +"_gen_"+ str(settings["generations"]) + "_loss_" + str(settings["lossfunction"]) + '_prior_' + str(settings["addPrior"]) + "_priorInit_"+ str(settings["priorInit"]) +"_dw0_" + str(settings["dw0"]) + "_rho0_" + str(settings["rho0"])+"/"
    os.makedirs(path, exist_ok=True)
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    writeNii(resultTumor, path = path+"result.nii.gz")
    
    if debugMode:
        modality = 'result_CMAES_v6differentLikelihoods_DEBUG'
    else:
        modality = 'result_CMAES_v6differentLikelihoods_loss-' + settings["lossfunction"] + '_prior-' + str(settings["addPrior"]).replace('.', '_') + '_nSamples-' + str(resultDict['nsamples'][-1]) + '_priorInit-' + str(settings["priorInit"]) + "_factorSTD-" + str(settings["factorSTD"]).replace('.', '_')

    testDataset.addNewDataToDataset(patientID, modality, 0, "dat128JanasSolver", resultTumor)
    testDataset.addDictToDataset(patientID, modality, 0, "dat128JanasSolver", {'results': resultDict, 'settings': settings})


if __name__ == '__main__':
    import sys


    # Access command-line arguments
    # sys.argv[0] is the script name, sys.argv[1:] are the arguments
    script_name = sys.argv[0]
    arguments = sys.argv[1:]

    if len(arguments) == 0:
        print("No patient specified: python runOnSimulatedData.py for patient 0")

        for i in [0]:#range(14):
            print(i)
            runOnSimulatedData(i)


    # Use the arguments as needed
    for arg in arguments:
        print("Argument:", arg)

        runOnSimulatedData(int(arg))
