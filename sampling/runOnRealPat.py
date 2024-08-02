
#%%
from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats
from cmaesSettingsWrapper import readNii, writeNii, CmaesSolver
import numpy as np
import os
import time
import nibabel as nib


from scipy import ndimage

#%%
def runOnrealPat(patientID = 0):
    print("start")

    directoryPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/"
    workdirPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/"
    
    patStr = str(patientID).zfill(3)

    tumorSegmentationPath = directoryPath + "rescaled_128_PatientData_SRI/rec" + patStr + "_pre/"
    tissuePath = directoryPath + "mich_rec_SRI_S3_maskedAndCut_rescaled_128/rec" + patStr + "_pre/"
    datPath = directoryPath + "mich_rec_SRI_S3_maskedAndCut/rec" + patStr + "_pre_dat/"
    print(datPath)
    
    GM = nib.load(tissuePath + "GM_flippedCorrectly.nii").get_fdata().astype(np.float32)
    
    WM = nib.load(tissuePath + "WM_flippedCorrectly.nii").get_fdata().astype(np.float32)

    T1c = nib.load(tumorSegmentationPath + "tumorCore_flippedCorrectly.nii").get_fdata()

    FLAIR = nib.load(tumorSegmentationPath + "tumorFlair_flippedCorrectly.nii").get_fdata()

    settings = {}
    # ranges from LMI paper with T = 100
    parameterRanges = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3]] 
    settings["parameterRanges"] = parameterRanges

    settings["datPath"] = datPath

    settings["bpd"] = 16
    settings["rho0"] = 0.001#0.1
    settings["dw0"] = 0.001#0.2
    settings["origin"] = np.divide(ndimage.center_of_mass(FLAIR), np.shape(FLAIR))
    settings["workers"] = 0#0
    settings["sigma0"] = 0.02#0.02 
    settings["generations"] = 75#10 #75 #75#250#75 
    settings["lossfunction"] ="dice"#"bernoulli"#### "dice"#
    settings["Tend"] = 100

  
    settings["priorInit"] = True# True#True#True#True 
    settings["addPrior"] = 0.5 # 0.5 #0.5#0.05#0.05 
    # the factor to increase the range for the ensemble
    settings["factorSTD"] = 13.5 #25# originial  with addPrior = 0.5: 13.5# addprior = 0.05 => factor std = 3
    settings["diffLikelihood"] = False


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

  
    # get Params from csv
    paramsPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/parameterResultsCSV/"

    settings["stdMeasured"], settings["xMeasured"] =[], []
    for key in ["x", "y", "z", "D-cm-d", "rho1-d"]:
        
        #read csv
        params = np.genfromtxt(paramsPath + key + ".csv", delimiter=',')
        data = np.loadtxt(paramsPath + key + ".csv", delimiter=',', skiprows=1)
        for patient in data:
            if int(patient[0]) == patientID:
                print(patient)
                ensemble = patient[2:]

        settings["xMeasured"].append(np.mean(ensemble))
        settings["stdMeasured"].append(np.std(ensemble) * settings["factorSTD"])

    #if settings["addPrior"]:
    if settings["priorInit"]:
        settings["rho0"] = settings["xMeasured"][4]
        settings["dw0"] = settings["xMeasured"][3]
        settings["origin"] = settings["xMeasured"][:3]

    solver = CmaesSolver(settings, None, None, FLAIR, T1c)
    #solver = CmaesSolver(settings, WM, GM, FLAIR, T1c)
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

    # save all files   
    path = workdirPath + modality + "/rec" + patStr + "_pre/"
    os.makedirs( path, exist_ok=True)

    #save dict
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    #save nii
    writeNii(resultTumor, path = path+"result.nii.gz")
    nib.save(nib.Nifti1Image(resultTumor, np.eye(4)), path + "resultTumor.nii.gz")


if __name__ == "__main__":
    for i in range(100):

        try:
            runOnrealPat(i)
        except:
            print("not found", i)





# %%
