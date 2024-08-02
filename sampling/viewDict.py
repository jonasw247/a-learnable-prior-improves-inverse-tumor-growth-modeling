#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats


def getDict(path):
    results = np.load(path + '/results.npy', allow_pickle=True).item()
    settings = np.load(path + '/settings.npy', allow_pickle=True).item()

    return results, settings

def plotValues(results, settings):
    import matplotlib.pyplot as plt
    plt.title('values - loss: ' +  settings["lossfunction"] 
              + ' - time: ' + str(round(results["time_min"]/60,1))+ 'h'
   )
    vals = ['x', 'y', 'z', 'D-cm-d', 'rho1-d']
    for i in range(len(vals)):
        plt.plot(results['nsamples'],np.array(results['xmeans']).T[i], label = vals[i])
    plt.yscale('log')
    plt.xlabel('# Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

def plotLoss(results, settings, log = True, p = 0.01):
    print('time', results["time_min"]   )
    plt.title(' loss: ' +  settings["lossfunction"] 
              + ' - time: ' + str(round(results["time_min"]/60,1))+ 'h')

    loglikelihood = np.mean(np.array(results['likelihoods']),axis=1)
    try:
        logPrior = np.mean(np.array(results['priors']),axis=1)
    except:
        print('no prior')
    
    if log:
        plt.plot(results['nsamples'],np.array(results['y0s']), label = 'logposterior')
        plt.plot(results['nsamples'], -(1-p) *loglikelihood, label = str(1-p) + ' *loglikelihood')
        try:
            plt.plot(results['nsamples'],-p * logPrior, label = str(p) + ' * logprior')
        except:
            pass
    else: 
        plt.plot(results['nsamples'],np.exp(-np.array(results['y0s'])), label = 'posterior')
        plt.plot(results['nsamples'],np.exp(loglikelihood), label = 'likelihood')

        try:
            plt.plot(results['nsamples'],np.exp(p*logPrior), label = str(p) + ' * prior')
        except:
            pass

    #plt.plot(results['nsamples'],(1- p)  * loglikelihood + p * logPrior, label = 'logposterior - loglikelihood')

    #plt.yscale('log')
    plt.xlabel('# Samples')
    plt.ylabel('Value')
    plt.legend()
    #plt.show()

def printValues(results, settings, params, mode = 'abs'):
    vals = ['x', 'y', 'z', 'D-cm-d', 'rho1-d']
    color =  ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i in range(len(settings['xMeasured'])):
        if mode == 'rel':
            res = (np.array(results['xmeans']).T[i] - settings['xMeasured'][i] ) /np.array(results['xmeans']).T[i]
            gtErr = (np.array(results['xmeans']).T[i] - params[vals[i]]['groundTruth'] ) /np.array(results['xmeans']).T[i]
            plt.title('relative error')
            plt.ylabel('relative error')
        else:
            res = (np.array(results['xmeans']).T[i] - settings['xMeasured'][i] )
            gtErr = (np.array(results['xmeans']).T[i] - params[vals[i]]['groundTruth'] )

            plt.title('absolute error')
            plt.ylabel('error')
        plt.plot(results['nsamples'], res, label = vals[i], color = color[i])
        plt.plot(results['nsamples'], gtErr, label = vals[i] + ' GT', color = color[i], linestyle = '--')

        plt.ymin = -1
   
    plt.xlabel('# Samples')
    plt.legend()
    plt.show()

def compareToGT():
    pass

if __name__ == '__main__':
    directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
    workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
    testDataset = Dataset_SyntheticBrats(directoryPath, workdir)


#%%
if __name__ == '__main__':

    paths = ["/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_0_dtime2023_10_12-13_02_26_gen_10_loss_dice_prior_True_dw0_0.008038116132598106_rho0_0.3065637420701083", "resultsSynData/patient_0_dtime2023_10_12-14_36_19_gen_10_loss_bernoulli_prior_True_dw0_0.008038116132598106_rho0_0.3065637420701083", "/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_0_dtime2023_10_12-15_59_53_gen_10_loss_bernoulli_prior_False_dw0_0.008038116132598106_rho0_0.3065637420701083", "/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_0_dtime2023_10_12-16_32_35_gen_10_loss_dice_prior_False_dw0_0.008038116132598106_rho0_0.3065637420701083"]

    paths = ["/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_0_dtime2023_10_12-21_13_04_gen_50_loss_dice_prior_True_dw0_0.008038116132598106_rho0_0.3065637420701083"]

    noInitPaths = ["/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_5_dtime2023_10_15-00_35_49_gen_15_loss_dice_prior_0.05_dw0_0.010987881705688506_rho0_0.2169042699123216" ]

    paths = ["/home/jonas/workspace/programs/monteCarloSolver/resultsSynData/patient_2_dtime2023_10_16-03_16_18_gen_75_loss_dice_prior_0.05_priorInit_True_dw0_0.002612429750423341_rho0_0.3090490486910405"]

    for path in paths:#pathes:pathRealPatient

        patient = int(path.split('patient_')[-1][0])
        loss = path.split('loss_')[-1].split('_')[0]
        addedPrior = path.split('prior_')[-1].split('_')[0]

        results, settings = getDict(path)
        plotLoss(results, settings)
        plt.show()
        plt.title('Patient ' + str(patient) + ' - Loss: ' + loss + ' - Prior: ' + addedPrior)
        plotValues(results, settings)
        plt.show()
        print('Final Dice T1', round(results['diceT1_67'],2))
        print('Final Dice FLAIR', round(results['diceFLAIR_25'], 2))
        #print('Final Loss', round(results['final_loss'], 2))

        printValues(results, settings, testDataset.getParams(0), 'abs')
        plt.title('Patient ' + str(patient) + ' - Loss: ' + loss + ' - Prior: ' + addedPrior)
        plt.show()
        patient = int(path.split('patient_')[-1][0])

        array = nib.load(path + '/result.nii.gz').get_fdata().astype(np.float32)
        modality = 'result_' + loss + '_' + addedPrior + '_' + str(results['nsamples'][-1])
        testDataset.addNewDataToDataset(patient, modality, 0, "dat128JanasSolver", array)

        #break
    #plotValues(results, settings)

    #%%
    import nibabel as nib
    from scipy import ndimage

    groundTruth = testDataset.loadPatientImageEnumarated(patient, "groundTruth", '0', "dat128JanasSolver").astype(np.float32)


    centerOfMass = int(ndimage.center_of_mass(groundTruth)[2])
    for path in paths:
        #load nifti
        patient = int(path.split('patient_')[-1][0])
        loss = path.split('loss_')[-1].split('_')[0]
        addedPrior = path.split('prior_')[-1].split('_')[0]

        plt.title('Patient ' + str(patient) + ' - Loss: ' + loss + ' - Prior: ' + addedPrior)

        proposal =  nib.load(path + '/result.nii.gz').get_fdata().astype(np.float32)


        plt.imshow((proposal-groundTruth)[:,:,centerOfMass], alpha = 0.5, cmap='bwr', vmin = -1, vmax = 1)
        totalErr = np.sum(np.abs(proposal-groundTruth))
        plt.show()

        print(path)
        print('totalErr', totalErr)

    #%%
    results.keys()
    results, settings = getDict()
    plotLoss(results, settings)
    plotValues(results, settings)

    # %%
    print(dict["loss_function"])
    plt.plot(dict['nsamples'], dict['y0s'])# %%
    plt.xlabel('nsamples')
    plt.ylabel('loss')
    dict.keys()
    # %%
    dict.keys()

    # %%

    # %%
