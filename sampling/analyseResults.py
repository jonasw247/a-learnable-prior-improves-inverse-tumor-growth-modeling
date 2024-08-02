#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats
import viewDict as vd
from tool import calcLikelihood
from scipy import ndimage
from tools import pairedTTest, t_test_from_stats


directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
testDataset = Dataset_SyntheticBrats(directoryPath, workdir)

colorsList = ["tab:blue", "tab:red","tab:orange",  "tab:olive"]
colors = {"sampling": colorsList[0], "lnmi": colorsList[1], "ensemble": colorsList[2], "lmi": colorsList[3]}
thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# %%
dices, allresults, allpatients, allSettings, maes, mses, flairCOMs, t1COMs = [],[],[],[], [],[],[], []
for i in range(0,180):
    try: 
        print('')
    
        print(i, ' - ' , testDataset.getInternalPatientID(i))

        gt = testDataset.loadPatientImageEnumarated(i, "groundTruth", '0', "dat128JanasSolver")
        lnmi = testDataset.loadPatientImageEnumarated(i, "result_lnmi", '0', "dat128JanasSolver")
        lmi = testDataset.loadPatientImageEnumarated(i, "result_lmi_registered", '0', "dat128JanasSolver")

        ensemble = testDataset.loadPatientImageEnumarated(i, "result_ensemble", '0', "dat128JanasSolver")

        flair = testDataset.loadPatientImageEnumarated(i, "seg-flair", '0', "dat128JanasSolver")

        t1c = testDataset.loadPatientImageEnumarated(i, "seg-t1c", '0', "dat128JanasSolver")

        flairCOM = np.array(list(ndimage.center_of_mass(flair)))/129
        t1COM = np.array(list(ndimage.center_of_mass(t1c)))/129

        flairCOMs.append(flairCOM)
        t1COMs.append(t1COM)

        print("flair" , np.sum(flair * (gt > 0.25) )/ np.sum(flair))
        print("t1c" , np.sum(t1c * (gt > 0.5) )/ np.sum(t1c))

        modality = "result_CMAES_v5InitEnsembleOrig_loss-dice_prior-0_5_nSamples-600_priorInit-True_factorSTD-13_5"

        samplingResults = testDataset.loadPatientImageEnumarated(i, modality, '0', "dat128JanasSolver")

    except:      
        print('-----------   failed: ', i)
        continue
    
    diceSampling, diceLnmi, diceEnsemble, diceLmi = [],[],[],[]
    for th in thresholds:
        diceSampling.append(calcLikelihood.dice(gt > th, samplingResults> th))
        diceLnmi.append(calcLikelihood.dice(gt > th, lnmi> th))
        diceLmi.append(calcLikelihood.dice(gt > th, lmi> th))
        diceEnsemble.append(calcLikelihood.dice(gt > th, ensemble> th))

    maeSampling = np.mean(np.abs(gt - samplingResults))
    mseSampling = np.mean((gt - samplingResults)**2)
    maeLnmi = np.mean(np.abs(gt - lnmi))
    mseLnmi = np.mean((gt - lnmi)**2)
    maeEnsemble = np.mean(np.abs(gt - ensemble))
    mseEnsemble = np.mean((gt - ensemble)**2)
    maeLmi = np.mean(np.abs(gt - lmi))
    mseLmi = np.mean((gt - lmi)**2)
    

    print(diceSampling, diceLnmi ,diceEnsemble)
    dictAll = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")
    allresults.append(dictAll["results"])
    allSettings.append(dictAll["settings"])
    allpatients.append(i)
    
    dices.append([diceSampling, diceLnmi, diceEnsemble, diceLmi])
    maes.append([maeSampling, maeLnmi, maeEnsemble, maeLmi])
    mses.append([mseSampling, mseLnmi, mseEnsemble, mseLmi])


#%% 
vals = ['x', 'y', 'z', 'D-cm-d', 'rho1-d']
#j = 4
allEnsembleDiff, allLnmiDiffs, allSamplDiffs, allLmiDiffs = [],[],[], []
allEnsembleDiffAbsDiff, allLnmiDiffsAbsDiff, allSamplDiffsAbsDiff, allLmiDiffsAbsDiff = [],[],[], []

for j in range(5):
    ensembleDiff, lnmiDiffs, samplDiffs, lmiDiffs = [],[],[], []
    absEnsembleDiff, absLnmiDiffs, absSamplDiffs, absLmiDiffs = [],[],[], []

    for i in range(len(allpatients)):

        params = testDataset.getParams(allpatients[i])
        val = vals[j]

        ensembleDiff.append(params[val]['groundTruth'] - np.mean(params[val]['ensemble']))
        lnmiDiffs.append(params[val]['groundTruth'] - params[val]['lnmi'])
        lmiDiffs.append(params[val]['groundTruth'] - params[val]['lmi'])

        samplDiffs.append(params[val]['groundTruth'] - allresults[i]['xs0s'][-1][j])


    allEnsembleDiff.append(np.abs(ensembleDiff / params[val]['groundTruth']))
    allLnmiDiffs.append(np.abs(lnmiDiffs / params[val]['groundTruth']))
    allSamplDiffs.append(np.abs(samplDiffs / params[val]['groundTruth']))
    allLmiDiffs.append(np.abs(lmiDiffs / params[val]['groundTruth']))

plt.errorbar(vals, np.mean(np.abs(allSamplDiffs), axis=1), np.std(np.abs(allSamplDiffs), axis=1)/np.sqrt(len(allSamplDiffs[0])), label = "sampling mean", color = colors["sampling"])
plt.errorbar(vals, np.mean(np.abs(allLnmiDiffs), axis=1), np.std(np.abs(allLnmiDiffs), axis=1)/np.sqrt(len(allLnmiDiffs[0])), label = "lnmi mean", color = colors["lnmi"] )
plt.errorbar(vals, np.mean(np.abs(allLmiDiffs), axis=1), np.std(np.abs(allLmiDiffs), axis=1)/np.sqrt(len(allLmiDiffs[0])), label = "lmi mean", color = colors["lmi"])
plt.errorbar(vals, np.mean(np.abs(allEnsembleDiff), axis=1), np.std(np.abs(allEnsembleDiff), axis=1)/np.sqrt(len(allEnsembleDiff[0])), label = "ensemble mean", color = colors["ensemble"])


plt.scatter(vals, np.median(np.abs(allSamplDiffs), axis=1), label = "sampling median" , color = colors["sampling"])
plt.scatter(vals, np.median(np.abs(allLnmiDiffs), axis=1), label = "lnmi median", color = colors["lnmi"]) 
plt.scatter(vals, np.median(np.abs(allLmiDiffs), axis=1), label = "lmi median", color = colors["lmi"] )
plt.scatter(vals, np.median(np.abs(allEnsembleDiff), axis=1), label = "ensemble median", color = colors["ensemble"] )

plt.xlabel("parameter")
plt.ylabel("relative error")
plt.legend()

'''plt.plot(allpatients, ensembleDiff, label = "ensemble")
plt.plot(allpatients, lnmiDiffs, label = "lnmi")
plt.plot(allpatients, samplDiffs, label = "sampling")
plt.legend()'''

print("ensemble", np.median(np.abs(ensembleDiff)))
print("lnmi", np.median(np.abs(lnmiDiffs)))
print("sampling", np.median(np.abs(samplDiffs)))
#%%
#%% 
vals = ['x', 'y', 'z', 'mu1-cm', 'mu2-unitless']
#j = 4

allAbsEnsemble, allAbsLnmi, allAbsSampl, allAbsLmi, allAbsGT = [],[],[], [], []

for j in range(5):
    absEnsemble, absLnmi, absSampl, absLmi, absGT = [],[],[], [], []

    for i in range(len(allpatients)):

        params = testDataset.getParams(allpatients[i])
        val = vals[j]

        absGT.append(params[val]['groundTruth'])
        absEnsemble.append( np.mean(params[val]['ensemble']))
        absLnmi.append( params[val]['lnmi'])
        absLmi.append(params[val]['lmi'])
        
        if val in ['mu1-cm', 'mu2-unitless']:
            absSampl.append( np.sqrt(allresults[i]['xs0s'][-1][j] * 100))
            #print("mu1-cm", np.sqrt(allresults[i]['xs0s'][-1][j] * 100))
        else:
            absSampl.append( allresults[i]['xs0s'][-1][j])
        

    allAbsEnsemble.append(absEnsemble)
    allAbsLnmi.append(absLnmi)
    allAbsSampl.append(absSampl)
    allAbsLmi.append(absLmi)
    allAbsGT.append(absGT)

allAbsEnsemble = np.array(allAbsEnsemble)
allAbsLnmi = np.array(allAbsLnmi)
allAbsSampl = np.array(allAbsSampl)
allAbsLmi = np.array(allAbsLmi)
allAbsGT = np.array(allAbsGT)


plt.errorbar(vals, np.mean(np.abs(allAbsSampl), axis=1), np.std(np.abs(allAbsSampl), axis=1)/np.sqrt(len(allAbsSampl[0])), label = "sampling mean", color = colors["sampling"])
plt.errorbar(vals, np.mean(np.abs(allAbsLnmi), axis=1), np.std(np.abs(allAbsLnmi), axis=1)/np.sqrt(len(allAbsLnmi[0])), label = "lnmi mean", color = colors["lnmi"] )
plt.errorbar(vals, np.mean(np.abs(allAbsLmi), axis=1), np.std(np.abs(allAbsLmi), axis=1)/np.sqrt(len(allAbsLmi[0])), label = "lmi mean", color = colors["lmi"])
plt.errorbar(vals, np.mean(np.abs(allAbsEnsemble), axis=1), np.std(np.abs(allAbsEnsemble), axis=1)/np.sqrt(len(allAbsEnsemble[0])), label = "ensemble mean", color = colors["ensemble"])
plt.errorbar(vals, np.mean(np.abs(allAbsGT), axis=1), np.std(np.abs(allAbsGT), axis=1)/np.sqrt(len(allAbsGT[0])), label = "ground truth mean", color = "black")

plt.legend()

#%% Table 1 MD 
def getMD_and_MD_err(values1, values2):
    mean_diff = np.mean(values1 - values2)
    standardErr_mean_diff = np.std(values1 - values2) / np.sqrt(len(values1))
    return mean_diff, standardErr_mean_diff

print(" ")
#mu1 in unitless
samplingMD, samplingMD_err = getMD_and_MD_err(10 * allAbsSampl[3], 10 *allAbsGT[3])
print('MD: sampling mu1 mm' , round(samplingMD, 2), '+-', round(samplingMD_err, 2))
lnmiMD, lnmiMD_err = getMD_and_MD_err(10 * allAbsLnmi[3], 10 *allAbsGT[3])
print('MD: lnmi mu1 mm' , round(lnmiMD, 2), '+-', round(lnmiMD_err, 2))
ensembleMD, ensembleMD_err = getMD_and_MD_err(10 * allAbsEnsemble[3], 10 *allAbsGT[3])
print('MD: ensemble mu1 mm' , round(ensembleMD, 2), '+-', round(ensembleMD_err, 2))
lmiMD, lmiMD_err = getMD_and_MD_err(10 * allAbsLmi[3], 10 *allAbsGT[3])
print('MD: lmi mu1 mm' , round(lmiMD, 2), '+-', round(lmiMD_err, 2))

#mu2
print(" ")
samplingMD, samplingMD_err = getMD_and_MD_err(allAbsSampl[4], allAbsGT[4])
print('MD: sampling mu2 unitless' , round(samplingMD, 2), '+-', round(samplingMD_err, 2))
lnmiMD, lnmiMD_err = getMD_and_MD_err(allAbsLnmi[4], allAbsGT[4])
print('MD: lnmi mu2 unitless' , round(lnmiMD, 2), '+-', round(lnmiMD_err, 2))
ensembleMD, ensembleMD_err = getMD_and_MD_err(allAbsEnsemble[4], allAbsGT[4])
print('MD: ensemble mu2 unitless' , round(ensembleMD, 2), '+-', round(ensembleMD_err, 2))
lmiMD, lmiMD_err = getMD_and_MD_err(allAbsLmi[4], allAbsGT[4])
print('MD: lmi mu2 unitless' , round(lmiMD, 2), '+-', round(lmiMD_err, 2))

#origin
print(" ")
scalingFactor = 256
samplingDiff = scalingFactor* np.sqrt((allAbsSampl[0] - allAbsGT[0])**2+ (allAbsSampl[1] - allAbsGT[1])**2+ (allAbsSampl[2] - allAbsGT[2])**2)
samplingMD, samplingMD_err = getMD_and_MD_err(samplingDiff, 0)
print('MD: sampling origin mm' , round(samplingMD, 2), '+-', round(samplingMD_err, 2))
lnmiDiff = scalingFactor*np.sqrt((allAbsLnmi[0] - allAbsGT[0])**2+ (allAbsLnmi[1] - allAbsGT[1])**2+ (allAbsLnmi[2] - allAbsGT[2])**2)
lnmiMD, lnmiMD_err = getMD_and_MD_err(lnmiDiff, 0)
print('MD: lnmi origin mm' , round(lnmiMD, 2), '+-', round(lnmiMD_err, 2))
ensembleDiff = scalingFactor*np.sqrt((allAbsEnsemble[0] - allAbsGT[0])**2+ (allAbsEnsemble[1] - allAbsGT[1])**2+ (allAbsEnsemble[2] - allAbsGT[2])**2)
ensembleMD, ensembleMD_err = getMD_and_MD_err(ensembleDiff, 0)
print('MD: ensemble origin mm' , round(ensembleMD, 2), '+-', round(ensembleMD_err, 2))
lmiDiff = scalingFactor*np.sqrt((allAbsLmi[0] - allAbsGT[0])**2+ (allAbsLmi[1] - allAbsGT[1])**2+ (allAbsLmi[2] - allAbsGT[2])**2)
lmiMD, lmiMD_err = getMD_and_MD_err(lmiDiff, 0)
print('MD: lmi origin mm' , round(8.72, 2), '+-', round(lmiMD_err / lmiMD * 8.72, 2)) # lmi is not in atlas space

#COM 
print(" ")
flairCOMs = np.array(flairCOMs)
t1COMs = np.array(t1COMs)
diffFlairCOMAll = flairCOMs - allAbsGT[:3].T
diffFlairCOM = scalingFactor*np.sqrt(np.sum(diffFlairCOMAll**2, axis=1))

flairMD = getMD_and_MD_err(diffFlairCOM, 0)
print('MD: flair COM mm' , round(flairMD[0], 2), '+-', round(flairMD[1], 2))

diffT1COMAll = t1COMs - allAbsGT[:3].T
diffT1COM = scalingFactor*np.sqrt(np.sum(diffT1COMAll**2, axis=1))

# remove nones
diffT1COM = diffT1COM[~np.isnan(diffT1COM)]

t1MD = getMD_and_MD_err(diffT1COM, 0)
print('MD: t1 COM mm' , round(t1MD[0], 2), '+-', round(t1MD[1], 2))


#%% table 1 RMSD
#mu1 in mm
print('')
print('mu D')
def get_RMSD_and_RMSD_err(values1, values2):
    sq_diff = (values1 - values2)**2
    mean_sq_diff = np.mean(sq_diff)
    standardErr_mean_sq_diff = np.std(sq_diff) / np.sqrt(len(values1))
    root_mean_sq_diff = np.sqrt(mean_sq_diff)
    err_root_mean_sq_diff = (1/ (2 * root_mean_sq_diff)) * standardErr_mean_sq_diff
    return root_mean_sq_diff, err_root_mean_sq_diff

samplingRMSD, samplingRMSD_err = get_RMSD_and_RMSD_err(allAbsSampl[3], allAbsGT[3])
print('RMSD: sampling mu1 mm' , round(10 * samplingRMSD, 2), '+-', round(10 * samplingRMSD_err, 2))
lnmiRMSD, lnmiRMSD_err = get_RMSD_and_RMSD_err(allAbsLnmi[3], allAbsGT[3])
print('RMSD: lnmi mu1 mm' , round(10 * lnmiRMSD, 2), '+-', round(10 * lnmiRMSD_err, 2))
ensembleRMSD, ensembleRMSD_err = get_RMSD_and_RMSD_err(allAbsEnsemble[3], allAbsGT[3])
print('RMSD: ensemble mu1 mm' , round(10 * ensembleRMSD, 2), '+-', round(10 * ensembleRMSD_err, 2))
lmiRMSD, lmiRMSD_err = get_RMSD_and_RMSD_err(allAbsLmi[3], allAbsGT[3])
print('RMSD: lmi mu1 mm' , round(10 * lmiRMSD, 2), '+-', round(10 * lmiRMSD_err, 2))

print("t test comparing sampling and ensemble")
t_test_from_stats(samplingRMSD, samplingRMSD_err * np.sqrt(len(allAbsSampl[3])), len(allAbsSampl[3]), ensembleRMSD, ensembleRMSD_err * np.sqrt(len(allAbsEnsemble[3])), len(allAbsEnsemble[3]))
# mu2
print('')
print('mu rho')
samplingRMSD, samplingRMSD_err = get_RMSD_and_RMSD_err(allAbsSampl[4], allAbsGT[4])
print('RMSD: sampling mu2 unitless' , round(samplingRMSD, 2), '+-', round(samplingRMSD_err, 2))
lnmiRMSD, lnmiRMSD_err = get_RMSD_and_RMSD_err(allAbsLnmi[4], allAbsGT[4])
print('RMSD: lnmi mu2 unitless' , round(lnmiRMSD, 2), '+-', round(lnmiRMSD_err, 2))
ensembleRMSD, ensembleRMSD_err = get_RMSD_and_RMSD_err(allAbsEnsemble[4], allAbsGT[4])
print('RMSD: ensemble mu2 unitless' , round(ensembleRMSD, 2), '+-', round(ensembleRMSD_err, 2))
lmiRMSD, lmiRMSD_err = get_RMSD_and_RMSD_err(allAbsLmi[4], allAbsGT[4])
print('RMSD: lmi mu2 unitless' , round(lmiRMSD, 2), '+-', round(lmiRMSD_err, 2))

print("t test comparing sampling and ensemble")
t_test_from_stats(samplingRMSD, samplingRMSD_err * np.sqrt(len(allAbsSampl[4])), len(allAbsSampl[4]), ensembleRMSD, ensembleRMSD_err * np.sqrt(len(allAbsEnsemble[4])), len(allAbsEnsemble[4]))


#origin
print('')
print('origin')
samplingRMSD, samplingRMSD_err = get_RMSD_and_RMSD_err(samplingDiff, 0)
print('RMSD: sampling origin mm' , round(samplingRMSD, 2), '+-', round(samplingRMSD_err, 2))
lnmiRMSD, lnmiRMSD_err = get_RMSD_and_RMSD_err(lnmiDiff, 0)
print('RMSD: lnmi origin mm' , round(lnmiRMSD, 2), '+-', round(lnmiRMSD_err, 2))
ensembleRMSD, ensembleRMSD_err = get_RMSD_and_RMSD_err(ensembleDiff, 0)
print('RMSD: ensemble origin mm' , round(ensembleRMSD, 2), '+-', round(ensembleRMSD_err, 2))
lmiRMSD, lmiRMSD_err = get_RMSD_and_RMSD_err(lmiDiff, 0)
print('RMSD: lmi origin mm' , round(10.3, 2), '+-', round(lmiRMSD_err / lmiRMSD * 10.3, 2)) # lmi is not in atlas space

print("t test comparing sampling and ensemble")
t_test_from_stats(samplingRMSD, samplingRMSD_err * np.sqrt(len(samplingDiff)), len(samplingDiff), ensembleRMSD, ensembleRMSD_err * np.sqrt(len(ensembleDiff)), len(ensembleDiff))

#COM
print('')
print('COM flair')
com_flair_RMSD, com_flair_RMSD_err = get_RMSD_and_RMSD_err(diffFlairCOM, 0)
print('RMSD: sampling COM flair mm' , round(com_flair_RMSD, 2), '+-', round(com_flair_RMSD_err, 2))
com_t1_RMSD, com_t1_RMSD_err = get_RMSD_and_RMSD_err(diffT1COM, 0)
print('RMSD: sampling COM t1 mm' , round(com_t1_RMSD, 2), '+-', round(com_t1_RMSD_err, 2))

#ttest
print(" ")
print("COM t1 - ensemble")
t_test_from_stats(com_t1_RMSD, com_t1_RMSD_err * np.sqrt(len(diffT1COM)), len(diffT1COM), ensembleRMSD, ensembleRMSD_err * np.sqrt(len(ensembleDiff)), len(ensembleDiff))


# %%
def find_missing_numbers(inpatients):
    missing_numbers = []
    for i in range(180):
        if not i in inpatients:
            missing_numbers.append(i)
    return missing_numbers
plt.figure(figsize=(14,7))

dices = np.array(dices)

plt.plot( allpatients, dices[:,0,1], label = 'sampling', linewidth = 0, marker = 'o', color = colors["sampling"])
plt.plot(allpatients, dices[:,1,1], label = 'lnmi', linewidth = 0, marker = 'o', color = colors["lnmi"])
plt.plot(allpatients, dices[:,3,1], label = 'lmi', linewidth = 0, marker = 'o', color = colors["lmi"])
plt.plot(allpatients, dices[:,2,1], label = 'ensemble', linewidth = 0, marker = 'o', color = colors["ensemble"])
plt.xlabel("patients")
plt.legend()
print("dices below 0.7 ", np.array(allpatients)[ dices[:,0,1] < 0.7])
print("missing patients", find_missing_numbers(allpatients))
# %%
meanSampling = np.median(dices[:,0])
meanLnmi = np.median(dices[:,1])
meanEnsemble = np.median(dices[:,2])
thIDX = 3
#print(meanSampling, meanLnmi, meanEnsemble)
print("sampling   median:", round(np.median(dices[:,0,thIDX]), 4), " mean: ", round(np.mean(dices[:,0,thIDX]), 4), "+-", round(np.std(dices[:,0,thIDX]/np.sqrt(len(dices))), 4))

print("lnmi        median:", round(np.median(dices[:,1,thIDX]), 4), " mean: ", round(np.mean(dices[:,1,thIDX]), 4), "+-", round(np.std(dices[:,1,thIDX]/ np.sqrt(len(dices))), 4))

print("ensemble   median:", round(np.median(dices[:,2,thIDX]), 4), " mean: ", round(np.mean(dices[:,2,thIDX]), 4), "+-", round(np.std(dices[:,2,thIDX]/ np.sqrt(len(dices))), 4))

print("")
print("attention lmi is without registration into atlas space")
print("lmi median:", round(np.median(dices[:,3,thIDX]), 4), " mean: ", round(np.mean(dices[:,3,thIDX]), 4), "+-", round(np.std(dices[:,3,thIDX]/ np.sqrt(len(dices))), 4))

#plot mean dice 
plt.figure(figsize=(4,3))
plt.bar([0,1,2], [ np.mean(dices[:,1,thIDX]), np.mean(dices[:,2,thIDX]), np.mean(dices[:,0,thIDX])], yerr = [ np.std(dices[:,1,thIDX]/np.sqrt(len(dices))), np.std(dices[:,2,thIDX]/np.sqrt(len(dices))), np.std(dices[:,0,thIDX]/np.sqrt(len(dices)))], color = [ "tab:blue", "tab:red", "tab:purple"])
plt.xticks([0,1,2], [ "Ours", "Ensemble","Sampling"])
plt.ylabel("Mean Dice")
plt.ylim(0.8,1)
plt.savefig("figures/meanDice.pdf", bbox_inches='tight')


#plot median dice
plt.figure(figsize=(4,3))
plt.bar([0,1,2], [ np.median(dices[:,1,thIDX]), np.median(dices[:,2,thIDX]), np.median(dices[:,0,thIDX])], color = [ "tab:blue", "tab:red","tab:purple"])
plt.xticks([0,1,2], [ "Ours", "Ensemble","Sampling"])
plt.ylabel("Median Dice")
plt.ylim(0.9,1)
plt.savefig("figures/medianDice.pdf", bbox_inches='tight')


#%% barchart MAE - table 2
maes = np.array(maes)
plt.figure(figsize=(4,3))
plt.bar([0,1,2,3], [ np.mean(maes[:,0]), np.mean(maes[:,1]), np.mean(maes[:,2]), np.mean(maes[:,3])], yerr = [ np.std(maes[:,0]/np.sqrt(len(maes))), np.std(maes[:,1]/np.sqrt(len(maes))), np.std(maes[:,2]/np.sqrt(len(maes))), np.std(maes[:,3]/np.sqrt(len(maes)))], color = colorsList)
plt.xticks([0,1,2,3], [ "Sampling", "Ours", "Ensemble","LMI"])
plt.title("Mean Absolute Error")
plt.ylabel("MAE")
plt.savefig("figures/mae.pdf", bbox_inches='tight')

scalingPot = 4
 
print("sampling   mean: 1e", scalingPot ,10**scalingPot * np.mean(maes[:,0]), "+-",10**scalingPot* np.std(maes[:,0]/np.sqrt(len(maes))))
print("lnmi        mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,1]), "+-",10**scalingPot* np.std(maes[:,1]/np.sqrt(len(maes))))
print("ensemble   mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,2]), "+-",10**scalingPot* np.std(maes[:,2]/np.sqrt(len(maes))))
print("lmi        mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,3]), "+-",10**scalingPot* np.std(maes[:,3]/np.sqrt(len(maes))))

# p values
print("t test comparing sampling and DL")
pairedTTest(maes[:,0], maes[:,1])

#%% barchart MSE - table 2
mses = np.array(mses)
plt.figure(figsize=(4,3))
plt.bar([0,1,2,3], [ np.mean(mses[:,0]), np.mean(mses[:,1]), np.mean(mses[:,2]), np.mean(mses[:,3])], yerr = [ np.std(mses[:,0]/np.sqrt(len(mses))), np.std(mses[:,1]/np.sqrt(len(mses))), np.std(mses[:,2]/np.sqrt(len(mses))), np.std(mses[:,3]/np.sqrt(len(mses)))], color = colorsList)
plt.xticks([0,1,2,3], [ "Sampling", "Ours", "Ensemble","LMI"])
plt.title("Mean Squared Error")
plt.ylabel("MSE")
plt.savefig("figures/mse.pdf", bbox_inches='tight')


scalingPot =  5
print("sampling   mean: 1e",scalingPot , 10**scalingPot *np.mean(mses[:,0]), "+-", 10**scalingPot *np.std(mses[:,0]/np.sqrt(len(mses))))
print("lnmi        mean: 1e",scalingPot , 10**scalingPot *np.mean(mses[:,1]), "+-", 10**scalingPot *np.std(mses[:,1]/np.sqrt(len(mses))))
print("ensemble   mean: 1e",scalingPot , 10**scalingPot *np.mean(mses[:,2]), "+-", 10**scalingPot *np.std(mses[:,2]/np.sqrt(len(mses))))
print("lmi        mean: 1e",scalingPot , 10**scalingPot *np.mean(mses[:,3]), "+-", 10**scalingPot *np.std(mses[:,3]/np.sqrt(len(mses))))

print("t test comparing sampling and DL")
pairedTTest(mses[:,0], mses[:,1])

#%% dice line plot mean

import matplotlib.pyplot as plt
import numpy as np

# Assuming thresholds, dices, and colors are defined

plt.figure(figsize=(5,4))
labels = ["DL-Prior + Sampling", "DL", "DL Ensemble", "LMI"]

for i, label in enumerate(labels):
    mean_values = np.mean(dices[:,i,:], axis=0)
    std_values = np.std(dices[:,i,:]/ np.sqrt(len(dices)), axis=0)
    plt.plot(thresholds, mean_values, label=label, color=colorsList[i], marker='.')
    plt.fill_between(thresholds, mean_values-std_values, mean_values+std_values, color=colorsList[i], alpha=0.2)

    print("Dice at 0.5 " , label, mean_values[5], "+-", std_values[5])

#FOR Table 2

plt.xlabel("Tumor Concentration Threshold")
plt.ylabel("Dice")
plt.legend()
plt.savefig("figures/diceLinePlot.pdf", bbox_inches='tight')

# t - test
pairedTTest(dices[:,0,5], dices[:,2,5])

#%% dice line plot median
plt.figure(figsize=(5,4))
plt.plot(thresholds, np.median(dices[:,0,:], axis=0), label = 'Sampling', color = colors["sampling"])
plt.plot(thresholds, np.median(dices[:,1,:], axis=0), label = 'Ours', color = colors["lnmi"])
plt.plot(thresholds, np.median(dices[:,2,:], axis=0), label = 'Ensemble', color = colors["ensemble"])
plt.plot(thresholds, np.median(dices[:,3,:], axis=0), label = 'LMI', color = colors["lmi"])

plt.xlabel("Tumor Concentration Threshold")
plt.ylabel("Dice")
plt.legend()
plt.savefig("figures/diceLinePlotMedian.pdf", bbox_inches='tight')
# %%
plt.figure(figsize=(10,7))
plt.title("Posterior")
for i in range(len(allresults)):
    if i >10:
        maker = 'o'
    else:
        maker = 'x'

    plt.plot(allresults[i]['nsamples'], allresults[i]['y0s'], label = str(allpatients[i]) + "  dice: " + str(np.round(dices[i,0],2)) + " time "+ str(np.round(allresults[i]['time_min']/60,2)) + "h", marker = maker)

#plt.ylim(-2,1)
plt.xlabel("samples")
plt.legend()
# %%
plt.figure(figsize=(10,7))
plt.title("likelihoods")
for i in range(len(allresults)):
    p = allSettings[i]["addPrior"]

    plt.plot(allresults[i]['nsamples'], -(1-p)*np.mean(allresults[i]['likelihoods'], axis=1), label = str(allpatients[i]) + "  dice: " + str(np.round(dices[i,0],2)))
plt.xlabel("patients")
plt.xlabel("samples")
plt.legend()
plt.ylim(-1,4)

#plt.ylim(-2,1)

# %%
range = [68] 
plt.figure(figsize=(10,7))

plt.title("priors")
for i in np.arange(len(dices[:,0]))[ range]:
    p = allSettings[i]["addPrior"]

    plt.plot(allresults[i]['nsamples'], - p* np.mean(allresults[i]['priors'], axis=1), label = str(allpatients[i]) + "  dice: " + str(np.round(dices[i,0],2)))
plt.xlabel("samples")
plt.legend()
#%%
def moving_average(data, window_size):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

plt.figure(figsize=(10,7))
plt.title("dice likelihoods")
for i in np.arange(len(dices[:,0]))[ range] :
    p = allSettings[i]["addPrior"]

    values = np.exp(np.max(allresults[i]['likelihoods'], axis=1))
    #
    window = 6
    plt.plot( moving_average(allresults[i]['nsamples'], window), moving_average(values, window), label = str(allpatients[i])  + "  dice: " + str(np.round(dices[i,0],2))+ " time "+ str(np.round(allresults[i]['time_min']/60,2)) + "h", linewidth = 0.8)

plt.xlabel("patients")
plt.xlabel("samples")
plt.legend()
#%%
plt.figure(figsize=(10,7))

plt.title("sigmas")
for i in  np.arange(len(dices[:,0]))[range] :

    plt.plot(allresults[i]['nsamples'], allresults[i]['sigmas'], label = str(allpatients[i]) + "  dice: " + str(np.round(dices[i,0],2))+ " time "+ str(np.round(allresults[i]['time_min']/60,2)) + "h")
plt.xlabel("samples")

plt.legend()
#%%
allresults[0].keys()

# %%
allSettings[0].keys()
# %% this is a test of different starting sigmas....
i = 2
gt = testDataset.loadPatientImageEnumarated(i, "result_groundTruth", '0', "dat128JanasSolver")
lnmi = testDataset.loadPatientImageEnumarated(i, "result_lnmi", '0', "dat128JanasSolver")
modality = "result_CMAES_v3smallSigma_loss-dice_prior-0_05_nSamples-400_priorInit-True"
samplingDictv3 = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")["results"]
modality = "result_CMAES_v2_loss-dice_prior-0_05_nSamples-600_priorInit-True"
samplingDict = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")["results"]
modality = "result_CMAES_v4verysmallSigma_loss-dice_prior-0_05_nSamples-600_priorInit-True"
samplingDictsupermall = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")["results"]

plt.title("likelihoods")
plt.plot(samplingDict["nsamples"], samplingDict["y0s"], label = "normal sigma")
plt.plot(samplingDictv3["nsamples"], samplingDictv3["y0s"], label = "small sigma")
plt.plot(samplingDictsupermall["nsamples"], samplingDictsupermall["y0s"], label = "super small sigma")

plt.xlabel("samples")
#plt.ylim(-2,8)
plt.legend()
plt.figure()
plt.title("sigmas")
plt.plot(samplingDict["nsamples"], samplingDict["sigmas"], label = "normal sigma")
plt.plot(samplingDictv3["nsamples"], samplingDictv3["sigmas"], label = "small sigma")
plt.plot(samplingDictsupermall["nsamples"], samplingDictsupermall["sigmas"], label = "super small sigma")
plt.xlabel("samples")
plt.legend()
# %% #####################################
#single patient

i = 13
gt = testDataset.loadPatientImageEnumarated(i, "result_groundTruth", '0', "dat128JanasSolver")
lnmi = testDataset.loadPatientImageEnumarated(i, "result_lnmi", '0', "dat128JanasSolver")
modality   = "result_CMAES_v2_loss-dice_prior-0_0_nSamples-600_priorInit-False"
samplingDict7 = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")

vd.plotValues(samplingDict7["results"], samplingDict7["settings"])
vd.plotLoss(samplingDict7["results"], samplingDict7["settings"], p = 0.05, log = True)
#plt.ylim(-2,10)
plt.show()
vd.printValues(samplingDict7["results"], samplingDict7["settings"], testDataset.getParams(i) , mode = 'abs')

plt.show()
#%%
startingPoint = np.array(samplingDict7["results"]["xmeans"]).T[:3].T
csf = testDataset.loadPatientImageEnumarated(i, "CSF", '0', "dat128JanasSolver")
wm = testDataset.loadPatientImageEnumarated(i, "WM", '0', "dat128JanasSolver")
gm = testDataset.loadPatientImageEnumarated(i, "GM", '0', "dat128JanasSolver")
point = (startingPoint[1] * 128 ).astype(int)
ensemble = testDataset.loadPatientImageEnumarated(i, "result_ensemble", '0', "dat128JanasSolver")

priorPoint = np.array(samplingDict7["settings"]["xMeasured"])[:3]
priorPoint = (priorPoint * 128 ).astype(int)

plt.imshow(np.mean((wm+gm), axis = 2), cmap = "gray")
plt.imshow(ensemble[:,:,point[2]], alpha = 0.5, cmap = "Reds")
loss = samplingDict7["results"]["y0s"]

for i in range(len(startingPoint)):
    point = (startingPoint[i] * 128 ).astype(int)
    cmap = plt.get_cmap('Reds')
    color = cmap((loss/np.max(loss))[i])
    plt.scatter(point[1], point[0], s = 10,  c = color, cmap=cmap)
#
plt.scatter(priorPoint[1], priorPoint[0], s = 10,  c = "blue")
# %%
plt.plot(loss/np.max(loss))
#%%
plt.plot(startingPoint)
#%%
plt.imshow(np.array(samplingDict7["results"]["C1s"])[30,:,:])
plt.colorbar()
#%%
plt.plot(samplingDict7["results"]["pss"])
# %%
plt.title("sigmas")
plt.xlabel("samples")
plt.plot( samplingDict7["results"]["sigmas"], label = "normal sigma")
#plt.ylim(-2,1)
plt.legend()

# %%
testDataset.getInternalPatientID(72)
# %%
t1Seg = testDataset.loadPatientImageEnumarated(14, "seg-t1c", '0', "dat128JanasSolver")
# %%
plt.imshow(t1Seg[:,:,50])
# %%

# %% valuesFor samples
modalities = [  "result_CMAES_v2_loss-dice_prior-0_0_nSamples-600_priorInit-False", "result_CMAES_v5InitEnsembleOrig_loss-dice_prior-0_05_nSamples-600_priorInit-True"]
value = "likelihoods"

for i in range(len(modalities)):
    
    for patient in range(11):
        dict = testDataset.loadResultsDictEnumerated(patient, modalities[i], '0', "dat128JanasSolver")
        resutls = dict["results"]
        settings = dict["settings"]

        likelihood = np.array(resutls[value])
        samples = np.array(resutls["nsamples"])

        plt.plot(resutls["nsamples"], -likelihood, label = str(patient) + " " + modalities[i])
