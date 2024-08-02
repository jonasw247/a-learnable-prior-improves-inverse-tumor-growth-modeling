#%%
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
#calcLikelihood
from tool import calcLikelihood

#%%
mcmcResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jana_SRI/"

lmiResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_ivan_SRI/"

lnmiOriginal = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jonas_SRI/"

cmaesPriorResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_5_nSamples-600_priorInit-False_factorSTD-13_5/"

cmaesNaiveResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_0_nSamples-600_priorInit-False_factorSTD-13_5/" 
tumorSegmentationPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_PatientData_SRI/"

tissuePath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/mich_rec_SRI_S3_maskedAndCut_rescaled_128/"
ensemblePath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/ensembleResults/"


i = 1
patientName = "rec" + ("0000" + str(i))[-3:] + "_pre"

segmentation = nib.load(os.path.join(tumorSegmentationPath, patientName, "tumorFlair_flippedCorrectly.nii")).get_fdata()

mcmcResult = nib.load(os.path.join(mcmcResultPath, patientName, "MAP_flippedCorrectly.nii")).get_fdata()
lmiResult = nib.load(os.path.join(lmiResultPath, patientName, "inferred_tumor_patientspace_mri_flippedCorrectly.nii")).get_fdata()
lnmiResult = nib.load(os.path.join(lnmiOriginal, patientName, "predictionJonas_flippedCorrectly.nii")).get_fdata()
cmaesPrior = nib.load(os.path.join(cmaesPriorResults, patientName, "result.nii.gz")).get_fdata()
cmaesNaive = nib.load(os.path.join(cmaesNaiveResults, patientName, "result.nii.gz")).get_fdata()
wm = nib.load(os.path.join(tissuePath, patientName, "WM_flippedCorrectly.nii")).get_fdata()
gm = nib.load(os.path.join(tissuePath, patientName, "GM_flippedCorrectly.nii")).get_fdata()
csf = nib.load(os.path.join(tissuePath, patientName, "CSF_flippedCorrectly.nii")).get_fdata()
ensemble = nib.load(os.path.join(ensemblePath, patientName, "result.nii.gz")).get_fdata()


# %%

zSlice = int(ndimage.center_of_mass(segmentation)[2])

# %%
plt.imshow(segmentation[:,:,zSlice])
# %%
plt.imshow(mcmcResult[:,:,zSlice])
#%%
plt.imshow(lmiResult[:,:,zSlice])
#%%
plt.imshow(lnmiResult[:,:,zSlice])
#%%
plt.imshow(cmaesPrior[:,:,zSlice])
# %%
plt.imshow(cmaesNaive[:,:,zSlice])

# %%
plt.imshow(wm[:,:,zSlice])
# %%

colorsList = ["tab:blue", "tab:red", "tab:orange",  "tab:olive", "tab:purple", "tab:grey"]
colors = {"sampling": colorsList[0], "lnmi": colorsList[1], "ensemble": colorsList[2], "lmi": colorsList[3], "naivecmaes": colorsList[4], "MCMC": colorsList[5]}
thresholds = [ 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9]

# %%
dices, allresults, allpatients, allSettings, maes, mses = [],[],[],[], [], []
dicesFlair, dicesT1c = [], []
for i in range(0,100):
    if i == 30:
        continue
    try: 
        print('')
    
        patientName = "rec" + ("0000" + str(i))[-3:] + "_pre"

        segmentation = nib.load(os.path.join(tumorSegmentationPath, patientName, "tumorFlair_flippedCorrectly.nii")).get_fdata()

        mcmcResult = nib.load(os.path.join(mcmcResultPath, patientName, "MAP_flippedCorrectly.nii")).get_fdata()
        lmi = nib.load(os.path.join(lmiResultPath, patientName, "inferred_tumor_patientspace_mri_flippedCorrectly.nii")).get_fdata()
        lnmi = nib.load(os.path.join(lnmiOriginal, patientName, "predictionJonas_flippedCorrectly.nii")).get_fdata()
        cmaesPrior = nib.load(os.path.join(cmaesPriorResults, patientName, "result.nii.gz")).get_fdata()
        cmaesNaive = nib.load(os.path.join(cmaesNaiveResults, patientName, "result.nii.gz")).get_fdata()
        wm = nib.load(os.path.join(tissuePath, patientName, "WM_flippedCorrectly.nii")).get_fdata()
        gm = nib.load(os.path.join(tissuePath, patientName, "GM_flippedCorrectly.nii")).get_fdata()
        csf = nib.load(os.path.join(tissuePath, patientName, "CSF_flippedCorrectly.nii")).get_fdata()
        ensemble = nib.load(os.path.join(ensemblePath, patientName, "result.nii.gz")).get_fdata()


        segmentation = nib.load(os.path.join(tumorSegmentationPath, patientName, "tumorFlair_flippedCorrectly.nii")).get_fdata()

        flair = np.zeros_like(segmentation)
        t1c = np.zeros_like(segmentation)

        flair[segmentation > 0.1] = 1
        t1c[segmentation > 0.5] = 1

        # mask CSF region
        csfMask = csf > wm + gm
        mcmcResult[csfMask] = 0
        lmi[csfMask] = 0
        lnmi[csfMask] = 0
        cmaesPrior[csfMask] = 0
        cmaesNaive[csfMask] = 0
        ensemble[csfMask] = 0
        flair[csfMask] = 0
        t1c[csfMask] = 0
              

    except:      
        print('-----------   failed: ', i)
        continue
    gt = mcmcResult

    diceCmaesPrior, diceCmaesNaive , diceLnmi, diceEnsemble, diceLmi, diceMCMC = [],[],[],[], [] , []

    diceFlairCmaesPrior, diceFlairCmaesNaive , diceFlairLnmi, diceFlairEnsemble, diceFlairLmi, diceFlairMCMC = [],[],[],[], [], []
    diceT1cCmaesPrior, diceT1cCmaesNaive , diceT1cLnmi, diceT1cEnsemble, diceT1cLmi, diceT1cMCMC = [],[],[],[], [], []
    for th in thresholds:
        diceCmaesPrior.append(calcLikelihood.dice(gt > th, cmaesPrior> th))
        diceCmaesNaive.append(calcLikelihood.dice(gt > th, cmaesNaive> th))
        diceLnmi.append(calcLikelihood.dice(gt > th, lnmi> th))
        diceLmi.append(calcLikelihood.dice(gt > th, lmi> th))
        diceEnsemble.append(calcLikelihood.dice(gt > th, ensemble> th))
        diceMCMC.append(calcLikelihood.dice(gt > th, mcmcResult> th))


        diceFlairCmaesPrior.append(calcLikelihood.dice(flair > th, cmaesPrior> th))
        diceFlairCmaesNaive.append(calcLikelihood.dice(flair > th, cmaesNaive> th))
        diceFlairLnmi.append(calcLikelihood.dice(flair > th, lnmi> th))
        diceFlairLmi.append(calcLikelihood.dice(flair > th, lmi> th))
        diceFlairEnsemble.append(calcLikelihood.dice(flair > th, ensemble> th))
        diceFlairMCMC.append(calcLikelihood.dice(flair > th, mcmcResult> th))

        diceT1cCmaesPrior.append(calcLikelihood.dice(t1c > th, cmaesPrior> th))
        diceT1cCmaesNaive.append(calcLikelihood.dice(t1c > th, cmaesNaive> th))
        diceT1cLnmi.append(calcLikelihood.dice(t1c > th, lnmi> th))
        diceT1cLmi.append(calcLikelihood.dice(t1c > th, lmi> th))
        diceT1cEnsemble.append(calcLikelihood.dice(t1c > th, ensemble> th))
        diceT1cMCMC.append(calcLikelihood.dice(t1c > th, mcmcResult> th))

    maeCmaesPrior = np.mean(np.abs(gt - cmaesPrior))
    mseCmaesPrior = np.mean((gt - cmaesPrior)**2)
    maeCmaesNaive = np.mean(np.abs(gt - cmaesNaive))
    mseCmaesNaive = np.mean((gt - cmaesNaive)**2)
    maeLnmi = np.mean(np.abs(gt - lnmi))
    mseLnmi = np.mean((gt - lnmi)**2)
    maeEnsemble = np.mean(np.abs(gt - ensemble))
    mseEnsemble = np.mean((gt - ensemble)**2)
    maeLmi = np.mean(np.abs(gt - lmi))
    mseLmi = np.mean((gt - lmi)**2)
    

    print(diceCmaesPrior, diceLnmi ,diceEnsemble)

    
    dices.append([diceCmaesPrior,  diceLnmi, diceEnsemble, diceLmi, diceCmaesNaive, diceMCMC ])
    dicesFlair.append([diceFlairCmaesPrior,  diceFlairLnmi, diceFlairEnsemble, diceFlairLmi, diceFlairCmaesNaive, diceFlairMCMC])
    dicesT1c.append([diceT1cCmaesPrior,  diceT1cLnmi, diceT1cEnsemble, diceT1cLmi, diceT1cCmaesNaive, diceT1cMCMC])
    maes.append([maeCmaesPrior, maeLnmi, maeEnsemble, maeLmi, maeCmaesNaive])
    mses.append([mseCmaesPrior, mseLnmi, mseEnsemble, mseLmi, mseCmaesNaive])
#%%
dices = np.array(dices)
dicesFlair = np.array(dicesFlair)
dicesT1c = np.array(dicesT1c)
maes = np.array(maes)
mses = np.array(mses)

plt.figure(figsize=(5,4))
labels = ["DL-Prior + Sampling", "DL", "DL Ensemble", "LMI", "Naive Sampling"]

for i, label in enumerate(labels):
    #if label == "DL Ensemble":
    #    continue
    mean_values = np.mean(dices[:,i,:], axis=0)
    std_values = np.std(dices[:,2,:]/ np.sqrt(len(dices)))
    plt.plot(thresholds, mean_values, label=label, color=colorsList[i], marker='.')
    plt.fill_between(thresholds, mean_values-std_values, mean_values+std_values, color=colorsList[i], alpha=0.2)

plt.xlabel("Tumor Concentration Threshold")
plt.ylabel("Dice")
plt.title("MCMC as ground Truth")
plt.legend()

#%%
labels = ["DL-Prior + Sampling", "DL", "DL Ensemble", "LMI", "Naive Sampling", "MCMC"]
plt.figure(figsize=(5,4))
for i, label in enumerate(labels):
    thresholdValue = 0.25
    argwhereTH = np.argwhere(np.array(thresholds) == thresholdValue)[0][0]
    if label != "DL-Prior + Sampling" and label != "Naive Sampling":
        # this is the final result. We comapre the result at convergence in plotExampleFigureForOnePatientrealPat.py
        print(label, np.round(np.mean(dicesFlair[:,i,argwhereTH]), 2), "+-", np.round(np.std(dicesFlair[:,i,argwhereTH])/np.sqrt(len(dicesFlair)), 2))

    if label == "MCMC":
        continue
    mean_values = np.mean(dicesFlair[:,i,:], axis=0)
    std_values = np.std(dicesFlair[:,2,:]/ np.sqrt(len(dicesFlair)))
    plt.plot(thresholds, mean_values, label=label, color=colorsList[i], marker='.')
    plt.fill_between(thresholds, mean_values-std_values, mean_values+std_values, color=colorsList[i], alpha=0.2)

plt.xlabel("Tumor Concentration Threshold")
plt.ylabel("Dice")
plt.title("Flair")
plt.legend()

#%%
plt.figure(figsize=(5,4))
labels = ["DL-Prior + Sampling", "DL", "DL Ensemble", "LMI", "Naive Sampling", "MCMC"]

for i, label in enumerate(labels):
    thresholdValue = 0.675
    argwhereTH = np.argwhere(np.array(thresholds) == thresholdValue)[0][0]
    if label != "DL-Prior + Sampling" and label != "Naive Sampling":
        # this is the final result. We comapre the result at convergence in plotExampleFigureForOnePatientrealPat.py
        print(label, np.round(np.mean(dicesT1c[:,i,argwhereTH]), 2), "+-", np.round(np.std(dicesT1c[:,i,argwhereTH])/np.sqrt(len(dicesT1c)), 2))


    if label == "MCMC":
        continue
    mean_values = np.mean(dicesT1c[:,i,:], axis=0)
    std_values = np.std(dicesT1c[:,2,:]/ np.sqrt(len(dicesT1c)))
    plt.plot(thresholds, mean_values, label=label, color=colorsList[i], marker='.')
    plt.fill_between(thresholds, mean_values-std_values, mean_values+std_values, color=colorsList[i], alpha=0.2)

plt.xlabel("Tumor Concentration Threshold")
plt.ylabel("Dice")
plt.title("T1c")
plt.legend()

    # %%
#%% barchart MAE
maes = np.array(maes)
plt.figure(figsize=(4,3))
plt.bar([0,1,2,3], [ np.mean(maes[:,0]), np.mean(maes[:,1]), np.mean(maes[:,2]), np.mean(maes[:,3])], yerr = [ np.std(maes[:,0]/np.sqrt(len(maes))), np.std(maes[:,1]/np.sqrt(len(maes))), np.std(maes[:,2]/np.sqrt(len(maes))), np.std(maes[:,3]/np.sqrt(len(maes)))], color = colorsList)
plt.xticks([0,1,2,3], [ "Sampling", "DL", "Ensemble","LMI"])
plt.title("Mean Absolute Error")
plt.ylabel("MAE")

scalingPot = 4
 
print("sampling   mean: 1e", scalingPot ,10**scalingPot * np.mean(maes[:,0]), "+-",10**scalingPot* np.std(maes[:,0]/np.sqrt(len(maes))))
print("lnmi        mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,1]), "+-",10**scalingPot* np.std(maes[:,1]/np.sqrt(len(maes))))
print("ensemble   mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,2]), "+-",10**scalingPot* np.std(maes[:,2]/np.sqrt(len(maes))))
print("lmi        mean: 1e", scalingPot ,10**scalingPot* np.mean(maes[:,3]), "+-",10**scalingPot* np.std(maes[:,3]/np.sqrt(len(maes))))


#%% barchart MSE
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