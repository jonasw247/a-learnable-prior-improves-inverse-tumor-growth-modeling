#%%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries
from skimage import filters

import viewDict as vd
from tool.calcLikelihood import diceLogLikelihood

import os
import nibabel as nib
import numpy as np
from matplotlib.cm import viridis

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


#%%
#for patientID in [6]:#1]:#range(100):
#    try:
patientID = 14# 13#6 original in paper
# used
xRange = (20, 120)
yRange = (10, 95)


patientName = "rec" + ("0000" + str(patientID))[-3:] + "_pre"

mcmcResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jana_SRI/"
lmiResultPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_ivan_SRI/"
lnmiOriginal = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_mich_rec_jonas_SRI/"
cmaesPriorResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_5_nSamples-600_priorInit-True_factorSTD-13_5/"
cmaesNaiveResults = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/result_CMAES_v6differentLikelihoods_loss-dice_prior-0_0_nSamples-600_priorInit-False_factorSTD-13_5/"

tumorSegmentationPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/rescaled_128_PatientData_SRI/"
tissuePath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/mich_rec_SRI_S3_maskedAndCut_rescaled_128/"
ensemblePath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/ensembleResults/"

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
#lmi[csfMask] = 0
lnmi[csfMask] = 0
cmaesPrior[csfMask] = 0
cmaesNaive[csfMask] = 0
ensemble[csfMask] = 0
flairUnmasked = flair
t1cUnmasked = t1c
flair[csfMask] = 0
t1c[csfMask] = 0

centerOfMass = np.array(ndimage.center_of_mass(flair))
slice = int(centerOfMass[2])

colorsList = ["tab:blue", "tab:red", "tab:orange",  "tab:olive", "tab:purple", "tab:grey"]
colors = {"sampling": colorsList[0], "lnmi": colorsList[1], "ensemble": colorsList[2], "lmi": colorsList[3], "naivecmaes": colorsList[4], "MCMC": colorsList[5]}

cmapBlue = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:blue"])
cmapRed = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:red"])
cmapOlive = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:olive"])
cmapOrange = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:orange"])
cmapBlack = mcolors.LinearSegmentedColormap.from_list("", ["white", "black"])
cmapPurple = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:purple"])
cmapGrey = mcolors.LinearSegmentedColormap.from_list("", ["white", "tab:grey"])
cmapGreen = mcolors.LinearSegmentedColormap.from_list("", ["white", "darkgreen"])


def plotDistance(toPlot, cmap= "Reds", label="", plotGT=True, plotSegmentation = False, plotTissue = False):
    if plotTissue:
        wmim = plt.imshow((wm+0.5*gm)[:,:,slice], cmap='Greys', vmax=0.5, vmin=-0.5, alpha = 0.5 * (np.abs(wm)+0.5*np.abs(gm)) [:,:,slice])
    #ftim = ax.imshow((meanFineTuned)[:,:,slice], cmap='Greys', alpha = 0.5 * (meanFineTuned)[:,:,slice])	
    
    values = np.abs(  toPlot)[:,:,slice]

    if plotSegmentation:
        values = (0.3 * t1c + 0.7 * flair )[:,:,slice]	
    #values = np.mean(toPlot)[:,:,slice]

    # Assuming 'segmentation' is your binary segmentation mask
    boundaryT1c = find_boundaries(t1cUnmasked[:,:,slice] > 0.1, mode='inner', connectivity = 1)
    boundaryFlair= find_boundaries(flair[:,:,slice] > 0.1, mode='thick')
    
    im = plt.imshow(values, cmap=cmap, alpha=0.8*np.abs(values))#, vmax=0.5, vmin=-0.5)

    if plotGT:
        plt.imshow(boundaryT1c, cmap='Greys', alpha=0.9*boundaryT1c)#, vmax=0.5, vmin=-0.5)
        #plt.imshow(boundaryFlair, cmap='Greys', alpha=0.7*boundaryFlair)#, vmax=0.5, vmin=-0.5)


    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([]) 
    plt.axis('off')
    plt.xlim(xRange)
    plt.ylim(yRange)
    os.makedirs('./figures/exampleIMGrealPat/'+str(patientID) +'/', exist_ok=True)
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ label+'.png' , bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ label+'.pdf', bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ label+'.svg', bbox_inches='tight')  # Save the figure
    plt.show()


plotDistance(lnmi, cmap=cmapRed, label="lnmi")
plotDistance(ensemble, cmap=cmapOrange, label="ensemble")
plotDistance(cmaesPrior, cmap=cmapBlue, label="sampling")
plotDistance(cmaesNaive, cmap=cmapPurple, label="naivecmaes")
plotDistance(lmi, cmap=cmapOlive, label="lmi")
plotDistance(mcmcResult, cmap=cmapBlack, label="MCMC")

plotDistance(0*mcmcResult, cmap=cmapGreen, label="Segmentations", plotGT=False, plotSegmentation=True, plotTissue=False)

plotDistance(0*mcmcResult, cmap=cmapGreen, label="SegmentationsWithTissue", plotGT=False, plotSegmentation=True, plotTissue=True)



dirCmaesPrior = np.load(os.path.join(cmaesPriorResults, patientName, "results.npy"), allow_pickle=True).item()
settingsCmaesPrior = np.load(os.path.join(cmaesPriorResults, patientName, "settings.npy"), allow_pickle=True).item()
dirCmaesNaive = np.load(os.path.join(cmaesNaiveResults, patientName, "results.npy"), allow_pickle=True).item()
settingsCmaesNaive = np.load(os.path.join(cmaesNaiveResults, patientName, "settings.npy"), allow_pickle=True).item()
dirCmaesPrior.keys()

#%%

def getFinalLikelihoods(toPlot, label=""):
    likelihood = np.exp(diceLogLikelihood(toPlot, flair > 0, t1c > 0, 0.25, 0.75))
    print(label + "likelihood", round(likelihood, 2))
    return likelihood

lmiFinalLikelihood = getFinalLikelihoods(lmi, "lmi")
lnmiFinalLikelihood = getFinalLikelihoods(lnmi, "lnmi")
ensembleFinalLikelihood = getFinalLikelihoods(ensemble, "ensemble")
cmaesPriorFinalLikelihood = getFinalLikelihoods(cmaesPrior, "sampling")
cmaesNaiveFinalLikelihood = getFinalLikelihoods(cmaesNaive, "naivecmaes")

mcmcResultFinalLikelihood = getFinalLikelihoods(mcmcResult, "MCMC")


#%% plot values over time
def plotValues(results, settings):
    import matplotlib.pyplot as plt
    plt.title('values - loss: ' +  settings["lossfunction"] 
              + ' - time: ' + str(round(results["time_min"]/60,1))+ 'h'
   )
    vals = ['x', 'y', 'z', 'D-cm-d', 'rho1-d']
    for i in range(len(vals))[-2:]:
        plt.plot(results['nsamples'],np.array(results['xmeans']).T[i], label = vals[i])
    #plt.yscale('log')
    plt.xlabel('# Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



plotValues(dirCmaesPrior, settingsCmaesPrior)
plotValues(dirCmaesNaive, settingsCmaesNaive)

#%% plot parameter distribution
paramsPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/mich_rec/parameterResultsCSV/"

singleNetworkParams = []
ensembleParams = []

for key in ["x", "y", "z", "D-cm-d", "rho1-d"]:
    #read csv
    params = np.genfromtxt(paramsPath + key + ".csv", delimiter=',')
    data = np.loadtxt(paramsPath + key + ".csv", delimiter=',', skiprows=1)
    for patient in data:
        if int(patient[0]) == patientID:
            print(patient)
            singleNetworkParams.append(patient[1])
            ensembleParams.append(patient[2:])

ic = [ensembleParams[0], ensembleParams[1], ensembleParams[2]]
dw = ensembleParams[3]
rho = ensembleParams[4]

# plot violin plot of parameters
plt.violinplot(ensembleParams, showmeans=True)

#%%
fig = plt.figure(figsize=(6,6) )
ax = fig.add_subplot(111, projection='3d')

def plot3D(input, linesytle, finalColor, label):

    x = input.T[0]
    y = input.T[1]
    z = input.T[2]
    # Create a colormap based on the z values
    cmap = LinearSegmentedColormap.from_list("GreenBlue", [ finalColor,"black"])
    #colors = plt.cm.jet(np.linspace(0, 1, len(z)))
    colors = cmap(np.linspace(0, 1, len(z)))
    # Plot each segment with a different color
    addlabel = True
    for i in range(1, len(x)):
        if addlabel:
            ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=colors[i], linestyle=linesytle, label =label)
            addlabel = False    
        else:
            ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=colors[i], linestyle=linesytle)
ax.grid(True)  # Remove background grid
ax.xaxis.pane.fill = True
#ax.yaxis.pane.fill = False
#ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
#plot elypsis in 3D 

ax.scatter(np.mean(ic[0]), np.mean(ic[1]), np.mean(ic[2]), color="tab:orange", label = "Ensemble", alpha=0.5, s = 1000 * 10 * np.mean(np.std(ic, axis=0))**0.5)
ax.scatter(singleNetworkParams[0], singleNetworkParams[1], singleNetworkParams[2], color="tab:red", label = "DL", s= 40)


plot3D(np.array(dirCmaesPrior['xmeans']),linesytle="-", finalColor="tab:blue", label = "DL-Prior + Sampling")

plot3D(np.array(dirCmaesNaive['xmeans']), linesytle="--", finalColor="tab:purple", label = "Naive Sampling")

legend = plt.legend(scatterpoints=1)
for handle in legend.legendHandles:
    handle._sizes = [60]

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z", labelpad=1)
#save 
nameOfFig = "3D_Origin_Space"
plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ nameOfFig+'.png' , bbox_inches='tight')  # Save the figure
plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ nameOfFig+'.pdf', bbox_inches='tight')  # Save the figure
plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ nameOfFig+'.svg', bbox_inches='tight')

plt.show()





# %%
def getLosses(lossDir):
    losses, likelihoods, priors, diceT1_67, diceFLAIR_25, times = [], [], [], [], [], []
    for epoch in range(len(lossDir)):
        thisLoss, thisLikelihood, thisPrior, thisDiceT1_67, thisDiceFLAIR_25, thisTimes = [], [], [], [], [], []
        for i in range(len(lossDir[epoch])):
            thisLoss.append(lossDir[epoch][i]["loss"])
            thisLikelihood.append(lossDir[epoch][i]["likelihood"])
            thisPrior.append(lossDir[epoch][i]["prior"])
            thisDiceT1_67.append(lossDir[epoch][i]["diceT1_67"])
            thisDiceFLAIR_25.append(lossDir[epoch][i]["diceFLAIR_25"])
            thisTimes.append(lossDir[epoch][i]["time_run"])

        losses.append(thisLoss)
        likelihoods.append(thisLikelihood)
        priors.append(thisPrior)
        diceT1_67.append(thisDiceT1_67)
        diceFLAIR_25.append(thisDiceFLAIR_25)
        times.append(thisTimes)
    return {"losses": losses, "likelihoods": likelihoods, "priors": priors, "diceT1_67": diceT1_67, "diceFLAIR_25": diceFLAIR_25, "times": times}

cmaesPriorLosses = getLosses(dirCmaesPrior["lossDir"])
cmaesNaiveLosses = getLosses(dirCmaesNaive["lossDir"])




#%%
def plotTimeseriesOfValues(dirCmaesPrior, dirCmaesNaive, value = 4, title = None):
    i = value

    plt.figure(figsize=(6, 3))
    vals = ['x', 'y', 'z', 'D-cm-d', 'rho1-d']
        # add horizontal bar
    plt.axhline(y=singleNetworkParams[i], color='tab:red', linestyle='-', label = "DL")

    mean_value = np.mean(ensembleParams[i])
    std_value = np.std(ensembleParams[i])
    #plt.axhline(y=mean_value, color='tab:orange', linestyle='-', label="Ensemble", alpha=0.8)

    time = dirCmaesPrior['time_min']
    samples = dirCmaesPrior['nsamples']
    #timeSamples = np.array(samples) * time / samples[-1] 
    simultaneousTrheads = len(cmaesPriorLosses["times"][0])
    timeSamples = simultaneousTrheads * np.cumsum(np.max(cmaesPriorLosses["times"], axis=1))/ 60 


    plt.fill_between(timeSamples/60, mean_value - 3* std_value, mean_value + 3* std_value, color='tab:orange', alpha=0.2, label="Ensemble")

    plt.plot(timeSamples /60,np.array(dirCmaesPrior['xmeans']).T[i], label = "DL-Prior + Sampling", color = "tab:blue")

    time = dirCmaesNaive['time_min']
    samples = dirCmaesNaive['nsamples']
    #timeSamples = np.array(samples) * time / samples[-1] 
    simultaneousTrheads = len(cmaesNaiveLosses["times"][0])
    timeSamples = simultaneousTrheads * np.cumsum(np.max(cmaesNaiveLosses["times"], axis=1))/ 60

    plt.plot(timeSamples/60,np.array(dirCmaesNaive['xmeans']).T[i], label = "Naive Sampling", color = "tab:purple", linestyle = "--")

    plt.xlabel('Time in h', fontsize=14)
    plt.ylabel(title, fontsize=14)
    plt.legend()
    #save
    title = title.replace(" ", "_").replace("$", "").replace("\\", "")
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ title+'.png' , bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ title+'.pdf', bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ title+'.svg', bbox_inches='tight')  # Save the figure
    
    plt.show()


plotTimeseriesOfValues(dirCmaesPrior, dirCmaesNaive, 3, title = r'$\mu_D \text{ in cm}$')
plotTimeseriesOfValues(dirCmaesPrior, dirCmaesNaive, 4, title = r'$\mu_\rho$')


#%%
def plotLossX(lossName = "losses", isLog = False):

    if isLog:
        prior = np.exp(np.array(cmaesPriorLosses[lossName]))
        naive = np.exp(np.array(cmaesNaiveLosses[lossName]))
    else:
        prior = np.array(cmaesPriorLosses[lossName])
        naive = np.array(cmaesNaiveLosses[lossName])

    plt.figure(figsize=(6, 2))

    # we take "max" for the time, as the longest thread is the one that determines the time using multithreading. As all patients were running in single but multiple patients at once, we multiply by 8 / simultaneousTrheads. 
    simultaneousTrheads = len(cmaesPriorLosses["times"][0])
    timesOfPrior = simultaneousTrheads * np.cumsum(np.max(cmaesPriorLosses["times"], axis=1))/ 60 /60
    plt.plot(timesOfPrior, np.mean(prior, axis=1), label="DL-Prior + Sampling", color = "tab:blue")

    timesOfNaive = simultaneousTrheads * np.cumsum(np.max(cmaesNaiveLosses["times"], axis=1))/ 60 /60
    plt.plot(timesOfNaive, np.mean(naive, axis=1), label="Naive Sampling", color = "tab:purple", linestyle = "--")

    plt.fill_between( timesOfPrior, np.min(prior, axis=1), np.max(prior, axis=1), alpha=0.2, color = "tab:blue")
    plt.fill_between( timesOfNaive, np.min(naive, axis=1), np.max(naive, axis=1), alpha=0.2, color = "tab:purple")

    plt.xlabel("Time in h")

    if lossName == "likelihoods":
        #plt.yscale("log")
        plt.ylabel("Likelihood")
    elif lossName == "diceFLAIR_25":
        plt.ylabel("Dice Edema")
    elif lossName == "diceT1_67":
        plt.ylabel("Dice Enhancing")

    else:
        plt.ylabel(lossName) 

    plt.legend()

    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ lossName+'.png' , bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ lossName+'.pdf', bbox_inches='tight')  # Save the figure
    plt.savefig('./figures/exampleIMGrealPat/'+str(patientID) +'/'+ lossName+'.svg', bbox_inches='tight')  # Save the figure
    plt.show()

plotLossX("likelihoods", isLog=True)
plotLossX("diceT1_67")

plotLossX("diceFLAIR_25")
plotLossX("losses")


 # %%
plotLossX("losses")

plotLossX("likelihoods", isLog=True)

plotLossX("diceT1_67")

plotLossX("diceFLAIR_25")



def getConvTimeInH(cmeasDir, relThreshold = 0.99, specificTimpoint = None):
    meanLikelihood = np.max(np.exp(cmeasDir["likelihoods"]), axis =1 )
    maxValueTh = np.max(meanLikelihood) * relThreshold
    argwhere = np.argwhere(meanLikelihood > maxValueTh)

    simultaneousTrheads = len(cmeasDir["times"][0])
    times = np.cumsum(simultaneousTrheads * np.max(cmeasDir["times"], axis=1))[argwhere[0]] / 60 /60

    maxDiceT1 = np.max(cmeasDir["diceT1_67"], axis =1 )[argwhere[0]]
    maxDiceFlair = np.max(cmeasDir["diceFLAIR_25"], axis =1 )[argwhere[0]]
    return times[0], maxDiceT1, maxDiceFlair
#%%
patientID = 6
timesPrior, timesNaive = [], []
maxT1DicePrior, maxT1DiceNaive = [], []
maxFlairDicePrior, maxFlairDiceNaive = [], []
for patientID in range(1,100):

    try:     
        patientName = "rec" + ("0000" + str(patientID))[-3:] + "_pre"

        dirCmaesPrior = np.load(os.path.join(cmaesPriorResults, patientName, "results.npy"), allow_pickle=True).item()
        dirCmaesNaive = np.load(os.path.join(cmaesNaiveResults, patientName, "results.npy"), allow_pickle=True).item()

        cmaesPriorLosses = getLosses(dirCmaesPrior["lossDir"])
        cmaesNaiveLosses = getLosses(dirCmaesNaive["lossDir"])

        priorTime, priorT1Dice, priorFlairDice = getConvTimeInH(cmaesPriorLosses)
        timesPrior.append(priorTime)
        maxT1DicePrior.append(priorT1Dice)
        maxFlairDicePrior.append(priorFlairDice)

        naiveTime, naiveT1Dice, naiveFlairDice = getConvTimeInH(cmaesNaiveLosses)
        timesNaive.append(naiveTime)
        maxT1DiceNaive.append(naiveT1Dice)
        maxFlairDiceNaive.append(naiveFlairDice)
    except:
        print("Error in patient " + str(patientID))

timesPrior = np.array(timesPrior)
timesNaive = np.array(timesNaive)
maxT1DicePrior = np.array(maxT1DicePrior)
maxT1DiceNaive = np.array(maxT1DiceNaive)
maxFlairDicePrior = np.array(maxFlairDicePrior)
maxFlairDiceNaive = np.array(maxFlairDiceNaive)
# %%
plt.plot(timesPrior, label = "DL-Prior + Sampling")
plt.plot(timesNaive, label = "Naive Sampling")
# %%
print("mean time improvement:", np.mean(timesNaive/timesPrior), "+-", np.std(timesNaive/timesPrior)/np.sqrt(len(timesNaive/timesPrior)))

#%%
print("mean time prior:", np.mean(timesPrior), "+-", np.std(timesPrior)/np.sqrt(len(timesPrior)))
print("mean time naive:", np.mean(timesNaive), "+-", np.std(timesNaive)/np.sqrt(len(timesNaive)))

#%% mean dice t1
print("mean dice T1 prior:", np.mean(maxT1DicePrior), "+-", np.std(maxT1DicePrior)/np.sqrt(len(maxT1DicePrior)))
print("mean dice T1 naive:", np.mean(maxT1DiceNaive), "+-", np.std(maxT1DiceNaive)/np.sqrt(len(maxT1DiceNaive)))

# %% mean dice flair
print("mean dice Flair prior:", np.mean(maxFlairDicePrior), "+-", np.std(maxFlairDicePrior)/np.sqrt(len(maxFlairDicePrior)))
print("mean dice Flair naive:", np.mean(maxFlairDiceNaive), "+-", np.std(maxFlairDiceNaive)/np.sqrt(len(maxFlairDiceNaive)))


# %%
print(np.mean(maxFlairDicePrior))
np.mean(maxFlairDiceNaive)
#%%
plt.plot(maxT1DicePrior, label = "DL-Prior + Sampling")
plt.plot(maxT1DiceNaive, label = "Naive Sampling")

# %%
