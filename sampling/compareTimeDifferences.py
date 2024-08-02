
#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from  recPred.dataset_syntheticBrats import Dataset_SyntheticBrats
import viewDict as vd
from tool import calcLikelihood
from  tools import pairedTTest


directoryPath = "/mnt/8tb_slot8/jonas/datasets/lmiSynthetic/"
workdir = "/mnt/8tb_slot8/jonas/workingDirDatasets/lmiSynthetic"
testDataset = Dataset_SyntheticBrats(directoryPath, workdir)
# %%
modalities = ["result_CMAES_v5InitEnsembleOrig_loss-dice_prior-0_5_nSamples-600_priorInit-True_factorSTD-13_5", "result_CMAES_v5InitEnsembleOrig_loss-dice_prior-0_0_nSamples-2000_priorInit-False_factorSTD-13_5"]
allAllresults, allAllpatients, allAllSettings, allAllDices, allAllMses, allAllMaes = [],[],[],[], [], []

for j in range(len(modalities)):
    modality = modalities[j]
    dices, allresults, allpatients, allSettings, maes, mses = [],[],[],[], [], []
    for i in range(0,13):
        try: 
            print('')

            print(i, ' - ' , testDataset.getInternalPatientID(i))

            gt = testDataset.loadPatientImageEnumarated(i, "result_groundTruth", '0', "dat128JanasSolver")
            lnmi = testDataset.loadPatientImageEnumarated(i, "result_lnmi", '0', "dat128JanasSolver")
            lmi = testDataset.loadPatientImageEnumarated(i, "result_lmi", '0', "dat128JanasSolver")

            ensemble = testDataset.loadPatientImageEnumarated(i, "result_ensemble", '0', "dat128JanasSolver")

            flair = testDataset.loadPatientImageEnumarated(i, "seg-flair", '0', "dat128JanasSolver")

            t1c = testDataset.loadPatientImageEnumarated(i, "seg-t1c", '0', "dat128JanasSolver")

            print("flair" , np.sum(flair * (gt > 0.25) )/ np.sum(flair))
            print("t1c" , np.sum(t1c * (gt > 0.5) )/ np.sum(t1c))



            samplingResults = testDataset.loadPatientImageEnumarated(i, modality, '0', "dat128JanasSolver")

        except:      
            print('-----------   failed: ', i)
            continue
        
        th = 0.05
        diceSampling = calcLikelihood.dice(gt > th, samplingResults> th)
        diceLnmi = calcLikelihood.dice(gt > th, lnmi> th)
        diceLmi = calcLikelihood.dice(gt > th, lmi> th)
        diceEnsemble = calcLikelihood.dice(gt > th, ensemble> th)

        maeSampling = np.mean(np.abs(gt - samplingResults))
        maeLnmi = np.mean(np.abs(gt - lnmi))
        maeLmi = np.mean(np.abs(gt - lmi))
        maeEnsemble = np.mean(np.abs(gt - ensemble))

        mseSampling = np.mean(np.square(gt - samplingResults))
        mseLnmi = np.mean(np.square(gt - lnmi))
        mseLmi = np.mean(np.square(gt - lmi))
        mseEnsemble = np.mean(np.square(gt - ensemble))

        print(diceSampling, diceLnmi ,diceEnsemble)
        dictAll = testDataset.loadResultsDictEnumerated(i, modality, '0', "dat128JanasSolver")
        allresults.append(dictAll["results"])
        allSettings.append(dictAll["settings"])
        allpatients.append(i)

        dices.append([diceSampling, diceLnmi, diceEnsemble, diceLmi, i])
        maes.append([maeSampling, maeLnmi, maeEnsemble, maeLmi, i])
        mses.append([mseSampling, mseLnmi, mseEnsemble, mseLmi, i])

    allAllresults.append(allresults)
    allAllpatients.append(allpatients)
    allAllSettings.append(allSettings)
    allAllDices.append(dices)
    allAllMses.append(mses)
    allAllMaes.append(maes)

factor = 4
allAllMaes = np.array(allAllMaes) * 10**factor
factorMSE = 5
allAllMses = np.array(allAllMses) * 10**factorMSE
#%%
allAllDices = np.array(allAllDices)
plt.plot( allAllpatients[0], allAllDices[0,:,0], label = 'prior', linewidth = 0, marker = 'o')
plt.plot( allAllpatients[1], allAllDices[1,:,0], label = 'no prior', linewidth = 0, marker = 'o')

# 
plt.xlabel("patients")
plt.legend()
print('prior', np.mean(allAllDices[0,:,0]), '+-', np.std(allAllDices[0,:,0])/np.sqrt(len(allAllDices[0,:,0])))
print('no prior', np.mean(allAllDices[1,:,0]), '+-', np.std(allAllDices[1,:,0])/np.sqrt(len(allAllDices[1,:,0])))
#%% plot maes
plt.title("Mean Absolute Error")
plt.plot( allAllpatients[0], allAllMaes[0,:,0], label = 'prior', linewidth = 0, marker = 'o')
plt.plot( allAllpatients[1], allAllMaes[1,:,0], label = 'no prior', linewidth = 0, marker = 'o')
plt.legend()
plt.xlabel("patients")
plt.ylabel("mae")
print('prior 1e', factor, np.mean(allAllMaes[0,:,0]), '+-', np.std(allAllMaes[0,:,0])/np.sqrt(len(allAllMaes[0,:,0])))
print('no prior 1e ', factor,  np.mean(allAllMaes[1,:,0]), '+-', np.std(allAllMaes[1,:,0])/np.sqrt(len(allAllMaes[1,:,0])))

#%% plot mses
plt.title("Mean Squared Error")
plt.plot( allAllpatients[0], allAllMses[0,:,0], label = 'prior', linewidth = 0, marker = 'o')
plt.plot( allAllpatients[1], allAllMses[1,:,0], label = 'no prior', linewidth = 0, marker = 'o')
plt.legend()
plt.xlabel("patients")
plt.ylabel("mse")
print('prior', factorMSE, np.mean(allAllMses[0,:,0]), '+-', np.std(allAllMses[0,:,0])/np.sqrt(len(allAllMses[0,:,0])))
print('no prior', factorMSE,  np.mean(allAllMses[1,:,0]), '+-', np.std(allAllMses[1,:,0])/np.sqrt(len(allAllMses[1,:,0])))

#%%
def moving_average(data, window_size):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

plt.figure(figsize=(10,7))
plt.title("Posterior distrubution til maximum")
color = ['r', 'g', 'b', 'y']	
allendsamples, alltotalTimes, alltotalMax = [], [], []
for i in range(0,13): #range(len(allAllresults[0])):
    p = allSettings[i]["addPrior"]

    endsamples, totalTimes, allMax  = [], [], []

    for j in range(len(modalities)):

        values = np.exp(-np.array(allAllresults[j][i]['y0s']))

        maximum = np.max(values)
        print(maximum , " - ", np.argmax(values))
        
        #first to reach 95% of maximum
        infLim = np.where(values > 0.99 * maximum)[0][0]
        convLim = allAllresults[j][i]['nsamples'][infLim]

        maxLikelihood = np.exp(np.max(np.array(allAllresults[j][i]['likelihoods']), axis=1))[infLim]

        #plot vertical line at first95
        plt.axvline(x=convLim, color=color[i%len(color)], linestyle='--')


        if j > 0:
            linestyle = '-'
        else:
            linestyle = ':'
        #
        window = 1
        # start from 1
        starfrom = 0	
        plt.plot(moving_average(allAllresults[j][i]['nsamples'][starfrom:], window_size=window), moving_average(values[starfrom:], window)/maximum, label = str(allAllpatients[j][i])  + "  dice: " + str(round(allAllDices[j,i,0],2))+ " time "+ str(round(allAllresults[j][i]['time_min']/60,2)) + "h", linewidth = 0.8, color = color[i%len(color)], linestyle = linestyle)

        endsamples.append(convLim)
        totalTimes.append(allAllresults[j][i]['time_min']/60)
        allMax.append(maxLikelihood)
    allendsamples.append(endsamples)
    alltotalTimes.append(totalTimes)
    alltotalMax.append(allMax)

allendsamples = np.array(allendsamples)
alltotalTimes = np.array(alltotalTimes)
alltotalMax = np.array(alltotalMax)

plt.xlabel("patients")
plt.xlabel("samples")
plt.legend()
plt.figure(figsize=(10,7))




# %%
a = alltotalTimes *0 
a[:,0] = 600
a[:,1] = 2000

sampleTime = allendsamples/a

plt.title("Time per sample")
plt.plot(sampleTime[:,0], label = "prior")
plt.plot(sampleTime[:,1], label = "no prior")
plt.ylabel("sample time in min")
plt.xlabel("patients")
plt.legend()
#%%
totalCovergenceTime = sampleTime * allendsamples
plt.title("Total time to converge")
plt.plot(totalCovergenceTime[:,0], label = "prior")
plt.plot(totalCovergenceTime[:,1], label = "no prior")
plt.ylabel("total time in min")
plt.xlabel("patients")
plt.legend()
plt.yscale("log")


# %%
# %%
labels = ["DL-Inference", 'DL-Prior + Sampling', 'Naive Sampling'][::-1] # Reverse the order
color = ["tab:orange", "tab:blue", "tab:purple"][::-1] # Reverse the order
edgecolors = ["tab:orange", "black", "black"][::-1] # Reverse the order
linewidth = [2, 0, 0][::-1] # Reverse the order
hatches = ["", "", "//"][::-1] # Reverse the order
means = [2 * np.mean(sampleTime), np.mean(totalCovergenceTime[:,0]), np.mean(totalCovergenceTime[:,1])][::-1] 
std_devs = [2 * np.std(sampleTime), np.std(totalCovergenceTime[:,0]), np.std(totalCovergenceTime[:,1])][::-1] / np.sqrt(len(totalCovergenceTime[:,0])) 

pairedTTest(totalCovergenceTime[:,0], totalCovergenceTime[:,1])

plt.rcParams['font.size'] = 17 # Increase the global font size
plt.figure(figsize=(7,2)) # Added to make the figure larger


plt.xlabel("Convergence Time in h") # Changed to xlabel as we are now dealing with horizontal bars
plt.xlim([-0.5, 15]) # Changed to xlim as we are now dealing with horizontal bars
# Add text after the bars
bar_container = plt.barh(labels, np.array(means) /60, xerr=std_devs/60, color=color,hatch=hatches, edgecolor=edgecolors, linewidth = linewidth ) # Changed to barh for horizontal bars

#for i in [3]:
rect = bar_container.patches[2]
width = rect.get_width()
plt.text(width + 0.5, rect.get_y() + rect.get_height() / 2, str(int(width * 60 *60)) +" s", ha='left', va='center')


plt.savefig("figures/convergenceTime.pdf", bbox_inches='tight') # Changed to savefig as we are now dealing with horizontal bars
# %% plot alltotalMax
plt.title("Maximum Likelihood")
plt.plot(alltotalMax[:,0], label = "prior")
plt.plot(alltotalMax[:,1], label = "no prior")
plt.ylabel("maximum likelihood")
plt.xlabel("patients")
plt.legend()
        

# %%

#%% plot mean likelihood over time
labels = ['DL-Prior + Sampling', 'Naive Sampling']
color = ["tab:blue", "tab:purple"]
#colors = ["tab:blue", "tab:purple"]
plt.figure(figsize=(10, 3))
valuesAt2h = []
for j in range(len(allAllresults)):
    mod = allAllresults[j]

    likelihoods = []
    timeSamplesS = []

    for i in range(len(mod)):
        patient = mod[i]
        time = patient['time_min']
        samples = patient['nsamples']
        timeSamples = np.array(samples) * time / samples[-1] # individual time was not recoreded

        likelihood = np.exp(np.max(np.array(patient['likelihoods']), axis=1))

        likelihoods.append(likelihood)
        timeSamplesS.append(timeSamples)

    #plot mean
    likelihoods = np.array(likelihoods)
    timeSamplesS = np.array(timeSamplesS)

    allTimes = np.unique(np.concatenate(timeSamplesS))
    
    allInterpolatedTimeseries = []
    for i in range(len(likelihoods)):
        likelihoodInterpolated = np.interp(allTimes, timeSamplesS[i], likelihoods[i])
        allInterpolatedTimeseries.append(likelihoodInterpolated)
    
    allInterpolatedTimeseries = np.array(allInterpolatedTimeseries)
    mean = np.mean(allInterpolatedTimeseries, axis=0)
    std = np.std(allInterpolatedTimeseries, axis=0) 
    
    #add moving average
    movingAvg = 1

    if j > 0:
        linestyle = '--'
    else:
        linestyle = '-'

    plt.plot(moving_average(allTimes / 60, movingAvg), moving_average(mean, movingAvg), label=labels[j], color=color[j], linewidth=1.5, linestyle=linestyle)

    #std   = np.std(allInterpolatedTimeseries, axis=0)
    stdErr = moving_average(std / np.sqrt(len(allInterpolatedTimeseries)), movingAvg)

    plt.fill_between(moving_average(allTimes / 60, movingAvg), moving_average(mean, movingAvg) + stdErr ,moving_average(mean, movingAvg) -stdErr, color = color[j], alpha = 0.4)

    # fill std
    plt.fill_between(moving_average(allTimes / 60, movingAvg), moving_average(mean, movingAvg) + 3 * stdErr ,moving_average(mean, movingAvg) - 3 * stdErr, color = color[j], alpha = 0.2)

    for i in range(allTimes.shape[0]):
        if allTimes[i] / 60 >= 2:
            print(labels[j])
            print(allTimes[i] / 60)
            print(mean[i] , "+-", std[i] / np.sqrt(len(allInterpolatedTimeseries)))

            valuesAt2h.append(allInterpolatedTimeseries.T[i])
            break

    #add horizontal gray line at 2h
    plt.axvline(x=2, color='gray', linestyle='-')



plt.xlim([0, 14.6])
#plt.ylim([0,0.95])

plt.xlabel("Time in h")
plt.ylabel("Likelihood")
plt.legend()
plt.savefig("figures/meanLikelihoodOverTime.pdf", bbox_inches='tight') 

valuesAt2h = np.array(valuesAt2h)
#ttest:
pairedTTest(valuesAt2h[1], valuesAt2h[0])
# %%
