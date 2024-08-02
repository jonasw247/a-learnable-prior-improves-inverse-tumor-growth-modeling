#%%
import numpy as np
from toolsForEvaluation import evalModel
import pickle
import matplotlib.pyplot as plt

#%%
numberOfSamples = 100

modelStatePath = "/home/home/jonas/programs/learn-morph-infer/log/1403-15-48-19-v7_1-jonasTest_with25kSamplesDiffBGTissueAll/epoch49.pt"

test_data_path = "/mnt/Drive3/jonas/synthetic_data/2023_3_5___21_55_34_Sampled30k/Dataset/npz_data"

ys, yPreds, paths, flair_thrs, t1gd_thrs = evalModel(modelStatePath, test_data_path, numberOfSamples)

# %%
brainNames = []
for i in range(numberOfSamples):
    path = paths[i]
    with open(path[0]+'/parameter_tag.pkl', 'rb') as f:
        groundTruthParams = pickle.load(f)

    brainNames.append(groundTruthParams['bgTissue'])

#%% test distribution of yPreds
yPreds.shape
plt.hist(yPreds[:,1], bins=10)


# %%
savedir = {'ys': ys, 'yPreds': yPreds, 'paths': paths, 'flair_thrs': flair_thrs, 't1gd_thrs': t1gd_thrs, 'brainNames': brainNames}
np.save( './results/'+str(numberOfSamples)+'_eval_savedir.npy', savedir)

#%%

loadDir = np.load( './results/'+str(numberOfSamples)+'_eval_savedir.npy', allow_pickle=True).item()

ys = loadDir['ys']
yPreds = loadDir['yPreds']
paths = loadDir['paths']
flair_thrs = loadDir['flair_thrs']
t1gd_thrs = loadDir['t1gd_thrs']
brainNames = loadDir['brainNames']


#%%
plt.figure(figsize=(10,10))
plt.errorbar(ys.T[1], brainNames, xerr=np.abs(ys.T[2]))
# %%
plt.hist(ys.T[1], bins=brainNames)
# %%
np.histogram(ys.T[1], bins=brainNames)
# %%
arrayOfInterest = ((ys.T[4] -yPreds.T[4])/ys.T[4])
mean, std, median, values, allBrainNames, allValues = [], [], [], [], [], []
for brain in np.unique(brainNames):
    arr = arrayOfInterest[np.array(brainNames) == brain]
    mean.append(np.mean(arr))
    std.append(np.std(arr))
    median.append(np.median(arr))
    
    for i in range(len(arr)):
        allBrainNames.append(brain)
        values.append(arr[i])
    allValues.append(arr)
allBrainNames = np.array(allBrainNames)
values = np.array(values)
mean = np.array(mean)
std = np.array(std)
median = np.array(median)

# %%
firstN = 40
plt.figure(figsize=(8,10))

lenplotVals = []
for brain in np.unique(brainNames)[:firstN]:
    plotVals = values[allBrainNames == brain]
    plt.scatter( plotVals , [brain]*len(plotVals), marker='o', color = 'black', alpha=0.1)
    lenplotVals.append(len(plotVals))

plt.errorbar( mean[:firstN], np.unique(brainNames)[:firstN],xerr = std[:firstN], linestyle='None', marker='o')

plt.xlabel('relative error')
plt.ylabel('brain')
plt.xlim(-2,2)
