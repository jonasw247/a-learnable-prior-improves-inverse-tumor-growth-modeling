#%% run with .venv 
import numpy as np
import matplotlib.pyplot as plt
import ants

import time

from evalOriginalLMI import evaluation_utils # from learn-morph-infer repo
import torch
import sys

import toolsForEvaluation
import os
import subprocess
from scipy import ndimage
import shutil
import multiprocessing

print('ok')

#%%
def getResults(patientNumber):

    #this is the path of the LNMI files.  
    lnmiPredictionFolder = '/mnt/Drive3/jonas/LMITestDataJonas/25kSamplesDiffBGTissueAll_epoch49_Tend_100/test/npzs' + str(patientNumber)

    lnmiParams = np.load(lnmiPredictionFolder+'/allParams.npy', allow_pickle=True).item()
    
    # we read the ground truth from the lnmiParams, this could be redone such that we start from GT and simulate with lmi and lnmi predictions seperately
    groundTruthFolder = lnmiParams['gtdataPath'] + "Data_0001.npz"
    gtTumor = np.load(groundTruthFolder)
    lnmiImage = np.load(lnmiPredictionFolder+'/Data_0001.npz')
    imageAtlas = np.load('./npzstuffData_0001.npz')

    wmAtlas = imageAtlas['data'][:,:,:,2]
    #get the wm used in simulation
    wmPatient = lnmiImage['data'][:,:,:,2]
    tumor = gtTumor['data'][:,:,:]
    lnmiTumor = lnmiImage['data'][:,:,:,0]

    # getCenterOFMass
    edemaTH = lnmiParams['flair_thr'].item()
    coreTH = lnmiParams['t1gd_thr'].item()
    tumorCore = tumor > edemaTH

    centerOfMass = np.array(ndimage.center_of_mass(tumorCore))/129
       

    antsWMPatient = ants.from_numpy(np.flip(wmPatient, axis=1))
    antsTumor = ants.from_numpy(np.flip(tumor, axis=1))
  
    print('register', patientNumber)
    targetRegistration = ants.from_numpy(wmAtlas)
    regRig =  ants.registration( targetRegistration, antsWMPatient, type_of_transform='SyNCC')

    wmPatientTransformed = ants.apply_transforms(targetRegistration, antsWMPatient, regRig['fwdtransforms'])

    tumorTransformed = ants.apply_transforms(targetRegistration, antsTumor, regRig['fwdtransforms'])

    modelStatePathIvan = "/mnt/Drive3/ivan_katharina/log/torchimpl/2406-19-08-23-v7_1-mrionly-batch32-mu1mu2-xyz/bestval-model.pt"
    modelIvan = evaluation_utils.NetConstant_noBN_64_n4_l4_inplacefull(5,True)
    checkpoint = torch.load(modelStatePathIvan, map_location=torch.device('cpu'))
    modelIvan.load_state_dict(checkpoint['model_state_dict'])
    modelIvan = modelIvan.eval()

    edemaTH = lnmiParams['flair_thr'].item()
    coreTH = lnmiParams['t1gd_thr'].item()

    tumorTransformedNP = tumorTransformed.numpy()
    numpyInput = 0 * tumorTransformedNP
    numpyInput[tumorTransformedNP >= edemaTH] = 0.3333
    numpyInput[tumorTransformedNP >= coreTH] = 1

    inputIvan = torch.from_numpy(np.array([[numpyInput]]))

    print('infer', patientNumber)
    with torch.set_grad_enabled(False):
            with torch.autograd.profiler.profile() as prof:
                predictedIvan = modelIvan(inputIvan) 
                
    predictedIvan = predictedIvan.numpy()[0]

    convPredIvan = toolsForEvaluation.convert(predictedIvan[0], predictedIvan[1], predictedIvan[2], predictedIvan[3], predictedIvan[4])

    # The following is done to get the predicted tumor origin from LMI in patient space

    originIMGAtlasSpace = 0 * tumorTransformedNP

    # y axis is flipped
    xAtlasLMIOrigin, yAtlasLMIOrigin, zAtlasLMIOrigin = int(convPredIvan[3]*129) , int((convPredIvan[4])*129), int(convPredIvan[5]*129)
    originIMGAtlasSpace[xAtlasLMIOrigin,yAtlasLMIOrigin,zAtlasLMIOrigin] = 1

    originIMGAtlasSpaceAnts = ants.from_numpy(originIMGAtlasSpace)

    originIMGLMIPatientSpace = ants.apply_transforms(targetRegistration, originIMGAtlasSpaceAnts, regRig['invtransforms'])

    ivanProposedOriginPatientSpace = np.array(ndimage.center_of_mass(originIMGLMIPatientSpace.numpy()))/129

    ivanProposedOriginPatientSpace[1] = 1- ivanProposedOriginPatientSpace[1]

    return convPredIvan, lnmiParams, centerOfMass, ivanProposedOriginPatientSpace

def getFinetuned():
    
    #readFile
    loadDir = np.load( './results/multipleFinetunedModels/180_eval_savedir.npy', allow_pickle=True).item()
    
    allYPreds = np.array(loadDir['allYPreds'])
    print(allYPreds.shape)
    return allYPreds
    
#%% time the loop

start = time.time()

allAllParams, convPredIvans, centerOfMasses, ivanProposedOriginPatientSpaces = [], [], [], []
for i in range(180):#300
    print('----------------------------------- - - - - - - run ', i)
    convPredIvan, allParams, centerOfMass, ivanProposedOriginPatientSpace  = getResults(i)
    centerOfMasses.append(centerOfMass)
    convPredIvans.append(convPredIvan)
    allAllParams.append(allParams)

    ivanProposedOriginPatientSpaces.append(ivanProposedOriginPatientSpace)
    np.savez('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', convPredIvans=convPredIvans, allAllParams=allAllParams, centerOfMasses=centerOfMasses, ivanProposedOriginPatientSpaces=ivanProposedOriginPatientSpaces)


end = time.time()
print('time', end - start)


#%%
allAllParams = np.load('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', allow_pickle=True)['allAllParams']
convPredIvans = np.load('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', allow_pickle=True)['convPredIvans']
centerOfMasses = np.load('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', allow_pickle=True)['centerOfMasses']
ivanProposedOriginPatientSpaces = np.load('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', allow_pickle=True)['ivanProposedOriginPatientSpaces']
allYPreds = getFinetuned()
convPredFinetuneds = np.mean(allYPreds, axis = 0)#np.load('./evalOriginalLMI/allParamsAndPredsUpdate_RealGT_invTest_fliplater.npz', allow_pickle=True)['convPredFinetuneds']
convPredFinetunedsMax = np.max(allYPreds, axis = 0)
convPredFinetunedsMin = np.min(allYPreds, axis = 0)

#%% plotting...
variableNames = ['D - cm/d', 'rho 1/d ', 'T', 'x', 'y', 'z' , 'mu1 - cm', 'mu2 - unitless', 'originDiff']
diffIvans, diffJonass, jonasValues , gtValues, ivansValues, COMDifferencces, ivanProposedOriginPatientSpaceDiffs, jonasFineTunedDiffs, jonasFineTunedValues  = [], [],[], [], [], [], [],[],[]
ivansInOtherDirection = np.array(convPredIvans)
ivansInOtherDirection[:,4] = 1-ivansInOtherDirection[:,4]
for i in range(len(allAllParams)):
    print(i)

    diffIvan = ivansInOtherDirection[i]- np.array(allAllParams[i]['gtConvertedAfterNetwork'])

    diffJonas = np.array(allAllParams[i]['predConverted']) - np.array(allAllParams[i]['gtConvertedAfterNetwork'])

    diffJonasFinetuned = np.array(convPredFinetuneds[i]) - np.array(allAllParams[i]['gtConvertedAfterNetwork'])


    gtOrigin = np.array([np.array(allAllParams[i]['gtConvertedAfterNetwork'])[3], np.array(allAllParams[i]['gtConvertedAfterNetwork'])[4], np.array(allAllParams[i]['gtConvertedAfterNetwork'])[5]])

    ivanOrigin = np.array([ivansInOtherDirection[i][3], ivansInOtherDirection[i][4], ivansInOtherDirection[i][5]])

    jonasOrigin = np.array([np.array(allAllParams[i]['predConverted'])[3], np.array(allAllParams[i]['predConverted'])[4], np.array(allAllParams[i]['predConverted'])[5]])

    jonasFineTunedOrigin = np.array([convPredFinetuneds[i][3], convPredFinetuneds[i][4], convPredFinetuneds[i][5]])
    
    centerOfMass = centerOfMasses[i]
    ivanProposedOriginPatientSpace = ivanProposedOriginPatientSpaces[i]


    jonasOriginDiff = np.linalg.norm(jonasOrigin - gtOrigin)
    jonasFineTunedOriginDiff = np.linalg.norm(jonasFineTunedOrigin - gtOrigin)
    ivanOriginDiff = np.linalg.norm(ivanOrigin - gtOrigin)
    centerOfMassDiff = np.linalg.norm(centerOfMass - gtOrigin)
    ivanProposedOriginPatientSpaceDiff = np.linalg.norm(ivanProposedOriginPatientSpace - gtOrigin)

    jonasValues.append(np.array(allAllParams[i]['predConverted']))
    gtValues.append(np.array(allAllParams[i]['gtConvertedAfterNetwork']))
    jonasFineTunedValues.append(np.array(convPredFinetuneds[i]))
    ivansValues.append(ivansInOtherDirection[i])

    diffIvans.append( np.append(diffIvan, ivanOriginDiff))
    diffJonass.append( np.append(diffJonas, jonasOriginDiff))
    jonasFineTunedDiffs.append(np.append( diffJonasFinetuned, jonasFineTunedOriginDiff ))
    COMDifferencces.append(centerOfMassDiff)
    ivanProposedOriginPatientSpaceDiffs.append(ivanProposedOriginPatientSpaceDiff)

#%%  plt  x patients

stoplim = 25
for i in range(diffIvans[0].shape[0]-1):
    plt.figure(figsize=(5,5))
    plt.title(variableNames[i])
    plt.plot(np.array(gtValues)[:,i][:stoplim], label='GT', color = 'black')
    plt.plot(np.array(jonasValues)[:,i][:stoplim], label='LNMI', marker = 'x', linestyle='None')
    plt.plot(np.array(ivansValues)[:,i][:stoplim], label='LMI', marker = 'x', linestyle='None')

    scalingErr = 1
    errors =  scalingErr * (np.array([convPredFinetunedsMin[:, i], convPredFinetunedsMax[:, i]]) - np.array(jonasValues)[:,i])
    plt.errorbar(np.arange( len(errors.T))[:stoplim],np.array(jonasFineTunedValues)[:,i][:stoplim], yerr=np.abs(errors).T[:stoplim].T, label='LNMI fine tuned', marker = '.', linestyle='None')

    plt.xlabel('patient')
    plt.ylabel('difference')
    plt.legend()
#%%  make csv files

for i in range(diffIvans[0].shape[0]-1):

    finalArr = []

    finalArr.append(np.array(gtValues)[:,i])
    finalArr.append(np.array(jonasValues)[:,i])
    finalArr.append(np.array(ivansValues)[:,i])

    
    for j in range(10):
        finalArr.append(np.array(allYPreds)[j,:,i])

    np.savetxt('./csvResults/' +variableNames[i].replace(' ','').replace('/', '-') +'.csv', np.array(finalArr).T, delimiter=',', header='gt, lnmi, lmi, ensemble0, ensemble1,ensemble2,ensemble3,ensemble4,ensemble5,ensemble6,ensemble7,ensemble8,ensemble9', comments='')

#%% plot mu1 and mu2 std over real absolute deviation 
for i in [3,4,5,6,7]:

    finalArr = []

    diff = np.abs(np.array(gtValues)[:,i] - np.mean(np.array(allYPreds)[:,:,i], axis = 0))
    std = np.std(np.array(allYPreds)[:,:,i], axis = 0)
    corrcoef = np.corrcoef(std, diff)[0,1]

    label = 'Pearson R: ' + str(round(corrcoef, 3)) + '\nstd mean: ' + str(round(np.mean(std), 3)) + ' +- '+str(round(np.std(std), 3)) + '\ndiff mean: ' + str(round(np.mean(diff), 3)) + ' +- '+str(round(np.std(diff), 3))

    plt.figure()
    plt.title(variableNames[i])
    plt.scatter(std,diff,  marker = '.', linestyle='None', color='blue', label=label)
    plt.xlabel('ensemble std')
    plt.ylabel('absolute mean distance to ground truth')
    plt.legend()

#%% create a csv file in the folder 'csvResults' from the array gtValues
np.savetxt('./csvResults/gtValues.csv', gtValues, delimiter=',', header='Dw, rho, T, x, y, z, mu1, mu2', comments='')
np.savetxt('./csvResults/jonasValues.csv', jonasValues, delimiter=',', header='Dw, rho, T, x, y, z, mu1, mu2, diffOrigin', comments='')
np.savetxt('./csvResults/ivansValues.csv', ivansValues, delimiter=',', header='Dw, rho, T, x, y, z, mu1, mu2, diffOrigin', comments='')


#%% plot mu1 over mu2 
for i in [6,7,8]:
    plt.figure()
    plt.title(variableNames[i])

    xVals = np.array(ivansValues)[:,i] -np.array(gtValues)[:,i]
    yVals = np.array(jonasValues)[:,i] -np.array(gtValues)[:,i]
    plt.scatter(xVals, yVals, marker='.')

    plt.xlabel('LMI')
    plt.ylabel('Ours')
    plt.legend()

#%%
for i in range(diffIvans[0].shape[0]):
    plt.figure()
    plt.title(variableNames[i])
    plt.plot(np.array(diffIvans)[:,i], label='Ivan')
    plt.plot(np.array(diffJonass)[:,i], label='Jonas')
    plt.plot(np.array(jonasFineTunedDiffs)[:,i], label='Jonas fine tuned')
    if i == 8:
        plt.plot(np.array(COMDifferencces), label='COM')
        plt.plot(np.array(ivanProposedOriginPatientSpaceDiffs), label='Ivan patient space')
        

    plt.xlabel('patient')
    plt.ylabel('difference')
    plt.legend()

#%%
for i in range(diffIvans[0].shape[0]):

    plt.figure()
    plt.hist(np.array(diffIvans)[:,i], 100, label='Ivan')
    plt.hist(np.array(diffJonass)[:,i], 100, label='Jonas')

    if i == 8:
        plt.hist(np.array(COMDifferencces), 100, label='COM')

    plt.legend()
    plt.title(variableNames[i])
    plt.xlabel('difference')
    plt.ylabel('count')

#%%
plt.title('scatter plot')
plt.scatter(np.array(diffIvans)[:,6], np.array(diffIvans)[:,7], label='Ivan', marker='o', s=1)
plt.scatter(np.array(diffJonass)[:,6], np.array(diffJonass)[:,7
                                                            ], label='Jonas', marker='o', s=1)
plt.xlabel('mu1')
plt.ylabel('mu2')
plt.legend()

#%%
plt.title('scatter plot')
plt.scatter(np.array(diffIvans)[:,3], np.array(diffIvans)[:,4], label='Ivan', marker='o', s=1)
plt.scatter(np.array(diffJonass)[:,3], np.array(diffJonass)[:,4], label='Jonas', marker='o', s=1)
plt.xlabel('x')
plt.ylabel('y')

#%%
def violinPlot(dataList, relVarNames, colors):
    resultAll = np.array(dataList).T
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    parts = plt.violinplot(resultAll, points=50,  vert=True, widths=0.7, showmeans=False, showextrema=False)

    # Setting colors for individual violins
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    ax.set_xticks(range(1,1+len(relVarNames)))
    ax.set_xticklabels( relVarNames)
    #plt.legend()
    
    for i in range(len(dataList)):
        print('mean', relVarNames[i], round( np.mean(dataList[i]), 2) ,'(',  round(np.sqrt(np.mean(np.array(dataList[i])**2)), 2), ')')
         

#%%
relVarNames = ['Ours', 'Fine-tuned', 'LMI', 'Center of Mass']
colors = ['tab:blue', 'tab:red', 'tab:orange',  'tab:green']
scalingFactor = 256 
dataList = [ (scalingFactor*np.array(diffJonass)[:,8]).tolist(),
            (scalingFactor* np.array(jonasFineTunedDiffs)[:,8]), 
            (scalingFactor * np.array(ivanProposedOriginPatientSpaceDiffs)).tolist(),
            (scalingFactor * np.array(COMDifferencces)[np.array(COMDifferencces) > 0]).tolist()]

violinPlot(dataList, relVarNames, colors)
plt.ylabel("Deviation - mm")
plt.show()
#%%
i = 6 
scalingFactor = 10

relVarNames = ['Ours', 'Fine-tuned','LMI']
colors = ['tab:blue', 'tab:red', 'tab:orange']

dataList = [ (scalingFactor*np.array(diffJonass)[:,i]).tolist(), (scalingFactor* np.array(jonasFineTunedDiffs)[:,i]), 
            (scalingFactor * np.array(diffIvans)[:,i]).tolist()]

violinPlot(dataList, relVarNames, colors)

plt.ylabel("Deviation - mm")
plt.show()

#%% mu2
i=7 
relVarNames = ['Ours', 'Fine-tuned','LMI']
colors = ['tab:blue', 'tab:red', 'tab:orange']
dataList = [ np.array(diffJonass)[:,i].tolist(), np.array(jonasFineTunedDiffs)[:,i], np.array(diffIvans)[:,i].tolist()]

violinPlot(dataList, relVarNames, colors)

plt.ylabel("Deviation - unitless")

#%%
patientNumber = 16
testSetFolder = '/mnt/Drive3/jonas/LMITestDataJonas/25kSamplesDiffBGTissueAll_epoch49_Tend_100/test/npzs' + str(patientNumber)
image = np.load(testSetFolder+'/Data_0001.npz')
imageAtlas = np.load('./npzstuffData_0001.npz')
allParams = np.load(testSetFolder+'/allParams.npy', allow_pickle=True).item()

wmAtlas = imageAtlas['data'][:,:,:,2]
wmPatient = image['data'][:,:,:,2]
tumor = image['data'][:,:,:,0]
slice = int(allParams['gtConvertedAfterNetwork'][5]*129) #+ 15 #50

for space in ['patient', 'atlas tumor is wrong!']:
    plt.figure()
    if space == 'patient':
        plt.imshow(wmPatient[:,:,slice], cmap='Greys')
        plt.imshow(tumor[:,:,slice] , alpha=0.6, cmap='Reds')
    
    else:
        plt.imshow(wmAtlas[:,:,slice], cmap='Greys')
        plt.imshow(np.flip(tumor[:,:,slice], axis=1) , alpha=0.6, cmap='Reds')

            
    convPredIvan  = convPredIvans[patientNumber]
    x = int(convPredIvan[3]*129)
    y = int(convPredIvan[4]*129)
    z = int(convPredIvan[5]*129)

    plt.scatter(y,x, c='b', label='Ivan')

    x = int(allParams['gtConvertedAfterNetwork'][3]*129)
    y = int(allParams['gtConvertedAfterNetwork'][4]*129)
    z = int(allParams['gtConvertedAfterNetwork'][5]*129)

    plt.scatter(y,x, c='g', label='GT')

    x = int(allParams['predConverted'][3]*129)
    y = int(allParams['predConverted'][4]*129)
    z = int(allParams['predConverted'][5]*129)

    plt.scatter(y,x, c='r', label='Jonas')
    plt.title(space)
    plt.legend()

# %% 
def worker(command, vtuPath, npzPath, predConverted ):
    print('run ', command)

    start = time.time()
    simulation = subprocess.check_call([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/

    vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --vtk_path " + vtuPath + " --npz_path " + npzPath ], shell=True)
    
    shutil.rmtree(vtuPath)

    end = time.time()
    saveDict = {}

    saveDict['predConverted'] = predConverted
    saveDict['simtime'] = start-end


    np.save( os.path.join(npzPath, "allParams.npy"), saveDict)

numberOfProcesses = 4
pool = multiprocessing.Pool( numberOfProcesses)
outputFolder = '/mnt/Drive3/jonas/lmni_stuff/ivansPredictionsOn_LNMI_Data_realGT'

for i in range(len(convPredIvans)):
    print('----------------------------------- - - - - - - run ', i)

    predDw, predRho, predTend, predIcx, predIcy, predIcz, predMu1, predMu2 = convPredIvans[i]

    dumpFreq = 0.9999 * predTend
    
    anatomyFolder = './Atlas/anatomy_dat/'

    command = "./brain -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
    predDw) + " -rho " + str(predRho) + " -Tend " + str(int(predTend )) + " -dumpfreq " + str(dumpFreq) + " -icx " + str(
    predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"

   
    parapid = i 

    
    vtuPath = os.path.join(outputFolder, "vtus" + str(parapid) + '/')
    os.makedirs(vtuPath, exist_ok=True)
    npzPath = os.path.join(outputFolder, "npzs" + str(parapid) +'/')
    os.makedirs(npzPath, exist_ok=True)
    
    time.sleep(5)
    pool.apply_async(worker, args=(command, vtuPath, npzPath,  convPredIvans[i]))


pool.close()
pool.join()

