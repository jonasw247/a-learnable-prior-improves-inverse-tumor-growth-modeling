#%%
import torch
import sys
import numpy as np
import network
import dataloader
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader


# this is the standard method used for the testset using the dataloader 
def evalModel(modelStatePath, test_data_path, endTest = 1000, device = 'cpu'):
    checkpoint = torch.load(modelStatePath, map_location=torch.device(device=device))

    model = network.NetConstant_noBN_64_n4_l4_inplacefull(5, None, False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.eval()

    mri_threshold_path= "/mnt/Drive3/ivan_kevin/thresholds/files"

    necro_threshold_path= None

    startTest = 0
    batchSize = 1
    test_dataset = dataloader.Dataset2(test_data_path, startTest, endTest,
                        mri_threshold_path, necro_threshold_path,
                        includesft=False, outputmode=8, isOnlyAtlas=False, isEvalMode=True)
    test_generation = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=batchSize, shuffle=False, num_workers=1)

    ys = []
    yPreds = []
    paths = []
    flair_thrs, t1gd_thrs = [], []
    with torch.set_grad_enabled(False):
        for batch_idy, (x,y, paramsarray, flair_thr, t1gd_thr, file_path) in enumerate(test_generation):
            x, y = x.to(device), y.to(device)
            y_predicted = model(x)
            y = y.cpu().detach().numpy()
            yPred = y_predicted.cpu().detach().numpy()
            ys.append(y[0])
            yPreds.append(yPred[0])
            paths.append(file_path)
            flair_thrs.append(flair_thr)
            t1gd_thrs.append(t1gd_thr)


    return np.array(ys), np.array(yPreds), paths, flair_thrs, t1gd_thrs


def combineNPZTimeFilesToNPY(path):
    files = np.sort(os.listdir(path))

    sim_tumors = []
    for i in range(len(files)):
        if not 'Data' in files[i]:
            continue
        if not '.npz' in files[i]:
            continue

        with np.load(os.path.join(path,files[i])) as simtumor:
            print('load timepoint:', files[i])
            sim_tumor = simtumor['data'][:, :, :, 0]
        
        sim_tumors.append(sim_tumor)

    np.save(os.path.join(path, 'sim_tumors_time_evolution.npy'), np.array(sim_tumors))
    return np.array(sim_tumors)

def convert(mu1, mu2, x, y, z, selectedTEnd = 100): 

    T = selectedTEnd
    normalization_range = [-1.0, 1.0]

    mu1 = np.interp(mu1, normalization_range, [np.sqrt(0.01), np.sqrt(22.5)]) 
    mu2 = np.interp(mu2, normalization_range, [np.sqrt(0.1), np.sqrt(300)])  

    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = mu1**2 / T
    rho =  mu2**2 / T 

    return D, rho, T, x, y, z , mu1, mu2

        
# %%
