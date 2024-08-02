import torch
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
import sys
import os
from glob import glob
#e.g. Dataset("/home/kevin/Desktop/thresholding/", 0, 9)
normalization_range = [-1.0, 1.0]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath, beginning, ending, thrpath, num_thresholds=100):
        'Initialization'
        #beginning inclusive, ending exclusive
        self.datapath = datapath
        self.beginning = beginning
        self.ending = ending
        self.thrpath = thrpath
        self.all_paths = sorted(glob("{}/*/".format(self.datapath)))[self.beginning : self.ending]
        self.threshold_paths = sorted(glob("{}/*".format(self.thrpath)))[self.beginning : self.ending]
        self.datasetsize = len(self.all_paths)
        self.epoch = 0
        self.num_thresholds = num_thresholds
 
        assert len(self.all_paths) <= len(self.threshold_paths)


  def __len__(self):
        'Denotes the total number of samples'
        #return (self.ending - self.beginning)
        return self.datasetsize

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        file_path = self.all_paths[index]
        thr_path = self.threshold_paths[index]

        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][self.epoch % self.num_thresholds]
            flair_thr = thresholdsfile['flair'][self.epoch % self.num_thresholds]

        #print("got pos " + str(index) + " which corresponds to " + str(file_path))
        with np.load(file_path + "Data_0001_thr2.npz") as data:
            #thrvolume = data['thr2_data']
            volume = data['data']
            volume_resized = np.delete(np.delete(np.delete(volume, 128, 0), 128, 1), 128, 2) #from 129x129x129 to 128x128x128

            t1gd_volume = (volume_resized >= t1gd_thr).astype(float)
            flair_volume = (volume_resized >= flair_thr).astype(float)

            input_volume = 0.666 * t1gd_volume + 0.333 * flair_volume

            thrvolume_resized = np.expand_dims(input_volume, -1) #now it is 128x128x128x1
            #print(thrvolume_resized.shape)

        with open(file_path + "parameter_tag2.pkl", "rb") as par:

            paramsarray = np.zeros(8)
            params = pickle.load(par)
            paramsarray[0] = np.interp(t1gd_thr, [0.5, 0.85], normalization_range)
            paramsarray[1] = np.interp(flair_thr, [0.05, 0.5], normalization_range)
            paramsarray[2] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
            paramsarray[3] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
            paramsarray[4] = np.interp(params['Tend'], [50, 1500], normalization_range)
            paramsarray[5] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
            paramsarray[6] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
            paramsarray[7] = np.interp(params['icz'], [0.15, 0.7], normalization_range)

        thrvolume_resized = thrvolume_resized.transpose((3,0,1,2))
        return torch.from_numpy(thrvolume_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32))

##########################################################################################################

class Dataset2(Dataset):
    # We remove tanh from last layer when predicting infiltration length + Tp + velocity, because mean of Dw and p
    # after bringing into range [-1, 1] was at 0. Since we now predict products of these factors, we can observe
    # our data and see that when we normalize into [-1, 1] range, the mean (of our TRAINING DATA) is not at 0 anymore!
    def __init__(self, datapath, beginning, ending, thrpath, necroticpath, num_thresholds=100, includesft=False, outputmode=0, isOnlyAtlas = False, isEvalMode =False, thresholds=None):
        Dataset.__init__(self, datapath, beginning, ending, thrpath, num_thresholds=num_thresholds)
        self.isOnlyAtlas = isOnlyAtlas

        self.isEvalMode = isEvalMode

        if isOnlyAtlas:
            self.npzFileName = "Data_0001_thr2.npz"
            self.paramsFileName = "parameter_tag2.pkl"
        else: 
            self.npzFileName = "Data_0001.npz"
            self.paramsFileName = "parameter_tag.pkl"


        self.includesft = includesft
        self.outputmode = outputmode
        self.necroticpath = necroticpath
        self.fixedthresholds = thresholds
        #self.necrotic_paths = sorted(glob("{}/*".format(self.necroticpath)))[self.beginning : self.ending]
        #assert len(self.necrotic_paths) == self.datasetsize

    def __len__(self):
        return self.datasetsize

    def __getitem__(self, index):
        file_path = self.all_paths[index]
        thr_path = self.threshold_paths[index]
        #necrotic_path = self.necrotic_paths[index]

        with np.load(thr_path) as thresholdsfile:
            if self.fixedthresholds is not None:
                t1gd_thr = self.fixedthresholds[1]
                flair_thr = self.fixedthresholds[0]
                #print('fixed thresholds: ', t1gd_thr, flair_thr)
            else:
                t1gd_thr = thresholdsfile['t1gd'][self.epoch % self.num_thresholds]
                flair_thr = thresholdsfile['flair'][self.epoch % self.num_thresholds]
            assert t1gd_thr >= 0.5 and t1gd_thr <= 0.85
            assert flair_thr >= 0.05 and flair_thr <= 0.5


        with np.load(file_path + self.npzFileName) as data:
            # thrvolume = data['thr2_data']
            volume = data['data']
        volume_resized = volume
        
        t1gd_volume = (volume_resized >= t1gd_thr).astype(float)
        flair_volume = (volume_resized >= flair_thr).astype(float)

        thr_volume = 0.666 * t1gd_volume + 0.333 * flair_volume

        thrvolume_resized = np.expand_dims(thr_volume, -1)  # now it is 129x129x129x1


        #include white matter:
        if self.isOnlyAtlas:
            #put in tumor twice
            nn_input = np.concatenate((thrvolume_resized, thrvolume_resized, thrvolume_resized, thrvolume_resized), -1)
        else:
            with np.load(file_path + "sim_output_WM.npz") as data:
                whiteMatter = data['data']
            with np.load(file_path + "sim_output_GM.npz") as data:
                grayMatter = data['data']
            with np.load(file_path + "sim_output_CSF.npz") as data:
                csfMatter = data['data']


            whiteMatter_resized = np.expand_dims(whiteMatter, -1)
            grayMatter_resized = np.expand_dims(grayMatter, -1)
            csfMatter_resized = np.expand_dims(csfMatter, -1)

            # change take the tumor volume twice 
            #nn_input = np.concatenate((thrvolume_resized, thrvolume_resized), -1)
            nn_input = np.concatenate((thrvolume_resized, whiteMatter_resized, grayMatter_resized, csfMatter_resized), -1)

        if self.includesft:

            raise Exception("no support for ft in this version")

        with open(file_path + self.paramsFileName, "rb") as par:

            params = pickle.load(par)

            Dw = params['Dw'] #cm^2 / d
            rho = params['rho'] #1 / d
            Tend = params['Tend'] #d

            lambdaw = np.sqrt(Dw / rho) #cm
            mu = Tend * rho #constant

            mu1 = np.sqrt(Dw * Tend)
            mu2 = np.sqrt(rho * Tend)
            
            velocity = 2 * np.sqrt(Dw * rho) #cm / d

            if self.outputmode == 0:
                paramsarray = np.zeros(8)
                paramsarray[0] = np.interp(t1gd_thr, [0.5, 0.85], normalization_range)
                paramsarray[1] = np.interp(flair_thr, [0.05, 0.5], normalization_range)
                paramsarray[2] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[3] = np.interp(mu, [0.1, 300.0], normalization_range)
                paramsarray[4] = np.interp(velocity, [2*np.sqrt(4e-7), 2*np.sqrt(0.003)], normalization_range)
                paramsarray[5] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[6] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[7] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            elif self.outputmode == 1:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[1] = np.interp(mu, [0.1, 300.0], normalization_range)
                paramsarray[2] = np.interp(velocity, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)], normalization_range)
            elif self.outputmode == 2:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[1] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[2] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            elif self.outputmode == 3:
                paramsarray = np.zeros(2)
                paramsarray[0] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[1] = np.interp(mu, [0.1, 300.0], normalization_range)
            elif self.outputmode == 4:
                paramsarray = np.zeros(2)
                sqrtDT = np.sqrt(Dw * Tend)
                sqrtmu = np.sqrt(mu)
                paramsarray[0] = np.interp(sqrtDT, [0.1, np.sqrt(22.5)], normalization_range)
                paramsarray[1] = np.interp(sqrtmu, np.sqrt([0.1, 300.0]), normalization_range)
            elif self.outputmode == 5:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
                paramsarray[1] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
                paramsarray[2] = np.interp(params['Tend'], [50, 1500], normalization_range)
            elif self.outputmode == 6:
                paramsarray = np.zeros(5)
                paramsarray[0] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[1] = np.interp(mu, [0.1, 300.0], normalization_range)
                paramsarray[2] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[3] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[4] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            elif self.outputmode == 7:
                paramsarray = np.zeros(2)
                paramsarray[0] = np.interp(mu1, [np.sqrt(0.01), np.sqrt(22.5)], normalization_range)
                paramsarray[1] = np.interp(mu2, [np.sqrt(0.1), np.sqrt(300)], normalization_range)
            elif self.outputmode == 8:
                paramsarray = np.zeros(5)
                paramsarray[0] = np.interp(mu1, [np.sqrt(0.01), np.sqrt(22.5)], normalization_range)
                paramsarray[1] = np.interp(mu2, [np.sqrt(0.1), np.sqrt(300)], normalization_range)
                paramsarray[2] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[3] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[4] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            else:
                raise Exception("invalid output mode")


        nninput_resized = nn_input.transpose((3, 0, 1, 2))

        if self.isEvalMode:
            return torch.from_numpy(nninput_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32)), paramsarray, flair_thr, t1gd_thr, file_path
        return torch.from_numpy(nninput_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32))
