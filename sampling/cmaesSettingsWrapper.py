#!/usr/bin/python
#%%
import cmaes
import numpy as np
import os
import scipy
import struct
import sys
import nibabel as nib
import time
from tool import calcLikelihood
from runJanasForwardSolver import run as  runJanasForwardSolver

def readNii(path):
    return nib.load(path).get_fdata().astype(np.float32)

def write(a):
    path = "%dx%dx%dle.raw" % np.shape(a)
    with open(path, "wb") as file:
        file.write(a.tobytes('F'))
    sys.stderr.write("opt.py: write: %s\n" % path)

def writeNii(array, path = ""):
    if path == "":
        path = "%dx%dx%dle.nii.gz" % np.shape(array)
    nibImg = nib.Nifti1Image(array, np.eye(4))
    nib.save(nibImg, path)


class CmaesSolver():
    def __init__(self,  settings, wm, gm, flair, t1c):
        lossfunction = settings["lossfunction"]
        if lossfunction == "dice":
            self.lossfunction = calcLikelihood.logPosteriorDice
        elif lossfunction == "bernoulli":
            self.lossfunction = calcLikelihood.logPosteriorBern

        self.tend = settings["Tend"]
        self.flair_th = 0.25
        self.t1c_th = 0.675

        self.T1c = t1c
        self.FLAIR = flair
        self.GM = gm
        self.WM = wm

        self.settings = settings

        self.bpd = settings["bpd"]
        
        self.ic0 = settings["origin"]
        self.rho0 = settings["rho0"]
        self.dw0 = settings["dw0"]
        self.workers = settings["workers"]
        self.sigma0 = settings["sigma0"]
        self.generations = settings["generations"]
        self.parameterRanges = settings["parameterRanges"]
        self.diffLikelihood =  settings["diffLikelihood"]


        # check if key in settings
        if "datPath" in settings:
            self.datPath = settings["datPath"]

    def sim(self, x):
        icx = x[0]
        icy = x[1]
        icz = x[2]
        
        rho = x[4]
        dw = x[3]
        period = 0
        return runJanasForwardSolver(self.datPath, icx, icy, icz, dw, rho, self.tend)


    def fun(self, x):
        print("start fun - ", x)
        startRun = time.time()
        HG = self.sim(x)
        
        if self.diffLikelihood:
            thresholdsT1 = np.array([0.5, 0.6, 0.7, 0.8]) + 0.025
            thresholdsFLAIR =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            
            errs, likelihoods, priors = [], [], []
            for flairTH in thresholdsFLAIR:
                for t1cTH in thresholdsT1:
                    if self.settings["addPrior"] >0:
                        err_, likelihood_, prior_ =  self.lossfunction(HG, self.FLAIR, self.T1c, flairTH, t1cTH, addPrior=self.settings["addPrior"], x=x, xMeasured=self.settings["xMeasured"], stdMeasured=self.settings["stdMeasured"])
                    else:
                        err_, likelihood_, prior_ =  self.lossfunction(HG, self.FLAIR, self.T1c, flairTH, t1cTH, addPrior=0)
                    errs.append(err_)
                    likelihoods.append(likelihood_)
                    priors.append(prior_) 
            err = np.mean(errs)
            likelihood = np.mean(likelihoods)
            prior = np.mean(priors)
        else:
            if self.settings["addPrior"] >0:
                err, likelihood, prior =  self.lossfunction(HG, self.FLAIR, self.T1c, self.flair_th, self.t1c_th, addPrior=self.settings["addPrior"], x=x, xMeasured=self.settings["xMeasured"], stdMeasured=self.settings["stdMeasured"])
            else:
                err, likelihood, prior =  self.lossfunction(HG, self.FLAIR, self.T1c, self.flair_th, self.t1c_th, addPrior=0) 

        sys.stderr.write("allParameterOpt.py: %d: %.16e: %s\n" % (err, os.getpid(), str(x)))
        print(err, " --  n:")

        dicet1 = calcLikelihood.dice(HG > self.t1c_th, self.T1c)
        diceflair = calcLikelihood.dice(HG > self.flair_th, self.FLAIR)
        endrun = time.time()

        lossdir = {"loss": -err, "likelihood": likelihood, "prior": prior, "diceT1_67": dicet1, "diceFLAIR_25": diceflair}
        lossdir["allParams"] = x
        lossdir["time_run"] = endrun - startRun

        print("runtime = ", endrun - startRun)

        return -err, likelihood, prior, lossdir

    def run(self):
        start = time.time()
        
        trace = cmaes.cmaes(self.fun, ( *self.ic0, self.dw0, self.rho0), self.sigma0, self.generations, workers=self.workers, trace=True, parameterRange= self.parameterRanges)

        #trace = np.array(trace)
        nsamples, y0s, xs0s, sigmas, Cs, pss, pcs, Cmus, C1s, xmeans, likelihoods, priors, lossdirs = [], [], [], [], [], [], [], [], [], [], [], [], []
        for element in trace:
            nsamples.append(element[0])
            y0s.append(element[1])
            xs0s.append(element[2])
            sigmas.append(element[3])
            Cs.append(element[4])
            pss.append(element[5])
            pcs.append(element[6])
            Cmus.append(element[7])
            C1s.append(element[8])
            xmeans.append(element[9])
            likelihoods.append(element[10])
            priors.append(element[11])
            lossdirs.append(element[12])

        opt = xmeans[-1]

        HG = self.sim(opt)
        end = time.time()

        resultDict = {}

        resultDict["nsamples"] = nsamples
        resultDict["y0s"] = y0s
        resultDict["xs0s"] = xs0s
        resultDict["sigmas"] = sigmas
        resultDict["Cs"] = Cs
        resultDict["pss"] = pss
        resultDict["pcs"] = pcs
        resultDict["Cmus"] = Cmus
        resultDict["C1s"] = C1s
        resultDict["xmeans"] = xmeans
        resultDict["likelihoods"] = likelihoods
        resultDict["priors"] = priors
        resultDict["lossDir"] = lossdirs

        resultDict['final_loss'] = self.fun(opt)
        
        resultDict["opt_params"] = opt
        resultDict["time_min"] = (end - start) / 60
        
        return HG, resultDict

        
#%%
if __name__ == '__main__':
    print("start")

    GM = readNii("GM.nii.gz")#[::2, ::2, ::2]
    WM = readNii("WM.nii.gz")#[::2, ::2, ::2]
    T1c = readNii("tumT1c.nii.gz")[::2, ::2, ::2]
    FLAIR = readNii("tumFLAIR.nii.gz")[::2, ::2, ::2]
#%%
    settings = {}
    # ranges from LMI paper with T = 100
    parameterRanges = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3]] 
    settings["parameterRanges"] = parameterRanges

    settings["bpd"] = 16
    settings["rho0"] = 0.025
    settings["dw0"] = 0.2
    settings["workers"] = 8
    settings["sigma0"] = 0.05
    settings["generations"] =50
    settings["lossfunction"] = "bernoulli"#"dice"#
    print('Lossfunction:', settings["lossfunction"])

    solver = CmaesSolver(settings, WM, GM, FLAIR, T1c)
    resultTumor, resultDict = solver.run()

    # save results
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = "./results/"+ datetime +"_gen_"+ str(settings["generations"]) + "_loss_" + str(settings["lossfunction"]) + "/"
    os.makedirs(path, exist_ok=True)
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    writeNii(resultTumor, path = path+"result.nii.gz")
    
    print("diceT1_67",  resultDict["diceT1_67"])
    print("diceFLAIR_25",  resultDict["diceFLAIR_25"])
    print("likelihoodFlair_25",  resultDict["likelihoodFlair_25"])
    print("likelihoodT1_75",  resultDict["likelihoodT1_75"])
    print("final_loss",  resultDict["final_loss"])
    print("opt_params",  resultDict["opt_params"])
