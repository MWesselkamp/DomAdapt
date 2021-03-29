# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:30:24 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')
import os.path
import setup.dev_cnn as dev_cnn
import setup.dev_mlp as dev_mlp
import setup.dev_lstm as dev_lstm
import torch
import torch.nn as nn
import setup.preprocessing as preprocessing
import visualizations
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['bily_kriz'],
                                years = [2001,2003, 2004, 2005, 2006],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
X_test, Y_test = preprocessing.get_splits(sites = ['bily_kriz'],
                               years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
X_sims, Y_sims = preprocessing.get_simulations(os.path.join(data_dir, r"data\simulations\uniform_params"),
                                     to_numpy = True,
                                     DOY=False,
                                     standardized=True)
X_sims_ss, Y_sims_ss = X_sims[:2000,], Y_sims[:2000,]
X_sims_ss_test, Y_sims_ss_test = X_sims[3000:3300,], Y_sims[3000:3300,]
#%%
hparams = {"batchsize": 256, 
           "epochs":3000, 
           "history":2, 
           "hiddensize":[32],
           "learningrate":0.01}
model_design = {"dimensions": [X.shape[1], 32, Y.shape[1]],
                "activation": nn.ReLU,
                "featuresize": 7}

#eval_set = {"X_test":X_test, "Y_test":Y_test}
eval_set = {"X_test":X_sims_ss_test, "Y_test":Y_sims_ss_test}
running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X_sims_ss, Y_sims_ss, eval_set, 
                                                                       0.2, data_dir, False)

visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], True, True)

#%%
hparams = {"batchsize": 512, 
           "epochs":500, 
           "history":10, 
           "hiddensize":64, 
           "learningrate":0.01}
model_design = {"dimensions":[X.shape[1], 64, Y.shape[1]],
                "activation":nn.ReLU,
                "channels":[14,28],
                "kernelsize":2}

splits=5
eval_set = None

save=False
finetuning = False
feature_extraction=False

running_losses, performance, y_tests, y_preds = dev_cnn.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir, save, finetuning, 
                                                                       feature_extraction)

visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], True, True)


#%%

hparams = {"batchsize": 512, 
           "epochs":500, 
           "history":10, 
           "hiddensize":64, 
           "learningrate":0.01}
model_design = {"dimensions":[X.shape[1], 64, Y.shape[1]],
                "activation":torch.relu,
                "channels":[14,28],
                "kernelsize":2}

splits=5
eval_set = None

save=False
finetuning = False
feature_extraction=False

running_losses, performance, y_tests, y_preds = dev_lstm.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir, save, finetuning, 
                                                                       feature_extraction)

visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], hparams, "lstm")