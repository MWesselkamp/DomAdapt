# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:30:24 2020

@author: marie
"""
import os.path
import dev_cnn
import dev_mlp
import dev_lstm
import torch
import torch.nn as nn
import preprocessing
import visualizations
import numpy as np
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
#%%
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2003, 2004, 2005, 2006, 2007, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_t, Y_t = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2003],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = "preles",
                                drop_cols=True)
#%%
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, "data\simulations"), drop_parameters=True)

#%%
hparams = {"batchsize": 512, 
           "epochs":500, 
           "history":15, 
           "hiddensize":64, 
           "learningrate":0.001}
model_design = {"dimensions":[X.shape[1], 64, Y.shape[1]],
                "activation":nn.ReLU,
                "channels":[28,52],
                "kernelsize":3}

splits=5
eval_set = None#{"Y_test":Y_t, "X_test":X_t}

running_losses, performance, y_tests, y_preds = dev_cnn.finetuning_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False, 
                                                                       feature_extraction =False)

#%%
perf = np.mean(np.array(performance), axis=0)
visualizations.plot_running_losses(running_losses["rmse_train"], 
                                   running_losses["rmse_val"], 
                                   f"Epochs: {hparams['epochs']}, History: {hparams['history']}, Hiddensize: {hparams['hiddensize']},\n Batchsize: {hparams['batchsize']}, Learning_rate: {hparams['learningrate']}, n_channels: {model_design['channels']} \n rmse_train = {perf[0]:.3f}, rmse_val={perf[1]:.3f}", 
                                   "cnn")

#%%
hparams = {"batchsize": 512, 
           "epochs":500, 
           "history":2, 
           "hiddensize":[32, 32, 32], 
           "learningrate":0.001}
model_design = {"dimensions": [32, 32, 32, Y.shape[1]],
                "activation": nn.ReLU}

splits=5
eval_set = None#{"Y_test":Y_t, "X_test":X_t}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       featuresize=7,
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False)
#%%
perf = np.mean(np.array(performance), axis=0)
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   "", 
                                   "mlp")
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
#%%
perf = np.mean(np.array(performance), axis=0)
visualizations.plot_running_losses(running_losses["rmse_train"], 
                                   running_losses["rmse_val"], 
                                   f"Epochs: {hparams['epochs']}, History: {hparams['history']}, Hiddensize: {hparams['hiddensize']},\n Batchsize: {hparams['batchsize']}, Learning_rate: {hparams['learningrate']} \n rmse_train = {perf[0]:.3f}, rmse_val={perf[1]:.3f}",  
                                   "lstm")