# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:30:24 2020

@author: marie
"""
import os.path
import setup.dev_cnn as dev_cnn
import setup.dev_mlp as dev_mlp
import setup.dev_lstm as dev_lstm
import torch
import torch.nn as nn
import setup.preprocessing as preprocessing
import visualizations
import numpy as np
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
#%%
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2003, 2004],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_t, Y_t = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2003],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = "preles",
                                drop_cols=True)
#%%
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, "data\simulations"), drop_parameters=True)

#%%
hparams = {"batchsize": 256, 
           "epochs":1500, 
           "history":15, 
           "hiddensize":128, 
           "learningrate":0.01}
model_design = {"dimensions":[X.shape[1], 128, Y.shape[1]],
                "activation":nn.ReLU,
                "channels":[14,28],
                "kernelsize":3}

splits=2
eval_set = None#{"Y_test":Y_t, "X_test":X_t}

running_losses, performance, y_tests, y_preds = dev_cnn.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False)

#%%
print(np.mean(np.array(performance), axis=0))
# [0.08419205 2.260326   0.05480771 1.7048869 ]
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   "", 
                                   "cnn")

#%%
hparams = {"batchsize": 256, 
           "epochs":1500, 
           "history":0, 
           "hiddensize":[128], 
           "learningrate":0.01}
model_design = {"dimensions": [X.shape[1], 128, Y.shape[1]],
                "activation": nn.ReLU}

splits=2
eval_set = None#{"Y_test":Y_t, "X_test":X_t}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       featuresize=None,
                                                                       dropout_prob=0.0,
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False)
#%%
print(np.mean(np.array(performance), axis=0))
# [0.29031444 1.9847319  0.1960361  1.3493513 ]
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   "", 
                                   "mlp")
#%%

hparams = {"batchsize": 512, 
           "epochs":1500, 
           "history":15, 
           "hiddensize":128, 
           "learningrate":0.01}
model_design = {"dimensions":[X.shape[1], 128, Y.shape[1]],
                "activation":torch.relu}

splits=2
eval_set = None

running_losses, performance, y_tests, y_preds = dev_lstm.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False)
#%%
print(np.mean(np.array(performance), axis=0))
#[0.02860504, 2.2959127 , 0.0195967 , 1.7836074 ]
visualizations.plot_running_losses(running_losses["rmse_train"], 
                                   running_losses["rmse_val"], 
                                   "",  
                                   "lstm")