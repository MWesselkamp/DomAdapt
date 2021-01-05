# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:30:24 2020

@author: marie

Overfitting the networks: Try to get the MAE below 0.1.

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
import numpy as np
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
#%%
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2003],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%%
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"data\simulations\uniform_params"), drop_parameters=True)

#%%
hparams = {"batchsize": 256, 
           "epochs":1000, 
           "history":7, 
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
# [0.10712399 2.9577253  0.07802816 2.1554127 ]
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   legend=True,
                                   plot_train_loss=True)

#%%
hparams = {"batchsize": 256, 
           "epochs":8000, 
           "history":1, 
           "hiddensize":[128], 
           "learningrate":0.01}
model_design = {"dimensions": [X.shape[1], 128, Y.shape[1]],
                "activation": nn.ReLU,
                "featuresize":None}

eval_set = None#{"Y_test":Y_t, "X_test":X_t}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, eval_set, dropout_prob=0.0,
                                                                       data_dir = os.path.join(data_dir, r"python\outputs\models\mlp6"), 
                                                                       save =False,
                                                                       splits=2)
#%%
print(np.mean(np.array(performance), axis=0))
#[0.15485393 2.5280554  0.09155737 1.8440839 ]
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   legend=True,
                                   plot_train_loss=True)
#%%

hparams = {"batchsize": 256, 
           "epochs":500, 
           "history":7, 
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
#[0.03373894 2.5818696  0.01889104 1.8208545 ]
visualizations.plot_running_losses(running_losses["mae_train"], 
                                   running_losses["mae_val"], 
                                   legend=True,
                                   plot_train_loss=True)