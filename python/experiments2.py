# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:44:41 2020

@author: marie
"""
#%% Set working directory
import os
import os.path
#os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing

import pandas as pd
import dev_lstm
import dev_convnet

from dev_convnet import conv_selection_parallel
import multiprocessing as mp
import numpy as np
import visualizations


import torch.nn as nn
#%% Load Data
datadir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ["le_bray"], 
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound")
#%%
# Network training
hparams = {"batchsize": 64, 
           "epochs":100, 
           "history":10, 
           "hiddensize":32,
           "learningrate":0.01}
model_design = {"dimensions": [X.shape[1], 32, Y.shape[1]],
                "activation":nn.Sigmoid,
                "dim_channels":[7,14],
                "kernel_size":2}
   
running_losses,performance, y_tests_nn, y_preds_nn = dev_lstm.train_model_CV(hparams, model_design, X, Y, splits=6)

# performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
performance = np.mean(np.array(performance), axis=0)

#%% 
visualizations.plot_nn_loss(running_losses["rmse_train"], 
                            running_losses["rmse_val"], 
                            hparams = hparams, 
                            datadir = os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn"), 
                            figure = "ex", model="lstm")
visualizations.plot_nn_predictions(y_tests_nn, 
                                   y_preds_nn, 
                                   history = hparams["history"], 
                                   datadir = os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn"), 
                                   figure = "ex", model="lstm")