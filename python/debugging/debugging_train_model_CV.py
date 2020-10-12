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
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2003],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
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

visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], hparams, "cnn")

#%%
hparams = {"batchsize": 512, 
           "epochs":500, 
           "history":2, 
           "hiddensize":[64,64],
           "learningrate":0.01}
model_design = {"dimensions": [X.shape[1], 64,64, Y.shape[1]],
                "activation": nn.ReLU}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, splits, eval_set, 
                                                                       data_dir, save, finetuning, 
                                                                       feature_extraction)
visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], hparams, "mlp")
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