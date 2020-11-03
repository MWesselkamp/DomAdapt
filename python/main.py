# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import finetuning
import setup.preprocessing as preprocessing
import visualizations

import os.path
import numpy as np

#%% Load Data: Profound in and out.
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007,2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"data\simulations\uniform_params"))
#%%
pretrained_model = visualizations.losses("mlp", 7, "") 
#%%
running_losses,performance, y_tests, y_preds  = finetuning.finetune(X, Y, epochs = 1000, model="mlp", 
                                                                    pretrained_type=7, feature_extraction=None)
#%%
visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], "", "mlp")
print(np.mean(np.array(performance), axis=0))

#%% plot the losses of the baseline model MLP0 
res_mlp = visualizations.losses("mlp", 0, "") 
res_cnn = visualizations.losses("cnn", 0, "") 
res_lstm = visualizations.losses("lstm", 0, "") 

#%% plot the losses of the pretrained MLP on simulations with parameters drawn from truncated normal distribution.
visualizations.losses("mlp", 7, "", simulations = "normal", finetuned = False, setting = None)
#%% plot the losses of the pretrained MLP on simulations with parameters drawn from truncated normal distribution and dropout prob. 0.05
visualizations.losses("mlp", 8, "", simulations = "normal", finetuned = False, setting = None)
#%% plot the losses of the finetuned MLP7, finetuned WITHOUT feature_extraction (setting0)
visualizations.losses("mlp", 7, "", simulations = "normal", finetuned = True, setting = 0)
#%% plot the losses of the finetuned MLP7, finetuned WITH feature_extraction (setting1)
visualizations.losses("mlp", 7, "", simulations = "normal", finetuned = True, setting = 1)

#%%
visualizations.losses("mlp", 2, "")
visualizations.predictions("mlp", 2, "hyytiala")
visualizations.predictions("mlp", 2, "bilykriz")

visualizations.predictions("lstm", 2, "bilykriz")
visualizations.predictions("cnn", 2, "bilykriz")
visualizations.predictions("lstm", 2, "hyytiala")
visualizations.predictions("cnn", 2, "hyttiala")

visualizations.losses("mlp", 2, "", "bilykriz")
visualizations.losses("mlp", 2, "", "hyytiala")

visualizations.predictions("rf", 2)

#%%
visualizations.performance_boxplots(typ=2)
