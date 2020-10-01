# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import preprocessing
import os.path

import dev_rf

import pandas as pd

from ast import literal_eval 
import visualizations
import torch.nn.functional as F
import torch.nn as nn

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007, 2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = "preles")
X, Y_t = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007, 2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%% Grid search of hparams
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\grid_search_results_mlp1.csv"))
rets_cnn = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\grid_search_results_cnn1.csv"))
rets_lstm = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\grid_search_results_lstm1.csv"))
rets_rf = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_rf\grid_search_results_rf1.csv"))

rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()
rets_cnn.iloc[rets_cnn['rmse_val'].idxmin()].to_dict()
rets_lstm.iloc[rets_lstm['rmse_val'].idxmin()].to_dict()
rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
#%%
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_cnn, rets_lstm], 
                                           ["rf", "mlp", "cnn", "lstm"], 
                                           train_val = False)
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_cnn, rets_lstm], 
                                           ["rf", "mlp", "cnn", "lstm"], 
                                           train_val=True)

#%% Fit Random Forest
rf_minval = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
rf_mintrain = rets_rf.iloc[rets_rf['rmse_train'].idxmin()].to_dict()

y_preds_rf, y_tests_rf, errors = dev_rf.random_forest_CV(X, Y, 6, False, rf_minval["n_trees"], rf_minval["depth"], selected = True)
dev_rf.plot_rf_cv(y_preds_rf, y_tests_rf, rf_minval, datadir, save=False)


#%%
visualizations.plot_errors_selmod(errors, running_losses_mlp, running_losses_conv, datadir, save=True)

#%% Evaluate on test set.
epochs = 1000
best_model = rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()
best_model["hiddensize"] = '[128, 128]'
best_model["activation"] = "<class 'torch.nn.modules.activation.ReLU'>"
running_losses, y_tests, y_preds, rets = dev_mlp.selected(X, Y, best_model, epochs, 6, Y_t)

best_model["epochs"]=epochs
visualizations.plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], best_model, model="mlp")
