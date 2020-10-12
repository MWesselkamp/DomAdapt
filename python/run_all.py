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
import numpy as np

import train_selected
#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)
X_t, Y_t = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%% Grid search of hparams
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\mlp\grid_search_results_mlp4.csv"))
rets_cnn = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\cnn\grid_search_results_cnn4.csv"))
rets_cnn2 = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\grid_search_results_cnn1.csv"))
rets_lstm = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\lstm\grid_search_results_lstm1.csv"))
rets_rf = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\rf\grid_search_results_rf4.csv"))

rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()
rets_cnn.iloc[rets_cnn['rmse_val'].idxmin()].to_dict()
rets_lstm.iloc[rets_lstm['rmse_val'].idxmin()].to_dict()
rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
#%%
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_cnn], 
                                           ["rf", "mlp", "cnn"], 
                                           train_val = False)
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_cnn], 
                                           ["rf", "mlp", "cnn"], 
                                           train_val=True)

#%% Fit Random Forest
rf_minval = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
rf_mintrain = rets_rf.iloc[rets_rf['rmse_train'].idxmin()].to_dict()

y_preds_rf, y_tests_rf, errors = dev_rf.random_forest_CV(X, Y, 6, False, rf_minval["n_trees"], rf_minval["depth"], selected = True)
np.mean(np.array(errors["rmse_val"]), axis=0)
dev_rf.plot_rf_cv(y_preds_rf, y_tests_rf, rf_minval, datadir, save=False)


#%% Evaluate
model = "cnn"
typ = 1
epochs = 100
splits = 6
save=True
eval_set = None #{"X_test":X_test, "Y_test":Y_test}
finetuning = False
feature_extraction=False
data_dir = os.path.join(datadir, "python\outputs")
q=None

train_selected.train_selected(X, Y, model, typ, epochs, splits, 
                              save, eval_set, finetuning, feature_extraction, data_dir, q=None)