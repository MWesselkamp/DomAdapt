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
import dev_mlp
import dev_convnet
import pandas as pd

from ast import literal_eval
import visualizations


#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['collelongo','le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None,
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR","DOY_sin", "DOY_cos"],
                                to_numpy = False)

#%% Grid search of hparams
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\grid_search_results_mlp1.csv"))
rets_convnet = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\grid_search_results_cnn1.csv"))
rets_rf = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_rf\grid_search_results_rf1.csv"))

#%%
visualizations.plot_validation_errors([rets_rf, rets_mlp, rets_convnet], ["RandomForest", "mlp", "convnet"], train_val = False)
visualizations.plot_validation_errors([rets_rf, rets_mlp, rets_convnet], ["RandomForest", "mlp", "convnet"], train_val=True)

#%% Fit Random Forest
rfp = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
y_preds_rf, y_tests_rf, errors = dev_rf.random_forest_CV(X, Y, 6, False, rfp["n_trees"], rfp["depth"], selected = True)
dev_rf.plot_rf_cv(y_preds_rf, y_tests_rf, rfp, datadir, save=False)

#%% Fit NNs
epochs = 7000
#%% MLP
mlpp = rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()
running_losses_mlp, y_tests_mlp, y_preds_mlp = dev_mlp.mlp_selected(X, Y, [mlpp["hiddensize"], mlpp["batchsize"], mlpp["learningrate"], mlpp["history"]], epochs, splits=6, searchsize=1, datadir=datadir)
 #%% Convnet
convp = rets_convnet.iloc[rets_convnet['rmse_val'].idxmin()].to_dict()
running_losses_conv, y_tests_conv, y_preds_conv = dev_convnet.conv_selected(X, Y, [convp["hiddensize"], convp["batchsize"], convp["learningrate"], convp["history"], literal_eval(convp['channels']) , convp["kernelsize"]], epochs, splits=6, searchsize=1, datadir = datadir)

#%%
visualizations.plot_errors_selmod(errors, running_losses_mlp, running_losses_conv, datadir, save=True)

#%% Load Data: Profound in, Preles out.
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = "preles")

# Fit Random Forest
y_preds_rf, y_tests_rf, errors = dev_rf.random_forest_CV(X, Y, 6, False, rfp["n_trees"], rfp["depth"], selected = True)
dev_rf.plot_rf_cv(y_preds_rf, y_tests_rf, rfp, datadir, save=False)

# Fit MLP
running_losses_mlp, y_tests_mlp, y_preds_mlp = dev_mlp.mlp_selected(X, Y, [mlpp["hiddensize"], mlpp["batchsize"], mlpp["learningrate"], mlpp["history"]], epochs, splits=6, searchsize=1, datadir=datadir)

