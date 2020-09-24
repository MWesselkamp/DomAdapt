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
import dev_lstm
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
                                simulations = None,
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR","DOY_sin", "DOY_cos"],
                                to_numpy = True)

#%% Grid search of hparams
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\grid_search_results_mlp1.csv"))
rets_convnet = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\grid_search_results_cnn1.csv"))
rets_rf = pd.read_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_rf\grid_search_results_rf1.csv"))

#%%
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_convnet], 
                                           ["rf", "mlp", "cnn"], 
                                           train_val = False)
visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_convnet], 
                                           ["rf", "mlp", "cnn"], 
                                           train_val=True)

#%% Fit Random Forest
rf_minval = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
rf_mintrain = rets_rf.iloc[rets_rf['rmse_train'].idxmin()].to_dict()

y_preds_rf, y_tests_rf, errors = dev_rf.random_forest_CV(X, Y, 6, False, rf_minval["n_trees"], rf_minval["depth"], selected = True)
dev_rf.plot_rf_cv(y_preds_rf, y_tests_rf, rf_minval, datadir, save=False)


#%%
visualizations.plot_errors_selmod(errors, running_losses_mlp, running_losses_conv, datadir, save=True)

