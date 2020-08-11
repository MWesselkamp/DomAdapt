# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing

from dev_random_forest import random_forest_CV
import experiment
import pandas as pd

import visualizations


#%% Load Data
X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

#%% Grid search of hparams
rets_nn = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn\mlp\grid_search_results.csv")
rets_rf = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf\grid_search_results.csv")
visualizations.plot_validation_errors(rets_rf, "RandomForest")
visualizations.plot_validation_errors(rets_nn, "mlp")

visualizations.plot_validation_errors([rets_rf, rets_nn], ["RandomForest", "mlp"], train_val = False)
visualizations.plot_validation_errors([rets_rf, rets_nn], ["RandomForest", "mlp"], train_val=True)

#%% Fit Random Forest
rfp = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
y_preds_rf, y_tests_rf, errors = random_forest_CV(X, Y, 6, False, rfp["n_trees"], rfp["depth"])
    
#%% Fit NN
nnp = rets_nn.iloc[rets_nn['rmse_val'].idxmin()].to_dict()
running_losses, y_tests_nn, y_preds_nn = experiment.nn_selection(X, Y, [nnp["hiddensize"], nnp["batchsize"], nnp["learningrate"], nnp["history"]], splits=6, searchsize=1)
#%%

visualizations.main(y_preds_rf, y_tests_rf, y_tests_nn, y_preds_nn, rfp, nnp)
