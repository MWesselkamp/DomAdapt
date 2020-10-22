# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:42:47 2020

@author: marie
"""
#%%
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import numpy as np
import pandas as pd
import os.path
import visualizations
import preprocessing
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005, 2006, 2007, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%% load GRID SEARCH results
rets_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
rets_cnn = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\cnn\grid_search_results_cnn1.csv"))
rets_cnn2 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\cnn\grid_search_results_cnn1_2.csv"))
rets_lstm = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\grid_search_results_lstm1.csv"))
rets_rf = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\rf\grid_search_results_rf1.csv"))

#%% MAE and RMSE of Grid search
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="rmse",
                                           train_val = True)

visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="mae",
                                           train_val = True)

    
#%% Best Network Losses and Hyperparameters
res_mlp = visualizations.losses("mlp", 0, "") # best performing network .
res_mlp = visualizations.losses("mlp", 2, "") # network with this architecture on test set.
res_mlp = visualizations.losses("mlp", 3, "") # network with this architecture on preles GPP predictions.
res_mlp = visualizations.losses("mlp", 5, "") # network with this architecture on full simulations
res_mlp = visualizations.losses("mlp", 6, "")
res_cnn = visualizations.losses("cnn", 0, "")
res_cnn = visualizations.losses("cnn", 5, "")

#%% Predictions of Best Networks.
y_tests = np.load(os.path.join(data_dir,"python\outputs\models\cnn2\y_tests.npy"), allow_pickle=True).tolist()
y_preds = np.load(os.path.join(data_dir,"python\outputs\models\cnn2\y_preds.npy"), allow_pickle=True).tolist()

visualizations.plot_nn_predictions(y_tests, y_preds)

#%%
visualizations.performance(X, Y)

