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
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"
#%% GRID SEARCH

rets_mlp = pd.read_csv(os.path.join(data_dir, r"outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
rets_cnn = pd.read_csv(os.path.join(data_dir, r"outputs\grid_search\cnn\grid_search_results_cnn1.csv"))
rets_cnn2 = pd.read_csv(os.path.join(data_dir, r"outputs\grid_search\cnn\grid_search_results_cnn1_2.csv"))
rets_lstm = pd.read_csv(os.path.join(data_dir, r"outputs\grid_search\lstm\grid_search_results_lstm1.csv"))
rets_rf = pd.read_csv(os.path.join(data_dir, r"outputs\grid_search\rf\grid_search_results_rf1.csv"))

#%%
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="rmse",
                                           train_val = True)

visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="mae",
                                           train_val = True)

#%% SELECTED MODELS: PERFORMANCE

def model_performance(model, typ, suptitle):

    datadir = os.path.join(data_dir, f"outputs\models\{model}{typ}")

    results = pd.read_csv(os.path.join(datadir, "selected_results.csv"))
    print(results)
    running_losses = np.load(os.path.join(datadir,"running_losses.npy"), allow_pickle=True).item()
    #y_tests = np.load(os.path.join(datadir,"y_tests.npy"), allow_pickle=True).tolist()
    #y_preds = np.load(os.path.join(datadir,"y_preds.npy"), allow_pickle=True).tolist()

    visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], suptitle, model)
    #visualizations.plot_nn_predictions(y_tests, y_preds)
    #return(y_tests,y_preds)
    return(results)
    
#%%
res_mlp = model_performance("mlp", 1, "") # best performing network .
res_mlp = model_performance("mlp", 2, "") # network with this architecture on test set.
res_mlp = model_performance("mlp", 3, "") # network with this architecture on preles GPP predictions.
res_mlp = model_performance("mlp", 5, "") # network with this architecture on full simulations

res_cnn = model_performance("cnn", 5, "")
