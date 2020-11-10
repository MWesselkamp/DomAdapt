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
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

#%% load GRID SEARCH results
rets_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\mlp\grid_search_results_mlp2.csv"))
rets_mlp_adapt = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\mlp\AdaptPool\grid_search_results_mlp2.csv"))
rets_mlp4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\mlp\grid_search_results_mlp4.csv"))
rets_cnn = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\cnn\grid_search_results_cnn2.csv"))
rets_cnn4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\cnn\grid_search_results_cnn4.csv"))
rets_lstm = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\lstm\grid_search_results_lstm2.csv"))
rets_lstm4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\lstm\grid_search_results_lstm4.csv"))
rets_rf = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\rf\grid_search_results_rf2.csv"))
rets_rf4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\rf\grid_search_results_rf4.csv"))
#%% MAE and RMSE of Grid search
visualizations.hparams_optimization_errors([rets_mlp4, rets_cnn4, rets_lstm4,  rets_rf4], 
                                           ["mlp",  "cnn", "lstm", "rf"], 
                                           error="mae",
                                           train_val = True)

#%%
visualizations.hparams_optimization_errors([ rets_mlp_adapt, rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp_ap", "mlp",  "cnn", "lstm", "rf"], 
                                           error="mae",
                                           train_val = True)

#%% Best Network Losses and Hyperparameters
res_mlp = visualizations.losses("mlp", 0, "") # best performing network .
res_mlp = visualizations.losses("mlp", 2, "") # network with this architecture on test set.
res_mlp = visualizations.losses("mlp", 3, "") # network with this architecture on preles GPP predictions.
res_mlp = visualizations.losses("mlp", 5, "") # network with this architecture on full simulations (parameters included)
res_mlp = visualizations.losses("mlp", 6, "") # network with this architecture on only climate simulations
res_cnn = visualizations.losses("cnn", 0, "")
res_cnn = visualizations.losses("cnn", 5, "")

#%% Predictions of Best Networks.
y_tests1 = np.load(os.path.join(data_dir,"python\outputs\models\mlp1\y_tests.npy"), allow_pickle=True).tolist()
y_preds1 = np.load(os.path.join(data_dir,"python\outputs\models\mlp1\y_preds.npy"), allow_pickle=True).tolist()
res1 = visualizations.losses("mlp", 1, "")
visualizations.plot_nn_predictions(y_tests1, y_preds1)

#%%

y_tests2 = np.load(os.path.join(data_dir,"python\outputs\models\mlp2\bilykriz\y_tests.npy"), allow_pickle=True).tolist()
y_preds2 = np.load(os.path.join(data_dir,"python\outputs\models\mlp2\\bilykriz\y_preds.npy"), allow_pickle=True).tolist()
res2 = visualizations.losses("mlp", 2, "")

visualizations.plot_nn_predictions(y_tests2, y_preds2)


y_tests_test = np.array(y_preds2).squeeze(2)
