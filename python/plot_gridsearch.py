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
rets_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\mlp\grid_search_results_mlp2.csv"))
rets_mlp_adapt = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\mlp\AdaptPool\grid_search_results_mlp2.csv"))
rets_mlp4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\mlp\grid_search_results_mlp4.csv"))
rets_cnn = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\cnn\grid_search_results_cnn2.csv"))
rets_cnn4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\cnn\grid_search_results_cnn4.csv"))
rets_lstm = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\lstm\grid_search_results_lstm2.csv"))
rets_lstm4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\lstm\grid_search_results_lstm4.csv"))
rets_rf = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\rf\grid_search_results_rf2.csv"))
rets_rf4 = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\rf\grid_search_results_rf4.csv"))
#%% MAE and RMSE of Grid search
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm,  rets_rf], 
                                           ["MLP",  "CNN", "LSTM", "RF"], 
                                           error="mae",
                                           train_val = True)
#%% MAE and RMSE of Grid search
visualizations.hparams_optimization_errors([rets_mlp4, rets_cnn4, rets_lstm4,  rets_rf4], 
                                           ["MLP",  "CNN", "LSTM", "RF"], 
                                           error="mae",
                                           train_val = True)
#%%
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf,  rets_mlp_adapt], 
                                           ["MLP",  "CNN", "LSTM", "RF", "MLP_AP"], 
                                           error="mae",
                                           train_val = True)

#%%
rets_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\simulations\grid_search_results_mlp2_adaptPool.csv"))
rets_cnn = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\simulations\grid_search_results_cnn2.csv"))
rets_rf = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\simulations\grid_search_results_rf2.csv"))
rets_lstm = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\simulations\grid_search_results_lstm2.csv"))

visualizations.hparams_optimization_errors([rets_mlp, rets_rf, rets_cnn, rets_lstm], 
                                           ["mlp", "rf", "cnn", "lstm"], 
                                           error="mae",
                                           train_val = True)
#%% load performance of best models

best_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\models\mlp0\AdaptPool\dropout\selected_results.csv"))
best_cnn = pd.read_csv(os.path.join(data_dir, r"python\outputs\models\cnn0\selected_results.csv"))
best_lstm = pd.read_csv(os.path.join(data_dir, r"python\outputs\models\lstm0\selected_results.csv"))


