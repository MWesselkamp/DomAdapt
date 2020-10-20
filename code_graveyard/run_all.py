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

import pandas as pd
 
import visualizations

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
#%%
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(datadir, "data\preles\simulations"))
#%% Grid search of hparams
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
rets_cnn = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\cnn\grid_search_results_cnn1.csv"))
rets_cnn2 = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\cnn\grid_search_results_cnn1_2.csv"))
rets_lstm = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\lstm\grid_search_results_lstm1.csv"))
rets_rf = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\rf\grid_search_results_rf1.csv"))

adict = rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()
rets_cnn.iloc[rets_cnn['rmse_val'].idxmin()].to_dict()
rets_lstm.iloc[rets_lstm['rmse_val'].idxmin()].to_dict()
rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
#%%
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="rmse",
                                           train_val = True)
#%%
visualizations.hparams_optimization_errors([rets_mlp, rets_cnn, rets_lstm, rets_rf], 
                                           ["mlp", "cnn", "lstm", "rf"], 
                                           error="mae",
                                           train_val = True)

#%%

rets_mlp4 = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\mlp\grid_search_results_mlp4.csv"))
rets_rf4 = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\rf\grid_search_results_rf4.csv"))

visualizations.hparams_optimization_errors([rets_rf, rets_mlp, rets_rf4, rets_mlp4], 
                                           ["rf", "mlp", "rf4", "mlp4"], 
                                           train_val = True)
