# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:01:38 2020

@author: marie
"""

#%%
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import numpy as np
import pandas as pd

import os.path
import visualizations

import collect_results
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

#%%
rets_sparse1_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse1_mlp2.csv"))
rets_sparse2_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse2_mlp2.csv"))
rets_sparse3_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse3_mlp2.csv"))
rets_sparse4_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse4_mlp2.csv"))
rets_sparse5_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse5_mlp2.csv"))

visualizations.hparams_optimization_errors([rets_sparse1_mlp, rets_sparse2_mlp, rets_sparse3_mlp, rets_sparse4_mlp, rets_sparse5_mlp], 
                                           ["1 year", "2 years",  "3 years", "4 years",  "5 years"], 
                                           error="mae",
                                           train_val = True)

#%%
rets_sparse1_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse1_mlp2.csv"))
rets_sparse_r1_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse_r1_mlp2.csv"))

visualizations.hparams_optimization_errors([rets_sparse1_mlp, rets_sparse_r1_mlp], 
                                           ["1 year", "1 random year"], 
                                           error="mae",
                                           train_val = True)

#%%
visualizations.losses("mlp", 0, r"sparse1", sparse=True)
visualizations.losses("mlp", 0, r"sparse2", sparse=True)
visualizations.losses("mlp", 0, r"sparse3", sparse=True)
visualizations.losses("mlp", 0, r"sparse4", sparse=True)
visualizations.losses("mlp", 0, r"sparse5", sparse=True)
#%%
visualizations.losses("mlp", 2, r"sparse1", sparse=True)
visualizations.losses("mlp", 2, r"sparse2", sparse=True)
visualizations.losses("mlp", 2, r"sparse3", sparse=True)
visualizations.losses("mlp", 2, r"sparse4", sparse=True)
visualizations.losses("mlp", 2, r"sparse5", sparse=True)


#%% full backprob and frozen weights.

l = visualizations.losses("mlp", 6, r"sparse1/setting1", sparse=True)
l = visualizations.losses("mlp", 7, r"sparse1/setting1", sparse=True)
l = visualizations.losses("mlp", 8, r"sparse1/setting1", sparse=True)

l = visualizations.losses("mlp", 6, r"sparse2/setting1", sparse=True)
l = visualizations.losses("mlp", 7, r"sparse2/setting1", sparse=True)
l = visualizations.losses("mlp", 8, r"sparse2/setting1", sparse=True)

l = visualizations.losses("mlp", 6, r"sparse3/setting1", sparse=True)
l = visualizations.losses("mlp", 7, r"sparse3/setting1", sparse=True)
l = visualizations.losses("mlp", 8, r"sparse3/setting1", sparse=True)
#%%

sp_df = collect_results.sparse_networks_results(sparses = [1,2,3,4,5])

#%%
import matplotlib.pyplot as plt
def plot1(colors = ["blue", "red"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[sp_df.loc[(sp_df.task =="sparse_selected")]["mae_val"]],
                [sp_df.loc[(sp_df.task =="sparse_finetuning") & (sp_df.finetuned_type == "C-OLS")]["mae_val"]]]
    yi = [[sp_df.loc[(sp_df.task =="sparse_selected")]["mae_val"]],
                [sp_df.loc[(sp_df.task =="sparse_finetuning") & (sp_df.finetuned_type == "C-OLS")]["rmse_val"]]]
    
    m = ['o', "*"]
    s = [60, 200]
    labs = ["selected", "OLS"]
    for i in range(len(xi)):
        if log:
            plt.scatter(np.log(xi[i]), np.log(yi[i]), alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Log(Mean Absolute Error)")
            plt.ylabel("Log(Root Mean Squared Error)")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error")
            plt.ylabel("Root Mean Squared Error")
        plt.legend(loc="lower right")
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)    
#%%
plot1()
#%%
import finetuning
import matplotlib.pyplot as plt

predictions_test, errors = finetuning.featureExtractorC("mlp", 10, None, 50, classifier = "ols", 
                      years = [2005, 2006], sparse=2)

preds = np.array(predictions_test).squeeze(2)
plt.plot(np.transpose(preds))

errors = np.mean(np.array(errors), axis=1)
