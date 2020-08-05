# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import utils
import preprocessing

from random_forest import random_forest_CV
from random_forest import random_forest_fit
import pandas as pd
import os.path

from experiment import nn_selection


#%%

import matplotlib.pyplot as plt

def plot_validation_errors(results, model, data = "profound"):
    
    if model == "RandomForest":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf"
    elif model == "MLP":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn"
    else:
        raise ValueError("Don't know model or: To directory should the plot be saved?")

    fig, ax = plt.subplots()
    fig.suptitle(f"{model} Hyperparameter Optimization: \n Validation Errors \n ({data} data)")
    ax.scatter(ret["rmse_val"], ret["mae_val"])

    for i, txt in enumerate(ret["run"]):
        ax.annotate(txt, (ret["rmse_val"][i], ret["mae_val"][i]))

    plt.xlabel("RMSE Validation")
    plt.ylabel("MAE Validation")
    
    plt.savefig(os.path.join(data_dir, f"{model}_validation_errors_{data}"))
    plt.close()



#%% Load Data
X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

#%% Fit Random Forest
cv_splits = [5]
shuffled = [True, False]
n_trees = [200,300]
depth = [4,5]

p_list = utils.expandgrid(cv_splits, shuffled, n_trees, depth )


def rf_selection(X, Y, p_list):

    p_search = []
    
    for i in range(len(p_list[0])):
        
        search = [sublist[i] for sublist in p_list]
        
        results = random_forest_CV(X, Y, splits=search[0], shuffled = search[1], n_trees = search[2], depth = search[3])

        p_search.append([item for sublist in [[i], search, results] for item in sublist])

    results = pd.DataFrame(p_search, columns=["run", "cv_splits", "shuffled", "n_trees", "depth", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf\grid_search_results.csv', index = False)
    
    print("Best Model Run: \n", results.iloc[results['rmse_val'].idxmin()])
    
    return results 

ret = rf_selection(X, Y, p_list)
best_fit = ret.iloc[ret['rmse_val'].idxmin()].to_dict()
fitted_rf = random_forest_fit(X, Y, best_fit["shuffled"], best_fit["n_trees"], best_fit["depth"],data = "profound")
    
plot_validation_errors(ret, "RandomForest")
#%% Grid search of hparams
hiddensize = [16, 64, 128, 256]
batchsize = [2, 8, 64, 128, 256]
learningrate = [7e-3, 1e-2, 3e-2, 8e-2]
history = [0,1,2]
hp_list = [hiddensize, batchsize, learningrate, history]

best_model = nn_selection(X, Y, hp_list, searchsize=2)

