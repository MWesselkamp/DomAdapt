# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing
import random_forest
from experiment import train_model_CV
from experiment import plot_nn_loss
from experiment import plot_nn_predictions
from experiment import plot_prediction_error

import random 
import torch.nn as nn
import pandas as pd
import time

#%% Load Data
X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

#%% Fit Random Forest
random_forest.random_forest_CV(X, Y, splits=5, shuffled = False)
random_forest.random_forest_CV(X, Y, splits=5, shuffled = True)

fitted_rf = random_forest.random_forest_fit(X, Y, data = "profound")

#%% Random Grid search
def random_grid_search(X, Y, hp_list, searchsize):
    
    hp_search = []
    in_features = X.shape[1]
    out_features = Y.shape[1]

    for i in range(searchsize):
        
        search = [random.choice(sublist) for sublist in hp_list]

        # Network training
        hparams = {"batchsize": search[1], 
                   "epochs":1000, 
                   "history":search[3], 
                   "hiddensize":search[0],
                   "optimizer":"adam", 
                   "criterion":"mse", 
                   "learningrate":search[2],
                   "shuffled_CV":False}

        model_design = {"dimensions": [in_features, search[0], out_features],
                        "activation": nn.Sigmoid}
   
        start = time.time()
        running_losses,performance, predictions = train_model_CV(hparams, model_design, X, Y, splits=6)
        end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
        hp_search.append([item for sublist in [[i, (end-start)], search, performance] for item in sublist])

        plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], data="profound", history = hparams["history"], figure = i, hparams = hparams)
        plot_nn_predictions(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
        plot_prediction_error(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")

    results = pd.DataFrame(hp_search, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r'plots\data_quality_evaluation\fits_nn\grid_search_results.csv', index = False)
    
    print("Best Model Run: \n", results.iloc[results['RSME_val'].idxmin()])    
    
    return(results.iloc[results['RSME_val'].idxmin()].to_dict())

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256]
batchsize = [2, 8, 64, 128, 256]
learningrate = [7e-3, 1e-2, 3e-2, 8e-2]
history = [0,1,2]
hp_list = [hiddensize, batchsize, learningrate, history]

best_model = random_grid_search(X, Y, hp_list, searchsize=2)

