# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:42:47 2020

@author: marie
"""
#%%
import numpy as np
import pandas as pd
import os.path
import visualizations

#%%

def model_performance(model, typ, suptitle):

    datadir = os.path.join("OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models", f"{model}{typ}")

    results = pd.read_csv(os.path.join(datadir, "selected_results.csv"))
    print(results)
    running_losses = np.load(os.path.join(datadir,"running_losses.npy"), allow_pickle=True).item()
    #y_tests = np.load(os.path.join(datadir,"y_tests.npy"), allow_pickle=True).tolist()
    #y_preds = np.load(os.path.join(datadir,"y_preds.npy"), allow_pickle=True).tolist()

    visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], suptitle, "mlp")
    #visualizations.plot_nn_predictions(y_tests, y_preds)
    #return(y_tests,y_preds)
    return(results)
#%%
out = model_performance("mlp", 2) # evaluation on test set
res_mlp = model_performance("mlp", 1, "MLP") # 
res_mlp = model_performance("mlp", 3, "MLP") # evaluation on preles simulations

res_cnn = model_performance("cnn", 1, "CNN")

