# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:10 2020

@author: marie
"""

import os
import os.path

from setup.utils import minmax_scaler
from setup.utils import minmax_rescaler
import setup.utils as utils

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%% Train the Algorithm

def random_forest_CV(X, Y, splits, shuffled, n_trees, depth, eval_set = None, selected = True):
    
    X_mean, X_std = np.mean(X), np.std(X)
    X = minmax_scaler(X)
    
    # Divide into training and test
    kf = KFold(n_splits=splits, shuffle = shuffled)
    kf.get_n_splits(X)
    regressor = RandomForestRegressor(n_estimators=n_trees, max_depth  = depth, criterion = "mse")

    rmse_train = np.zeros((splits))
    mae_train = np.zeros((splits))
    rmse_val = np.zeros((splits))
    mae_val = np.zeros((splits))

    y_preds = []
    y_trains = []
    y_tests = []

    i = 0
    
    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        if not eval_set is None:
            X_test = eval_set["X_test"]
            y_test = eval_set["Y_test"]
    
        regressor.fit(X_train, y_train.ravel())
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
    

        # Evaluate the algorithm
        rmse_val[i] = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
        mae_val[i] = metrics.mean_absolute_error(y_test, y_pred_test)
        rmse_train[i] = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
        mae_train[i] = metrics.mean_absolute_error(y_train, y_pred_train)
        
        y_preds.append(y_pred_test)
        y_tests.append(y_test)
        y_trains.append(y_train)
        
        i+= 1
    
    if selected:
        errors = [rmse_train, rmse_val, mae_train, mae_val]
    else:
        errors = [np.mean(rmse_train), np.mean(rmse_val), np.mean(mae_train), np.mean(mae_val)]
        
    return(y_preds, y_tests, errors)

#%% Model selection
def rf_selection(X, Y, p_list):

    p_search = []
    
    for i in range(len(p_list[0])):
        
        search = [sublist[i] for sublist in p_list]
        
        y_preds, y_tests, errors = random_forest_CV(X, Y, splits=search[0], shuffled = search[1], n_trees = search[2], depth = search[3])

        p_search.append([item for sublist in [[i], search, errors] for item in sublist])

    results = pd.DataFrame(p_search, columns=["run", "cv_splits", "shuffled", "n_trees", "depth", "rmse_train", "rmse_val", "mae_train", "mae_val"])
        
    print("Best Model Run: \n", results.iloc[results['rmse_val'].idxmin()])
    
    return results.iloc[results['rmse_val'].idxmin()].to_dict() 

#%%
def plot_rf_cv(y_preds, y_tests, rfp, data_dir, figure = "selected", save=True):
    
    fig, ax = plt.subplots(len(y_preds), figsize=(10,9))
    fig.suptitle(f"Random Forest Fit \n (Grown Trees: {rfp['n_trees']}, Max. Tree Depth: {rfp['depth']})")
    for i in range(len(y_preds)):
        ax[i].plot(y_preds[i], color="gray", label="Observations", linewidth=0.8)
        ax[i].plot(y_tests[i].flatten(), color="darkblue", label="Predictions", linewidth=0.8)
        ax[i].plot(y_tests[i].flatten() - y_preds[i], color="lightgreen", label="Absolute Error", linewidth=0.6)
   # for a in ax.flat:
   #     a.set(xlabel="Time [days]", ylabel=r"GPP [g C m$^{-2}$ day$^{-1}$]")
        
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    if save:
        plt.savefig(os.path.join(data_dir, f"plots\data_quality_evaluation\fits_rf\_predictions_{figure}"))
        plt.close()
    
#%%
def plot_rf_fit(fitted, figure = "", data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf"):
    
    fig, ax = plt.subplots(2, figsize=(10,9))
    fig.suptitle(f"Random Forest Fit: {fitted['data']} data \n (Grown Trees: {fitted['n_trees']}, Max. Tree Depth: {fitted['depth']}) \n RMSE Training = {np.round(fitted['rmse_train'], 4)}, RSME Validation = {np.round(fitted['rmse_test'], 4)} \n MAE Training = {np.round(fitted['mae_train'], 4)}, MAE Validation = {np.round(fitted['mae_test'], 4)}")
    ax[0].plot(fitted['Y_train'], color="gray", label="Observations", linewidth=0.8)
    ax[0].plot(fitted['y_pred_train'], color="darkblue", label="Predictions (train)", linewidth=0.8)
    ax[0].plot(fitted['Y_train'].flatten() - fitted['y_pred_train'], color="lightgreen", linewidth=0.6)
    ax[1].plot(fitted['Y_test'], color="gray", linewidth=0.8)
    ax[1].plot(fitted['y_pred_test'], color="lightblue", label="Predictions (test)", linewidth=0.8)
    ax[1].plot(fitted['Y_test'].flatten() - fitted['y_pred_test'], color="lightgreen", label="Error", linewidth=0.6)
    for a in ax.flat:
        a.set(xlabel="Time [days]", ylabel=r"GPP [g C m$^{-2}$ day$^{-1}$]")
    fig.legend(loc="upper left")
    
    plt.savefig(os.path.join(data_dir, f"_predictions_{figure}"))
    plt.close()
