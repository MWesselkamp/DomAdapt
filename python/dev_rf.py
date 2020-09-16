# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:10 2020

@author: marie
"""

import os
import os.path

from utils import minmax_scaler
from utils import minmax_rescaler

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% Train the Algorithm

def random_forest_CV(X, Y, splits, shuffled, n_trees, depth, selected = False):
    
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
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
    
        regressor.fit(X_train, y_train.ravel())
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
    

        # Evaluate the algorithm
        rmse_val[i] = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
        mae_val[i] = metrics.mean_absolute_error(y_test, y_pred_test)
        rmse_train[i] = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
        mae_train[i] = metrics.mean_absolute_error(y_train, y_pred_train)
        
        y_train = minmax_rescaler(y_train, mu = Y_mean, sigma = Y_std)
        y_test = minmax_rescaler(y_test, mu = Y_mean, sigma = Y_std)
    
        y_pred_train = minmax_rescaler(y_pred_train, mu = Y_mean, sigma = Y_std)
        y_pred_test = minmax_rescaler(y_pred_test, mu = Y_mean, sigma = Y_std)
    
        y_preds.append(y_pred_test)
        y_tests.append(y_test)
        y_trains.append(y_train)
        
        i+= 1
    
    if selected:
        losses = {"rmse_train":rmse_train, "rmse_val":rmse_val, "mae_train":mae_train, "mae_val":mae_val}
    else:
        losses = [np.mean(rmse_train), np.mean(rmse_val), np.mean(mae_train), np.mean(mae_val)]
        
    return(y_preds, y_tests, losses)

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
#%% Model selection parallel

def rf_selection_parallel(X, Y, p_list, searchsize, q, p_search=[]):


    search = [sublist[searchsize] for sublist in p_list]
        
    y_preds, y_tests, errors = random_forest_CV(X, Y, splits=search[0], shuffled = search[1], n_trees = search[2], depth = search[3])

    p_search.append([item for sublist in [[searchsize], search, errors] for item in sublist])
    
    print("Rf fitted")
    
    q.put(p_search)

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


#%%
def random_forest_fit(X, Y, rfp, data="profound"):
    
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

    regressor = RandomForestRegressor(n_estimators=rfp['n_trees'], max_depth  = rfp['depth'], criterion = "mse")
    
    regressor.fit(X_train, Y_train.ravel())

    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    
    # Evaluate the algorithm
    rmse_train=np.sqrt(metrics.mean_squared_error(Y_train, y_pred_train))
    rmse_test=np.sqrt(metrics.mean_squared_error(Y_test, y_pred_test))
    mae_train=metrics.mean_absolute_error(Y_train, y_pred_train)
    mae_test=metrics.mean_absolute_error(Y_test, y_pred_test)
    
    Y_train = minmax_rescaler(Y_train, mu = Y_mean, sigma = Y_std)
    Y_test = minmax_rescaler(Y_test, mu = Y_mean, sigma = Y_std)
    
    y_pred_train = minmax_rescaler(y_pred_train, mu = Y_mean, sigma = Y_std)
    y_pred_test = minmax_rescaler(y_pred_test, mu = Y_mean, sigma = Y_std)

    fitted = {"data":data, 
              "depth":rfp['depth'], 
              "n_trees":rfp['n_trees'],
              "y_pred_train":y_pred_train,
              "y_pred_test":y_pred_test, 
              "Y_train":Y_train,
              "Y_test":Y_test,
              "rmse_train":rmse_train, 
              "rmse_test":rmse_test, 
              "mae_train":mae_train, 
              "mae_test":mae_test,
              }
    
    plot_rf_fit(fitted)
    
    return fitted




