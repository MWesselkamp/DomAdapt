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

#%% Train the Algorithm

def random_forest_CV(X, Y, splits, shuffled, n_trees, depth):
    
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
    # Divide into training and test
    kf = KFold(n_splits=splits, shuffle = shuffled)
    kf.get_n_splits(X)
    regressor = RandomForestRegressor(n_estimators=n_trees, max_depth  = depth, criterion = "mse")

    rmse_train = np.zeros((splits))
    mae_train = np.zeros((splits))
    rmse_test = np.zeros((splits))
    mae_test = np.zeros((splits))

    y_preds = []
    y_trains = []
    y_tests = []

    i = 0
    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_tests.append(y_test)
        y_trains.append(y_train)
    
        regressor.fit(X_train, y_train.ravel())
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        y_preds.append(y_pred_test)
    

        # Evaluate the algorithm
        rmse_test[i] = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
        mae_test[i] = metrics.mean_absolute_error(y_test, y_pred_test)
        rmse_train[i] = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
        mae_train[i] = metrics.mean_absolute_error(y_train, y_pred_train)
        
        i+= 1
    
    return([np.mean(rmse_train), np.mean(rmse_test), np.mean(mae_train), np.mean(mae_test)])

#%%
def plot_rf_fit(Y_train, Y_test, fitted, figure = "", data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf"):
    
    Y_train = minmax_rescaler(Y_train, mu = fitted['Y_mean'], sigma = fitted['Y_std'])
    Y_test = minmax_rescaler(Y_test, mu = fitted['Y_mean'], sigma = fitted['Y_std'])
    
    y_pred_train = minmax_rescaler(fitted['y_pred_train'], mu = fitted['Y_mean'], sigma = fitted['Y_std'])
    y_pred_test = minmax_rescaler(fitted['y_pred_test'], mu = fitted['Y_mean'], sigma = fitted['Y_std'])
    
    fig, ax = plt.subplots(2, figsize=(10,9))
    fig.suptitle(f"Random Forest Fit: {fitted['data']} data \n (Grown Trees: {fitted['n_trees']}, Max. Tree Depth: {fitted['depth']}) \n RMSE Training = {np.round(fitted['rmse_train'], 4)}, RSME Validation = {np.round(fitted['rmse_test'], 4)} \n MAE Training = {np.round(fitted['mae_train'], 4)}, MAE Validation = {np.round(fitted['mae_test'], 4)}")
    ax[0].plot(Y_train, color="gray", label="Observations", linewidth=0.8)
    ax[0].plot(y_pred_train, color="darkblue", label="Predictions (train)", linewidth=0.8)
    ax[0].plot(Y_train.flatten() - y_pred_train, color="lightgreen", linewidth=0.6)
    ax[1].plot(Y_test, color="gray", linewidth=0.8)
    ax[1].plot(y_pred_test, color="lightblue", label="Predictions (test)", linewidth=0.8)
    ax[1].plot(Y_test.flatten() - y_pred_test, color="lightgreen", label="Error", linewidth=0.6)
    for a in ax.flat:
        a.set(xlabel="Time [days]", ylabel=r"GPP [g C m$^{-2}$ day$^{-1}$]")
    fig.legend(loc="upper left")
    
    plt.savefig(os.path.join(data_dir, f"{fitted['data']}_predictions_{figure}"))
    plt.close()


#%%
def random_forest_fit(X, Y, shuffled, n_trees, depth, data):
    
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=shuffled)

    regressor = RandomForestRegressor(n_estimators=n_trees, max_depth  = depth, criterion = "mse")
    
    regressor.fit(X_train, Y_train.ravel())

    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    
    # Evaluate the algorithm

    fitted = {"data":data, 
              "depth":depth, 
              "n_trees":n_trees,
              "y_pred_train":y_pred_train,
              "y_pred_test":y_pred_test, 
              "rmse_train":np.sqrt(metrics.mean_squared_error(Y_train, y_pred_train)), 
              "rmse_test":np.sqrt(metrics.mean_squared_error(Y_test, y_pred_test)), 
              "mae_train":metrics.mean_absolute_error(Y_train, y_pred_train), 
              "mae_test":metrics.mean_absolute_error(Y_test, y_pred_test),
              "Y_mean":Y_mean,
              "Y_std":Y_std}
    
    plot_rf_fit(Y_train, Y_test,fitted)
    
    return fitted

    

