# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:10 2020

@author: marie
"""
#%% Set working directory
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing
import utils
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np

#%% Load data
X_profound, Y_profound = preprocessing.get_profound_data(data_dir = r'data\profound', ignore_env = True, preles=False)
X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', ignore_env = True, preles=False)

# Merge profound and preles data into one large data set.
X = np.concatenate((X_profound, X_borealsites), axis=0)
Y = np.concatenate((Y_profound, Y_borealsites), axis=0)

X = preprocessing.normalize_features(X)

#%% Plot the data
plt.plot(Y)
# Use only GPP
#Y = Y[:,0].reshape(-1,1)

#%% Divide into training and test
splits = 10
kf = KFold(n_splits=splits, shuffle = True)
kf.get_n_splits(X)

#%% Train the Algorithm
regressor = RandomForestRegressor(n_estimators=500, max_depth  = 5, criterion = "mse")

fig, axs = plt.subplots(splits)
mse = np.zeros((splits))
rmse = np.zeros((splits))
mpe = np.zeros((splits))
nash = np.zeros((splits))

i = 0
for train_index, test_index in kf.split(X):
    
    print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test)
    
    yrange = np.ptp(y_train, axis=0)
    # Evaluate the algorithm
    mse[i] = metrics.mean_squared_error(y_test, y_pred)
    rmse[i] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mpe[i] = utils.percentage_error(y_test, y_pred, y_range=yrange)
    nash[i] = utils.nash_sutcliffe(y_test, y_pred)
    
    print('Mean Percentage Error:', mpe[i])
    print('Mean Squared Error:', mse[i])
    print('Root Mean Squared Error:', rmse[i])
    print('Nash-Sutcliffe:', nash[i])

    axs[i].plot(y_test, color="black", label="y_test")
    axs[i].plot(y_pred, color="red", label="y_pred")
    
    i+= 1
    
print('Mean Percentage Error:', np.mean(mpe))
print('Mean Squared Error:', np.mean(mse))
print('Root Mean Squared Error:', np.mean(rmse))
print('Nash-Sutcliffe:', np.mean(nash))

fig.suptitle(f"Profound/borealsites input and output. \nRandomized  {splits}-fold CV. \n MPE = {np.round(np.mean(mpe), 4)} \n RMSE = {np.round(np.mean(rmse), 4)}")
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='bottom center')