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
X, Y = preprocessing.get_profound_data(data_dir = 'data\profound', ignore_env = True, preles=True)

X = preprocessing.normalize_features(X)

#%% Plot the data
plt.plot(Y)
# Use only GPP
Y = Y[:,0]

#%% Divide into training and test
splits = 4
kf = KFold(n_splits=splits, shuffle = True)
kf.get_n_splits(X)

#%% Train the Algorithm
regressor = RandomForestRegressor(n_estimators=50, max_depth  = 6, criterion = "mse")

fig, axs = plt.subplots(4)
mse = np.zeros((splits))
mpe = np.zeros((splits))

i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test)
    
    yrange = np.ptp(y_train, axis=0)
    # Evaluate the algorithm
    mse[i] = metrics.mean_squared_error(y_test, y_pred)
    mpe[i] = utils.percentage_error(y_test, y_pred, y_range=yrange)
    
    axs[i].plot(y_test, color="black", label="y_test")
    axs[i].plot(y_pred, color="red", label="y_pred")
    
    i+= 1
    
print('Mean Percentage Error:', np.mean(mpe))
print('Mean Squared Error:', np.mean(mse))

fig.suptitle("Profound input and PREles GPP simulations")
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')