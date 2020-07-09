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

import pandas as pd
import numpy as np
#%% Load data
X, Y = preprocessing.get_profound_data(data_dir = 'data\profound', ignore_env = True)

X = preprocessing.normalize_features(X)

#%% Divide into training and test
kf = KFold(n_splits=5, shuffle = True)
kf.get_n_splits(X)

#%% Train the Algorithm
regressor = RandomForestRegressor(n_estimators=50, max_depth  = 6, criterion = "mse")
yrange = np.ptp(Y, axis=0)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test)
    # Evaluate the algorithm
    print('Mean Percentage Error:', utils.percentage_error(y_test, y_pred, y_range=yrange))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))