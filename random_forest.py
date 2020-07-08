# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:10 2020

@author: marie
"""

import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import pandas as pd
import numpy as np
#%% Load data
X, Y = preprocessing.get_profound_data(data_dir = 'data\profound', ignore_env = True)

X = preprocessing.normalize_features(X)

#%% Divide into training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train the Algorithm

regressor = RandomForestRegressor(n_estimators=20, max_depth  = 4, criterion = "mse")
regressor.fit(X_train, y_train.ravel())
y_pred = regressor.predict(X_test)

# Evaluate the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
