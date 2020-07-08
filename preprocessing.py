# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:55:08 2020

@author: marie
"""

#%% Input normalization
import os
import os.path
import pandas as pd
import numpy as np

import random
from sklearn.utils import shuffle
from math import floor

import utils

#%%
def get_profound_data(data_dir = 'data\profound', ignore_env=True):
    
    filenames = [name for name in os.listdir(data_dir)]
    
    path_in = os.path.join(data_dir, filenames[0])
    path_out = os.path.join(data_dir, filenames[1])
    X = pd.read_csv(path_in, sep=";")
    if(ignore_env):
        X = X.drop(columns=['date', 'site']).to_numpy()
    Y = pd.read_csv(path_out, sep=";").to_numpy()
    
    return X, Y
    
#%%
def get_simulations(data_dir = 'data\preles\exp', ignore_env = True):

    filesnum = int(len([name for name in os.listdir(data_dir)])/2)
    filenames = [f'sim{i}' for i in range(1,filesnum+1)]
    
    X = [None]*filesnum
    Y = [None]*filesnum
    
    for i in range(filesnum):
        filename = filenames[i]
        path_in = os.path.join(data_dir, f"{filename}_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
        X[i] = pd.read_csv(path_in, sep=";")
        if(ignore_env):
            X[i] = X[i].drop(columns=['date']).to_numpy()
        Y[i] = pd.read_csv(path_out, sep=";").to_numpy()
        
    return X, Y#, filenames


#%% Normalize Features
def normalize_features(X):
    
    if (X.ndim > 2):
        X = [utils.minmax_scaler(data) for data in X]
    else:
        X = utils.minmax_scaler(X)
    
    return X

#%% Restructure Data 
def to_batches(X,Y, size=20):
    
    X = [np.dstack([inputs[(i-size):i] for i in range(size, inputs.shape[0]-1)]) for inputs in X]
    Y = [np.dstack([labels[i+1] for i in range(size, labels.shape[0]-1)]) for labels in Y]

    return X, Y

#%% Split into training and test data.
def split_data(X, Y, size=0.5):
    random.seed(42)
    x, y = shuffle(X, Y)
    d = floor(len(x)*size)
    x_train, ytrain = x[:d], y[:d]
    x_test, y_test = x[d:], y[d:]
    return x_train, ytrain, x_test, y_test
