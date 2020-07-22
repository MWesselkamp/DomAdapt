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
def get_profound_data(dataset, data_dir = r'data\profound', to_numpy=True, simulation=False):
    
    """
    
    Args:
        to_numpy (boolean): Default to true. If false, data is returned as Pandas data frame.
        simulation (boolean): Default to false, will return the observed GPP (and SW) values. 
                            If true, the simulations from preles will be returned.
        dataset (char vector): Must take one of the values "full", "test" or "trainval" and specifies, 
                            if the full dataset should be loaded ("full"), or the split version where "trainval" contains 
                            all stands but the "test" stand.
    
    Returns:
        X, Y (array/dataframe)
    """
    
    if simulation:
        filename = r"preles"
    else:
        filename = r"profound"
    
    if(dataset=="full"):
        path_in = os.path.join(data_dir, f"profound_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
    elif(dataset=="test") :
        path_in = os.path.join(data_dir, f"profound_in_test")
        path_out = os.path.join(data_dir, f"{filename}_out_test")
    elif(dataset=="trainval"):
        path_in = os.path.join(data_dir, f"profound_in_trainval")
        path_out = os.path.join(data_dir, f"{filename}_out_trainval")
    else:
        raise ValueError("Don't know dataset.")
    
    X = pd.read_csv(path_in, sep=";")
    if(to_numpy):
        X = X.drop(columns=['date', 'site']).to_numpy()
        Y = pd.read_csv(path_out, sep=";").to_numpy()
    else:
        X = X.drop(columns=['date'])
        Y = pd.read_csv(path_out, sep=";")
    
    return X, Y
    

#%%
def get_borealsites_data(data_dir = r'data\borealsites', to_numpy=True, preles=True):
    
    filename = r"boreal_sites"
    
    path_in = os.path.join(data_dir, f"{filename}_in")
    if(preles):
        path_out = os.path.join(data_dir, 'preles_out')
    else:
        path_out = os.path.join(data_dir, f"{filename}_out")
    
    X = pd.read_csv(path_in, sep=";")
    if(to_numpy):
        X = X.drop(columns=['site']).to_numpy()
        Y = pd.read_csv(path_out, sep=";").drop(columns=['ET']).to_numpy()
    else:
        Y = pd.read_csv(path_out, sep=";").drop(columns=['ET'])
        
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
