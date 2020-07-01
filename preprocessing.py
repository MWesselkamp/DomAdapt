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

import utils

def get_data(data_dir = 'data\preles\exp'):

    filesnum = int(len([name for name in os.listdir(data_dir)])/2)
    filenames = [f'sim{i}' for i in range(1,filesnum+1)]
    
    X = [None]*filesnum
    y = [None]*filesnum
    
    for i in range(filesnum):
        filename = filenames[i]
        path_in = os.path.join(data_dir, f"{filename}_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
        X[i] = pd.read_csv(path_in, sep=";").drop(columns=['date']).to_numpy()
        y[i] = pd.read_csv(path_out, sep=";").to_numpy()
        
    return X, y#, filenames



#%% Normalize Features
def normalize_features(X):
    X = [utils.minmax_scaler(data) for data in X]
    
    return X

#%% Restructure Data 
def to_batches(X,y, size=20):
    
    X = [np.dstack([inputs[(i-size):i] for i in range(size, inputs.shape[0]-1)]) for inputs in X]
    y = [np.dstack([labels[i+1] for i in range(size, labels.shape[0]-1)]) for labels in y]

    return X, y
