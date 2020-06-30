# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:35:10 2020

@author: marie
"""
import os
import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_data(data_dir = 'data\preles\exp'):
        
    data = {}

    filesnum = int(len([name for name in os.listdir(data_dir)])/2)
    filenames = [f'sim{i}' for i in range(1,filesnum+1)]

    for filename in filenames:
        data[filename] = [0,0]
        path_in = os.path.join(data_dir, f"{filename}_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
        data[f"{filename}"][0] = pd.read_csv(path_in, sep=";").drop(columns=['date']).to_numpy()
        data[f"{filename}"][1] = pd.read_csv(path_out, sep=";").to_numpy()
        
    return data, filenames

def merge_XY(data):
    """
    This function concatenates the target and the feature values to one large np.array.
    These are again saved in a dictionary ("sim1",...).
    
    Args:
        data(dict): requires an input dictionary as returned by function get_data(). 
                    This should again contain dictonaries of length two, containung an array X, 
                    representing features and an array y, representing labels.
                    
    Returns:
        data(dict): filled with one array per simulation.
    """
    data = {sim[0]: np.concatenate([v for k,v in sim[1].items()], 1) for sim in data.items()}
    return data

def minmax_scaler(data, nvars):
    """
    This function scales all features in an array between -1 and 1. If nvars equals
    the number of features, shape of the returned array will remain the same.
    
    Args:
        data(np.array): two dimensional array containing model features.
        nvars(int): number of features or new shape of array.
        
    Returns:
        data_norm(np.array): two dimensional array of scaled model features.
    """
    scaler = MinMaxScaler(feature_range = (-1,1))
    data_norm = scaler.fit_transform(data.reshape(-1,nvars))
    return data_norm


def flatten(t):
    # the second argument can be anything and -1 tells function to figure out the shape of the tensor.
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t