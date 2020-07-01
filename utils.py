# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:35:10 2020

@author: marie
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

def minmax_scaler(data):
    """
    This function scales all features in an array between -1 and 1. 
    
    Args:
        data(np.array): two dimensional array containing model features.
        
    Returns:
        data_norm(np.array): two dimensional array of scaled model features.
    """
    scaler = MinMaxScaler(feature_range = (-1,1))
    data_norm = scaler.fit_transform(data)
    return data_norm

