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
    This function scales all features in an array between mean and standard deviation. 
    
    Args:
        data(np.array): two dimensional array containing model features.
        
    Returns:
        data_norm(np.array): two dimensional array of scaled model features.
    """
    #scaler = MinMaxScaler(feature_range = (-1,1))
    data_norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    
    return data_norm


def encode_doy(doy):
    """Encode the day of the year on a circle.
    
    Thanks to: Philipp Jund.
    
    """
    doy_norm = doy / 365 * 2 * np.pi
    return np.sin(doy_norm), np.cos(doy_norm)

def percentage_error(targets, predictions, y_range):
    
    #p = (targets!=0).ravel()
    #targets, predictions = targets[p], predictions[p]
    pe = np.mean(np.abs((targets-predictions)/y_range))*100
    
    return pe

def nash_sutcliffe(targets, predictions):
    
    nash = 1-np.sum(np.square(predictions-targets) / np.square(targets - np.mean(targets)))
    
    return nash

