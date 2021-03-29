# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:35:10 2020

@author: marie
"""
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import setup.models as models
import setup.preprocessing as preprocessing
import os.path
from ast import literal_eval
from sklearn import metrics

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

def minmax_scaler(data, scaling = None):
    """
    This function scales all features in an array between mean and standard deviation. 
    
    Args:
        data(np.array): two dimensional array containing model features.
        
    Returns:
        data_norm(np.array): two dimensional array of scaled model features.
    """
    #scaler = MinMaxScaler(feature_range = (-1,1))
    if (scaling is None):
        
        if (isinstance(data, pd.DataFrame)):
            data_norm = (data - data.mean())/ data.std()
        elif (torch.is_tensor(data)):
            data_norm = (data - torch.mean(data)) / torch.std(data)
        else:
            data_norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    else: 
        data_norm = (data - scaling[0])/scaling[1]
        
    return data_norm

def minmax_rescaler(data, mu, sigma):
    
    """
    Rescale data to original values with mean and standard deviation.
    """
    
    data = data*sigma + mu
    
    return(data)

def encode_doy(doy):
    """Encode the day of the year on a circle.
    
    Thanks to: Philipp Jund.
    
    """
    doy_norm = doy / 365 * 2 * np.pi
    return np.sin(doy_norm), np.cos(doy_norm)

#%% 
def create_batches(X, Y, batchsize, history):
    
    """
    Creates Mini-batches from training data set.
    Used in: dev_mlp.train_model_CV
    """
    
    subset = [j for j in random.sample(range(X.shape[0]), batchsize) if j > history]
    subset_h = [item for sublist in [list(range(j-history,j)) for j in subset] for item in sublist]
    x = np.concatenate((X[subset], X[subset_h]), axis=0)
    y = np.concatenate((Y[subset], Y[subset_h]), axis=0)
    
    return x, y

def expandgrid(*itrs):
    """
    Expand lists to grid.
    Thanks to:
        https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python
    Used in: ms_parallel_[any]
    """
    product = list(itertools.product(*itrs))
    return [[x[i] for x in product] for i in range(len(itrs))]

def num_infeatures(dim_channels, kernel_size, length, stride=1):
    
    """
    Computes the number of input features for linear layer after 1d convolution. (No padding!)
    Used in: models.ConvN
    """
    
    linear_in = dim_channels[-1]*(length-len(dim_channels)*((kernel_size-stride)+(2-1)))
    
    return(linear_in)

def reshaping(X, seqlen, model):
    
    """
    Reshapes 2d Torch-Tensor to 3d Torch-Tensor with Minibatches of sequence length L.
    Used in: dev_convnet.train_model_CV
    
    """
    batchsize = X.shape[0]-seqlen
    
    if model=="lstm":
        x_out = torch.empty((seqlen, batchsize, X.shape[1]))
    
        for i in range(batchsize):
            x_out[:,i,:] = X[i:i+seqlen]
    else:
        x_out = torch.empty((batchsize, X.shape[1], seqlen))
    
        for i in range(batchsize):
            x_out[i,:,:] = X[i:i+seqlen].transpose(0,1)
        
    return x_out

def create_inout_sequences(x, y, batchsize, seqlen, model):
    
    """
    In-out Sequences for LSTM.
    Used in: dev_lstm.train_model_CV
    """
    if batchsize == "full":
        batchsize = x.shape[0]-seqlen-1
        batches = range(batchsize)
    else:
        batches = random.sample(range(x.shape[0]-1-seqlen), batchsize) 
    
    y_out = torch.empty((batchsize, y.shape[1]))
    
    if model=="lstm":
        
        x_out = torch.empty(( seqlen, batchsize, x.shape[1]))
        for i in range(batchsize):
        
            x_out[:,i,:] = x[batches[i]:batches[i]+seqlen]
            y_out[i,:] = y[batches[i]+seqlen+1]
    else:
        x_out = torch.empty((batchsize, x.shape[1], seqlen))
    
        for i in range(batchsize):
        
            x_out[i,:,:] = x[batches[i]:batches[i]+seqlen].transpose(0,1)
            y_out[i,:] = y[batches[i]+seqlen+1]
        
    return x_out, y_out


def rmse(targets, predictions):
    
    """
    Computes the Root Mean Squared Error.
    
    Args:
        targets (torch.tensor)
        predictions (torch.tensor)
    """
    if torch.is_tensor(targets):
        rmse = np.sqrt(np.mean(np.square(targets-predictions).numpy()))
    else:
        rmse = np.sqrt(np.mean(np.square(targets-predictions)))
    
    return rmse

def error_in_percent(error,
                     test_error = 2.8394):
    
    percent = np.round(((test_error*100)/(error*100))**(-1)*100,2)
    
    return percent

def settings(typ, 
             years = [2001,2002,2003, 2004, 2005, 2006, 2007],
             data_dir =  "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                    years = years,
                                    datadir = os.path.join(data_dir, "data"), 
                                    dataset = "profound",
                                    simulations = None)
    X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                              years = [2008],
                                              datadir = os.path.join(data_dir, "data"), 
                                              dataset = "profound",
                                              simulations = None)
    
    if ((typ == 5) | (typ == 13) | (typ==14)) :
      
        Xf = np.zeros((X.shape[0], 12))
        Xf_test = np.zeros((X_test.shape[0], 12))
        Xf[:,:7] = X
        X = Xf
        Xf_test[:,:7] = X_test
        X_test = Xf_test
        
        
    if ((typ == 6) | (typ==8)) :
        #gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\simulations\grid_search_results_{model}2_adaptPool.csv"))
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\simulations\\7features\grid_search_results_mlp2_np.csv"))
    elif ((typ==0) | (typ==9) | (typ == 10)| (typ == 13)):
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\observations\mlp\grid_search_results_mlp2.csv"))
    elif ((typ ==4) |(typ == 11)| (typ == 12) | (typ == 14)):
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\observations\mlp\grid_search_results_mlp2.csv"))
        gridsearch_results = gridsearch_results[(gridsearch_results.nlayers == 3)].reset_index()
    elif ((typ == 5) | (typ==7) ):
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\observations\mlp\AdaptPool\\7features\grid_search_results_mlp2.csv"))
    
    setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()

    dimensions = [X.shape[1]]
    for dim in literal_eval(setup["hiddensize"]):
        dimensions.append(dim)
    dimensions.append(Y.shape[1])
    
    if ((typ == 6) | (typ==7) | (typ==8) | (typ==5)):
        featuresize = setup["featuresize"]
    else:
        featuresize = None

    hparams = {"batchsize": int(setup["batchsize"]), 
               "epochs":None, 
               "history": int(setup["history"]), 
               "hiddensize":literal_eval(setup["hiddensize"]),
               "learningrate":setup["learningrate"]}

    model_design = {"dimensions":dimensions,
                    "activation":nn.ReLU,
                    "featuresize":featuresize}
    
    return hparams, model_design, X, Y, X_test, Y_test

