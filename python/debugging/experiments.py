# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:16:49 2020

@author: marie
"""
#%% Set working directory
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')
import os.path

import preprocessing


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import preprocessing
import utils
import models
import torch.nn.functional as F
#%% Load Data
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ["hyytiala"],
                                years = [2001, 2002, 2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound")

#x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
#y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

#%% Normalize features
X = utils.minmax_scaler(X)


#%% Prep data
N = 50
subset = random.sample(range(X.shape[0]), N)
X_batch, y_batch = X[subset], Y[subset]
        
x = torch.tensor(X_batch).type(dtype=torch.float)
y = torch.tensor(y_batch).type(dtype=torch.float)

x.shape
y.shape

def create_inout_sequences(x, y, batchsize, seqlen, model):
    
    """
    In-out Sequences for LSTM.
    Used in: dev_lstm.train_model_CV
    """
    
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

batchsize=16
seqlen=10

x, y = utils.create_inout_sequences(x, y, batchsize, seqlen, "cnn")


x.shape
y.shape


#%%Layers
D_in = 7
H = 32
D_out = 1
kernel_size=2

x.shape

conv1 = nn.Conv1d(D_in, 7, kernel_size)

out = F.sigmoid(conv1(x))
out.shape

max_pool = nn.MaxPool1d(kernel_size, 1)

out = max_pool(out)
out.shape

conv2 = nn.Conv1d(7, 14, kernel_size)
out = conv2(out)
out.shape
out = max_pool(out)

flat = Flatten()

out = flat(out)
out.shape

fc1 = nn.Linear(84, 1)
out = fc1(out)
out.shape
