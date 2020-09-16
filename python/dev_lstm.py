# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:04:40 2020

@author: marie
"""

#%%
import models
from sklearn import metrics
from sklearn.model_selection import KFold

import os.path
import utils
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import visualizations
import time

#%%

def train_model_CV(hparams, model_design, X, Y, splits):
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = utils.minmax_scaler(X), utils.minmax_scaler(Y)

    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    L = hparams["history"]
    dimensions = model_design["dimensions"]

    #opti = hparams["optimizer"]
    #crit = hparams["criterion"]
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    performance = []
    y_tests = []
    y_preds = []
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)

    split=0

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        Y_test = torch.tensor(Y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        Y_train = torch.tensor(Y_train).type(dtype=torch.float)
        
        model = models.LSTM(dimensions[0], dimensions[1], dimensions[2], L)
        
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"], weight_decay=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            
            x, y = utils.create_inout_sequences(X_train, Y_train, batchsize, L, model="lstm")
            
            # Training
            model.train()
            
            hidden = model.init_hidden(batchsize)
            
            optimizer.zero_grad()
            output = model(x, hidden)
            
            # Compute training loss
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                rst = utils.reshaping(X_train, L, model="lstm")
                rsp = utils.reshaping(X_test, L, model="lstm")
                pred_train = model(rst, model.init_hidden(rst.shape[1]))
                pred_test = model(rsp, model.init_hidden(rsp.shape[1]))
                rmse_train[split, epoch] = np.sqrt(criterion(pred_train, Y_train[L+1:]))
                rmse_val[split, epoch] = np.sqrt(criterion(pred_test, Y_test[L+1:]))
                mae_train[split, epoch] = metrics.mean_absolute_error(Y_train[L+1:], pred_train)
                mae_val[split, epoch] = metrics.mean_absolute_error(Y_test[L+1:], pred_test)
            
        with torch.no_grad():
            rst = utils.reshaping(X_train, L, model="lstm")
            rsp = utils.reshaping(X_test, L, model="lstm")
            preds_train = model(rst, model.init_hidden(rst.shape[1]))
            preds_test = model(rsp, model.init_hidden(rsp.shape[1]))
            performance.append([np.sqrt(criterion(preds_train, Y_train[L+1:]).numpy()),
                                np.sqrt(criterion(preds_test, Y_test[L+1:]).numpy()),
                                metrics.mean_absolute_error(Y_train[L+1:], preds_train),
                                metrics.mean_absolute_error(Y_test[L+1:], preds_test)])

        y_test, preds_test = utils.minmax_rescaler(Y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        y_tests.append(y_test[L:])
        y_preds.append(preds_test)
        
        split += 1
            
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
            
    return(running_losses, performance, y_tests, y_preds)
            
            