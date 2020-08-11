# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:09:06 2020

@author: marie
"""

#%%
import models
from sklearn import metrics
from sklearn.model_selection import KFold

import preprocessing
import utils
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import visualizations

#%% Load Data
X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

#%%

D_in, D_out = 7, 1
N, H, L = 1,100, 20
#%%
dimensions = [7, 100, 1]
dim_channels = [15, 30]
kernel_size=2

#%%

def train_model_CV(X, Y, hparams, splits):
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = utils.minmax_scaler(X), utils.minmax_scaler(Y)

    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    #history = hparams["history"]
    #opti = hparams["optimizer"]
    #crit = hparams["criterion"]
    learningrate = hparams["learningrate"]
    shuffled_CV = hparams["shuffled_CV"]

    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))

    performance = []
    y_tests = []
    y_preds = []

    kf = KFold(n_splits=splits, shuffle = shuffled_CV)
    kf.get_n_splits(X)

    split=0

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        Y_test = torch.tensor(Y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        Y_train = torch.tensor(Y_train).type(dtype=torch.float)
        
        model = models.ConvN(dimensions, dim_channels, kernel_size, L, activation = nn.Sigmoid)
        optimizer = optim.Adam(model.parameters(), lr = learningrate)
        criterion = nn.MSELoss()
            
        for epoch in range(epochs):
    
            x = torch.empty(size=(batchsize, X_train.shape[1], L))
            y = torch.empty(size=(batchsize,Y_train.shape[1]))
    
            for i in range(batchsize):
                subset = [range((j-L),j) for j in random.sample(range(L,X_train.shape[0]-1), N)]
                x[i], y[i] = X_train[subset].transpose(0,1), Y_train[np.max(subset)+1]

            # Training
            model.train()
            optimizer.zero_grad()
            output = model(x)
            
            # Compute training loss
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(utils.reshaping(X_train, L))
                pred_test = model(utils.reshaping(X_test, L))
                rmse_train[split, epoch] = np.sqrt(criterion(pred_train[:-1], Y_train[L+1:]))
                rmse_val[split, epoch] = np.sqrt(criterion(pred_test[:-1], Y_test[L+1:]))
                #mae_train[i, epoch] = metrics.mean_absolute_error(Y_train[L+1:], pred_train[:-1])
                #mae_val[i, epoch] = metrics.mean_absolute_error(Y_test[L+1:], pred_test[:-1])
            
        with torch.no_grad():
            preds_train = model(utils.reshaping(X_train, L))
            preds_test = model(utils.reshaping(X_test, L))
            performance.append([np.sqrt(criterion(preds_train[:-1], Y_train[L+1:]).numpy()),
                                np.sqrt(criterion(preds_test[:-1], Y_test[L+1:]).numpy())])
                            #metrics.mean_absolute_error(Y_train, preds_train.numpy()),
                            #metrics.mean_absolute_error(Y_test, preds_test.numpy())])
      
        y_test, preds_test = utils.minmax_rescaler(Y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        y_tests.append(y_test[L:])
        y_preds.append(preds_test)
        
        split += 1
            
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    performance = np.mean(np.array(performance), axis=0)
    
    return(running_losses, performance, y_tests, y_preds)

#%%
hparams = {"batchsize": 20, 
           "epochs":50, 
           "history":L, 
           "hiddensize":H,
           "optimizer":"adam", 
           "criterion":"mse", 
           "learningrate":0.05,
            "shuffled_CV":False}

running_losses, performance, y_tests, y_preds = train_model_CV(X, Y, hparams, splits=6)


visualizations.plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], hparams, figure="conv1")
visualizations.plot_nn_predictions(y_tests, y_preds, L , figure="conv1", model="convnet")

#%%
for i in range(5):
    plt.plot(y_preds[i])
    
    
#%%
fig, ax = plt.subplots(len(y_tests), figsize=(10,10))
fig.suptitle(f"Network Predictions")

for i in range(len(y_tests)):
    ax[i].plot(y_tests[i], color="grey", label="targets", linewidth=0.9, alpha=0.6)
    ax[i].plot(y_preds[i], color="darkblue", label="nn predictions", linewidth=0.9, alpha=0.6)
    ax[i].plot(y_tests[i] - y_preds[i], color="lightgreen", label="absolute error", linewidth=0.9, alpha=0.6)
    
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
