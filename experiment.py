# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:08:25 2020

@author: marie
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split

def train_model(model, hparams, X, Y, minibatches = True):
    
    batchsize = hparams["batchsize"]
    batches = hparams["batches"]
    history = hparams["history"]
    optimizer = hparams["optimizer"]
    criterion = hparams["criterion"]
    learningrate = hparams["learningrate"]
    
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr = learningrate)
    else: 
        raise ValueError("Don't know optimizer")
        
    if criterion == "mse":
        criterion = nn.MSELoss()
    else: 
        raise ValueError("Don't know criterion")
        
    train_loss = {"rmse":[], "mae":[]}
    val_loss = {"rmse":[], "mae":[]}
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
    X_test = torch.tensor(X_test).type(dtype=torch.float)
    
    for batch in range(batches):
        
    # Training
    # Subset data set to small batch in each epoch.
    # If hisotry > 0, batches are extended by the specified lags.
    
        if minibatches:
            subset = [j for j in random.sample(range(X_train.shape[0]), batchsize) if j > history]
            subset_h = [item for sublist in [list(range(j-history,j)) for j in subset] for item in sublist]
            x = np.concatenate((X_train[subset], X_train[subset_h]), axis=0)
            y = np.concatenate((y_train[subset], y_train[subset_h]), axis=0)
        else:
            hparams["batchsize"] = X_train.shape[0]
            x = X_train
            y = y_train
        
        x = torch.tensor(x).type(dtype=torch.float)
        y = torch.tensor(y).type(dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(x)
        
        loss = criterion(output, y)
        

        train_loss["rmse"].append(np.sqrt(metrics.mean_squared_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64"))))
        train_loss["mae"].append(metrics.mean_absolute_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64")))

        #training_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64")))
        #training_loss_mae[i:] = metrics.mean_absolute_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64"))
        
        loss.backward()
        optimizer.step()
    
    # Evaluate current model
        preds = model(X_test)
        
        val_loss["rmse"].append(np.sqrt(metrics.mean_squared_error(y_test, preds.detach().numpy().astype("float64"))))
        val_loss["mae"].append(metrics.mean_absolute_error(y_test, preds.detach().numpy().astype("float64")))


    return(train_loss, val_loss)

        #validation_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y_test, preds.detach().numpy().astype("float64")))
        #validation_loss_mae[i:] = metrics.mean_absolute_error(y_test, preds.detach().numpy().astype("float64"))
        
    # Stop early if validation loss isn't decreasing/increasing again
    