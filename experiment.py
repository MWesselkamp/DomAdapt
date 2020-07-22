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

def train_model(model, hparams, X, Y):
    
    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    history = hparams["history"]
    optimizer = hparams["optimizer"]
    criterion = hparams["criterion"]
    
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr = 1e-1)
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
    
    for i in range(epochs):
        
    # Training
    # Subset data set to small batch in each epoch.
    # If hisotry > 0, batches are extended by the specified lags.
    
        subset = [i for i in random.sample(range(X_train.shape[0]), batchsize) if i > history]
        subset_h = [item for sublist in [list(range(i-history,i)) for i in subset] for item in sublist]
        
        X_batch = np.concatenate((X_train[subset], X_train[subset_h]), axis=0)
        y_batch = np.concatenate((y_train[subset], y_train[subset_h]), axis=0)
        
        x = torch.tensor(X_batch).type(dtype=torch.float)
        y = torch.tensor(y_batch).type(dtype=torch.float)
        
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
    