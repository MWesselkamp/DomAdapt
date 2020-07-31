# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:08:25 2020

@author: marie
"""
import os
import os.path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold

import models
from utils import minmax_scaler
from utils import minmax_rescaler

import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

def train_model_CV(hparams, X, Y, splits = 6):
    
    
    hiddensize = hparams["hiddensize"]
    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    history = hparams["history"]
    opti = hparams["optimizer"]
    crit = hparams["criterion"]
    learningrate = hparams["learningrate"]
    shuffled_CV = hparams["shuffled_CV"]
    
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    
    kf = KFold(n_splits=splits, shuffle = shuffled_CV)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_test = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_test = np.zeros((splits, epochs))
    
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
    i = 0
    
    predictions = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = models.LinNet(D_in = X_train.shape[1], H = hiddensize, D_out = 1)
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        
        if opti == "adam":
            optimizer = optim.Adam(model.parameters(), lr = learningrate)
        else: 
            raise ValueError("Don't know optimizer")
        
        if crit == "mse":
            criterion = nn.MSELoss()
        else: 
            raise ValueError("Don't know criterion")

        for epoch in range(epochs):
    # Training
    # Subset data set to small batch in each epoch.
    # If hisotry > 0, batches are extended by the specified lags.
    

            subset = [j for j in random.sample(range(X_train.shape[0]), batchsize) if j > history]
            subset_h = [item for sublist in [list(range(j-history,j)) for j in subset] for item in sublist]
            x = np.concatenate((X_train[subset], X_train[subset_h]), axis=0)
            y = np.concatenate((y_train[subset], y_train[subset_h]), axis=0)

        
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
        
            
            output = model(x)
        
            loss = criterion(output, y)
            
            with torch.no_grad():
                rmse = criterion(output, y)
                rmse_train[i, epoch] = np.sqrt(rmse)
                
            mae_train[i, epoch] = metrics.mean_absolute_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64"))
            
            # Evaluate current model
            preds = model(X_test)

            #rmse = np.sqrt(metrics.mean_squared_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64")))
             #train_loss["rmse"].append()
            #train_loss["mae"].append()

        #training_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64")))
        #training_loss_mae[i:] = metrics.mean_absolute_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64"))

            with torch.no_grad():
                rmse = criterion(preds, y_test)
                rmse_test[i, epoch] = np.sqrt(rmse)
                
            #rmse_test[i,batch] = np.sqrt(metrics.mean_squared_error(y_test, preds.detach().numpy().astype("float64")))
            mae_test[i, epoch] = metrics.mean_absolute_error(y_test, preds.detach().numpy().astype("float64"))
            #val_loss["rmse"].append()
            #val_loss["mae"].append()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            preds_final = model(X_test).numpy()
        # rescale before returning
        y_test, preds_final = minmax_rescaler(y_test.numpy(), Y_mean, Y_std), minmax_rescaler(preds_final, Y_mean, Y_std)
            
        predictions.append([y_test, preds_final])
    
        i += 1
            
    train_loss = {"rmse":rmse_train, "mae":mae_train}
    val_loss = {"rmse":rmse_test, "mae":mae_test}

    return(train_loss, val_loss, predictions)

        #validation_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y_test, preds.detach().numpy().astype("float64")))
        #validation_loss_mae[i:] = metrics.mean_absolute_error(y_test, preds.detach().numpy().astype("float64"))
        
    # Stop early if validation loss isn't decreasing/increasing again
    
#%%
def plot_nn_loss(train_loss, val_loss, data, hparams, figure = "", data_dir = r"plots\data_quality_evaluation\fits_nn"):
    
    fig, ax = plt.subplots(figsize=(10,6))
    fig.suptitle(f"Fully connected Network: {data} data \n Epochs = {hparams['epochs']}, Shuffled_CV = {hparams['shuffled_CV']}, History = {hparams['history']} \n Hiddensize = {hparams['hiddensize']}, Batchsize = {hparams['batchsize']}, Learning_rate = {hparams['learningrate']}")

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        ax.fill_between(np.arange(hparams["epochs"]), ci_train[0],ci_train[1], color="lightgreen", alpha=0.3)
        ax.fill_between(np.arange(hparams["epochs"]), ci_val[0],ci_val[1], color="lightblue", alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    ax.plot(train_loss, color="green", label="Training loss", linewidth=0.8)
    ax.plot(val_loss, color="blue", label = "Validation loss", linewidth=0.8)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set(xlabel="Epochs", ylabel="Root Mean Squared Error")
    plt.ylim(bottom = 0)
    fig.legend(loc="upper left")

    plt.savefig(os.path.join(data_dir, f"{data}_loss_{figure}"))
    plt.close()

#%%
def plot_nn_predictions(predictions, data, figure = "", data_dir = r"plots\data_quality_evaluation\fits_nn"):
    
    """
    Plot model predictions.
    """
    fig, ax = plt.subplots(len(predictions), figsize=(10,10))
    fig.suptitle(f"Network Predictions: {data} data")

    for i in range(len(predictions)):
        ax[i].plot(predictions[i][0], color="grey", label="targets", linewidth=0.8, alpha=0.6)
        ax[i].plot(predictions[i][1], color="darkblue", label="nn predictions", linewidth=0.8, alpha=0.6)
        ax[i].plot(predictions[i][0] - predictions[i][1], color="lightgreen", label="nn predictions", linewidth=0.8, alpha=0.6)
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(data_dir, f"{data}_predictions_{figure}"))
    plt.close()
