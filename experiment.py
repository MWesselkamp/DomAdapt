# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:08:25 2020

@author: marie
"""
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold

import models
from utils import minmax_scaler
from utils import minmax_rescaler
import preprocessing

import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

import random
import time
import pandas as pd

#%%
def train_model_CV(hparams, model_design, X, Y, splits = 6):
    
    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    history = hparams["history"]
    opti = hparams["optimizer"]
    crit = hparams["criterion"]
    learningrate = hparams["learningrate"]
    shuffled_CV = hparams["shuffled_CV"]
    
    dimensions = model_design["dimensions"]
    activation = model_design["activation"]
    
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    
    kf = KFold(n_splits=splits, shuffle = shuffled_CV)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = minmax_scaler(X), minmax_scaler(Y)
    
    i = 0
    
    performance = []
    predictions = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = models.MLP(dimensions, activation)
        
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
            
    # If hisotry > 0, batches are extended by the specified lags.
            x, y = preprocessing.create_batches(X_train, y_train, batchsize, history)

            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
            
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
                pred_train = model(x)
                pred_test = model(X_test)
                rmse_train[i, epoch] = np.sqrt(criterion(pred_train, y))
                rmse_val[i, epoch] = np.sqrt(criterion(pred_test, y_test))
                mae_train[i, epoch] = metrics.mean_absolute_error(y, pred_train)
                mae_val[i, epoch] = metrics.mean_absolute_error(y_test, pred_test)
            
         
        # Predict with fitted model
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        y_train = torch.tensor(y_train).type(dtype=torch.float)
        with torch.no_grad():
            preds_train = model(X_train)
            preds_test = model(X_test)
            performance.append([np.sqrt(criterion(preds_train, y_train).numpy()),
                                np.sqrt(criterion(preds_test, y_test).numpy()),
                                metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                metrics.mean_absolute_error(y_test, preds_test.numpy())])
            
        # rescale before returning predictions
        y_test, preds_test = minmax_rescaler(y_test.numpy(), Y_mean, Y_std), minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        predictions.append([y_test, preds_test])
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    performance = np.mean(np.array(performance), axis=0)

    return(running_losses, performance, predictions)
        
    # Stop early if validation loss isn't decreasing/increasing again
    
#%%
def plot_nn_loss(train_loss, val_loss, data, hparams, history, figure = "", data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn"):
    
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

    plt.savefig(os.path.join(data_dir, f"{data}_loss_{history}{figure}"))
    plt.close()

#%% Plot Model Prediction and Prediction Error
def plot_nn_predictions(predictions, data, history, figure = "", data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn"):
    
    """
    Plot model predictions.
    """
    fig, ax = plt.subplots(len(predictions), figsize=(10,10))
    fig.suptitle(f"Network Predictions: {data} data")

    for i in range(len(predictions)):
        ax[i].plot(predictions[i][0], color="grey", label="targets", linewidth=0.9, alpha=0.6)
        ax[i].plot(predictions[i][1], color="darkblue", label="nn predictions", linewidth=0.9, alpha=0.6)
        ax[i].plot(predictions[i][0] - predictions[i][1], color="lightgreen", label="absolute error", linewidth=0.9, alpha=0.6)
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(data_dir, f"{data}_predictions_{history}{figure}"))
    plt.close()

def plot_prediction_error(predictions, data, history, figure = "", data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn"):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    fig, ax = plt.subplots(len(predictions), figsize=(10,10))
    fig.suptitle(f"Network Prediction: Root Mean Squared Error (RMSE) \n {data} data")

    for i in range(len(predictions)):
        ax[i].plot(np.sqrt(np.square(predictions[i][0] - predictions[i][1])), color="green", label="rmse", linewidth=0.9, alpha=0.6)
        ax[i].set(xlabel="Day of Year", ylabel="RMSE")
    
    #handles, labels = ax[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(data_dir, f"{data}_rmse_{history}{figure}"))
    plt.close()
    

#%% Random Grid Search:
    
def nn_selection(X, Y, hp_list, searchsize):
    
    hp_search = []
    in_features = X.shape[1]
    out_features = Y.shape[1]

    for i in range(searchsize):
        
        search = [random.choice(sublist) for sublist in hp_list]

        # Network training
        hparams = {"batchsize": search[1], 
                   "epochs":1000, 
                   "history":search[3], 
                   "hiddensize":search[0],
                   "optimizer":"adam", 
                   "criterion":"mse", 
                   "learningrate":search[2],
                   "shuffled_CV":False}

        model_design = {"dimensions": [in_features, search[0], out_features],
                        "activation": nn.Sigmoid}
   
        start = time.time()
        running_losses,performance, predictions = train_model_CV(hparams, model_design, X, Y, splits=6)
        end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
        hp_search.append([item for sublist in [[i, (end-start)], search, performance] for item in sublist])

        plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], data="profound", history = hparams["history"], figure = i, hparams = hparams)
        plot_nn_predictions(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
        plot_prediction_error(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")

    results = pd.DataFrame(hp_search, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn\grid_search_results.csv', index = False)
    
    print("Best Model Run: \n", results.iloc[results['rmse_val'].idxmin()])    
    
    return(results.iloc[results['rmse_val'].idxmin()].to_dict())
    
#%% Random Grid search: Paralellized
    
def nn_selection_parallel(X, Y, hp_list, searchsize, q):
    
    hp_search = []
    in_features = X.shape[1]
    out_features = Y.shape[1]

        
    search = [random.choice(sublist) for sublist in hp_list]

    # Network training
    hparams = {"batchsize": search[1], 
               "epochs":1000, 
               "history":search[3], 
               "hiddensize":search[0],
               "optimizer":"adam", 
               "criterion":"mse", 
               "learningrate":search[2],
               "shuffled_CV":False}
    model_design = {"dimensions": [in_features, search[0], out_features],
                    "activation": nn.Sigmoid}
   
    start = time.time()
    running_losses,performance, predictions = train_model_CV(hparams, model_design, X, Y, splits=6)
    end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    hp_search.append([item for sublist in [[searchsize, (end-start)], search, performance] for item in sublist])

    plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], data="profound", history = hparams["history"], figure = searchsize, hparams = hparams)
    plot_nn_predictions(predictions, history = hparams["history"], figure = searchsize, data = "Hyytiala profound")
    plot_prediction_error(predictions, history = hparams["history"], figure = searchsize, data = "Hyytiala profound")

    print("Model fitted!")
    
    q.put(hp_search)