# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:08:25 2020

@author: marie
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold

import models
import utils
#from sklearn.model_selection import train_test_split

import random
import time
import pandas as pd
import os.path

from ast import literal_eval
#%%
def train_model_CV(hparams, model_design, X, Y, splits = 6, eval_set = None):
    
    epochs = hparams["epochs"]
    criterion = hparams["criterion"]
    optimizer = hparams["optimizer"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = utils.minmax_scaler(X), utils.minmax_scaler(Y)
    
    if not eval_set is None:
        Yt_mean, Yt_std = np.mean(eval_set)
        Yt = utils.minmax_scaler(eval_set)
        yt_tests = []
        
    i = 0
    
    performance = []
    y_tests = []
    y_preds = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = models.MLP(model_design["dimensions"], model_design["activation"])
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        y_train = torch.tensor(y_train).type(dtype=torch.float)
        
        if not eval_set is None:
            yt_train, yt_test = Yt[train_index], Yt[test_index]
            yt_train, yt_test = torch.tensor(yt_train).type(dtype=torch.float), torch.tensor(yt_test).type(dtype=torch.float)
        
        for epoch in range(epochs):
            
            # Training
            model.train()
    # If hisotry > 0, batches are extended by the specified lags.

            x, y = utils.create_batches(X_train, y_train, hparams["batchsize"], hparams["history"])
            
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
            
                
            output = model(x)
            
            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(X_train)
                pred_test = model(X_test)
                if eval_set is None:
                    rmse_train[i, epoch] = np.sqrt(np.mean(np.square(y_train-pred_train).numpy()))
                    rmse_val[i, epoch] = np.sqrt(np.mean(np.square(y_test-pred_test).numpy()))
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(y_test, pred_test)  
                else:
                    rmse_train[i, epoch] = np.sqrt(np.mean(np.square(yt_train-pred_train).numpy()))
                    rmse_val[i, epoch] = np.sqrt(np.mean(np.square(yt_test-pred_test).numpy()))
                    mae_train[i, epoch] = metrics.mean_absolute_error(yt_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(yt_test, pred_test)
                    
         
        # Predict with fitted model
        with torch.no_grad():
            preds_train = model(X_train)
            preds_test = model(X_test)
            if eval_set is None:
                performance.append([np.sqrt(np.mean(np.square(y_train-preds_train).numpy())),
                                    np.sqrt(np.mean(np.square(y_test-pred_test).numpy())),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(y_test, preds_test.numpy())])
            else:
                performance.append([np.sqrt(np.mean(np.square(yt_train-preds_train).numpy())),
                                    np.sqrt(np.mean(np.square(yt_test-pred_test).numpy())),
                                    metrics.mean_absolute_error(yt_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(yt_test, preds_test.numpy())])
        
        # rescale before returning predictions
        y_test, preds_test = utils.minmax_rescaler(y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        y_tests.append(y_test)
        y_preds.append(preds_test)
        
        if not eval_set is None:
            yt_tests.append(utils.minmax_rescaler(yt_test.numpy(), Yt_mean, Yt_std))
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    #performance = np.mean(np.array(performance), axis=0)

    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, y_tests, y_preds, yt_tests)
        
    # Stop early if validation loss isn't decreasing/increasing again
    

#%% Random Grid search: Paralellized
def mlp_selection_parallel(X, Y, hp_list, epochs, splits, searchsize, datadir, q, hp_search = []):
    
    search = [random.choice(sublist) for sublist in hp_list]
    
    n_layers = search[5]
    dimensions = [X.shape[1]]
    for layer in range(n_layers):
        # randomly pick hiddensize from hiddensize list
        dimensions.append(random.choice(hp_list[0]))
    dimensions.append(Y.shape[1])

    # Network training
    hparams = {"batchsize": search[1], 
               "epochs":epochs, 
               "history":search[3], 
               "hiddensize":dimensions[1:-1],
               "optimizer":"adam", 
               "criterion":"mse", 
               "learningrate":search[2]}
    model_design = {"dimensions": dimensions,
                    "activation": search[4]}
   
    start = time.time()
    running_losses,performance, y_tests_nn, y_preds_nn = train_model_CV(hparams, model_design, X, Y, splits=splits)
    end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    hp_search.append([item for sublist in [[searchsize, (end-start)], [hparams["hiddensize"]], search[1:], performance] for item in sublist])

    print("Model fitted!")
    
    q.put(hp_search)
    
#%% Random Grid Search:
    
def selected(X, Y, model_params, epochs, splits,  datadir):
    
    hidden_dims = literal_eval(model_params["hiddensize"])
    
    dimensions = [X.shape[1]]
    for hdim in hidden_dims:
        dimensions.append(hdim)
    dimensions.append(Y.shape[1])
    
    hparams = {"batchsize": int(model_params["batchsize"]), 
               "epochs":epochs, 
               "history": int(model_params["history"]), 
               "hiddensize":hidden_dims,
               "optimizer":"adam", 
               "criterion":"mse", 
               "learningrate":model_params["learningrate"]}

    model_design = {"dimensions": dimensions,
                    "activation": eval(model_params["activation"][8:-2])}
   
    start = time.time()
    running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, X, Y, splits=splits)
    end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    
    rets = [(end-start), 
            model_params["hiddensize"], model_params["batchsize"], model_params["learningrate"], model_params["history"], model_params["activation"], 
            performance[0], performance[1], performance[2], performance[3]]

    results = pd.DataFrame(rets, columns=["execution_time", "hiddensize", "batchsize", "learningrate", "history", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    
    return(running_losses, y_tests, y_preds, results)