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
from sklearn.model_selection import train_test_split

import models
import utils
import visualizations
#from sklearn.model_selection import train_test_split

import random
import time
import pandas as pd
import os.path

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
    X, Y = utils.minmax_scaler(X), utils.minmax_scaler(Y)
    
    i = 0
    
    performance = []
    y_tests = []
    y_preds = []
    
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
            x, y = utils.create_batches(X_train, y_train, batchsize, history)

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
        y_test, preds_test = utils.minmax_rescaler(y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        y_tests.append(y_test)
        y_preds.append(preds_test)
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    #performance = np.mean(np.array(performance), axis=0)

    return(running_losses, performance, y_tests, y_preds)
        
    # Stop early if validation loss isn't decreasing/increasing again
    

#%% Random Grid search: Paralellized
def mlp_selection_parallel(X, Y, hp_list, epochs, splits, searchsize, datadir, q, hp_search = []):
    
    in_features = X.shape[1]
    out_features = Y.shape[1]

    search = [random.choice(sublist) for sublist in hp_list]

    # Network training
    hparams = {"batchsize": search[1], 
               "epochs":epochs, 
               "history":search[3], 
               "hiddensize":search[0],
               "optimizer":"adam", 
               "criterion":"mse", 
               "learningrate":search[2],
               "shuffled_CV":False}
    model_design = {"dimensions": [in_features, search[0], out_features],
                    "activation": nn.Sigmoid}
   
    start = time.time()
    running_losses,performance, y_tests_nn, y_preds_nn = train_model_CV(hparams, model_design, X, Y, splits=splits)
    end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    hp_search.append([item for sublist in [[searchsize, (end-start)], search, performance] for item in sublist])

    print("Model fitted!")
    
    #predictions = [[i,j] for i,j in zip(y_tests_nn, y_preds_nn)]
    
    visualizations.plot_nn_loss(running_losses["rmse_train"], 
                                running_losses["rmse_val"], 
                                hparams = hparams, 
                                datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                figure = searchsize, model="mlp")
    visualizations.plot_nn_predictions(y_tests_nn, 
                                       y_preds_nn, 
                                       history = hparams["history"], 
                                       datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                       figure = searchsize, model="mlp")
    #visualizations.plot_prediction_error(predictions, history = hparams["history"], figure = searchsize, model="mlp")

    q.put(hp_search)
    
#%% Random Grid Search:
    
def mlp_selected(X, Y, hp_list, epochs, splits, searchsize, datadir):
    
    hp_search = []
    in_features = X.shape[1]
    out_features = Y.shape[1]

    for i in range(searchsize):
        
        #search = [random.choice(sublist) for sublist in hp_list]

        # Network training
        hparams = {"batchsize": int(hp_list[1]), 
                   "epochs":epochs, 
                   "history": int(hp_list[3]), 
                   "hiddensize":int(hp_list[0]),
                   "optimizer":"adam", 
                   "criterion":"mse", 
                   "learningrate":hp_list[2],
                   "shuffled_CV":False}

        model_design = {"dimensions": [in_features, int(hp_list[0]), out_features],
                        "activation": nn.Sigmoid}
   
        start = time.time()
        running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, X, Y, splits=splits)
        end = time.time()
        # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
        performance = np.mean(np.array(performance), axis=0)
        hp_search.append([item for sublist in [[i, (end-start)], hp_list, performance] for item in sublist])

        #plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], data="profound", history = hparams["history"], figure = i, hparams = hparams)
        #plot_nn_predictions(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
        #plot_prediction_error(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
    
    results = pd.DataFrame(hp_search, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
        
    print("Best Model Run: \n", results.iloc[results['rmse_val'].idxmin()])    
    
    visualizations.plot_nn_loss(running_losses["rmse_train"], 
                                running_losses["rmse_val"], 
                                hparams = hparams, 
                                datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                figure = "selected", model="mlp")
    
    visualizations.plot_nn_predictions(y_tests, 
                                       y_preds, 
                                       history = hparams["history"], 
                                       datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                       figure = "selected", model="mlp")
    
    return(running_losses, y_tests, y_preds)
    
    
#%%NN Fit

def train_model(X, Y, nnp):
    
    in_features = X.shape[1]
    out_features = Y.shape[1]
    
    hparams = {"batchsize": int(nnp["batchsize"]), 
           "epochs":1000, 
           "history":int(nnp["history"]), 
           "hiddensize":int(nnp["hiddensize"]),
           "optimizer":"adam", 
           "criterion":"mse", 
           "learningrate":nnp["learningrate"],
           "shuffled_CV":False,
           "training_runs":1}

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    model_design = {"dimensions": [in_features, hparams["hiddensize"], out_features],
                "activation": nn.Sigmoid}
    
    rmse_train = np.zeros((hparams["training_runs"], hparams["epochs"]))
    rmse_val = np.zeros((hparams["training_runs"], hparams["epochs"]))
    mae_train = np.zeros((hparams["training_runs"], hparams["epochs"]))
    mae_val = np.zeros((hparams["training_runs"], hparams["epochs"]))
    
    # z-score data
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    X, Y = utils.minmax_scaler(X), utils.minmax_scaler(Y)
    
    performance = []
    predictions = []
    
    for run in range(hparams["training_runs"]):
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

        model = models.MLP(model_design["dimensions"], model_design["activation"])
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        
        if hparams["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        else: 
            raise ValueError("Don't know optimizer")
        
        if hparams["criterion"] == "mse":
            criterion = nn.MSELoss()
        else: 
            raise ValueError("Don't know criterion")

        for epoch in range(hparams["epochs"]):
            
    # If hisotry > 0, batches are extended by the specified lags.
            x, y = utils.create_batches(X_train, y_train, hparams["batchsize"], hparams["history"])

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
                rmse_train[run, epoch] = np.sqrt(criterion(pred_train, y))
                rmse_val[run, epoch] = np.sqrt(criterion(pred_test, y_test))
                mae_train[run, epoch] = metrics.mean_absolute_error(y, pred_train)
                mae_val[run, epoch] = metrics.mean_absolute_error(y_test, pred_test)
            
         
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
        y_test, preds_test = utils.minmax_rescaler(y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        predictions.append([y_test, preds_test])
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    performance = np.mean(np.array(performance), axis=0)
    
    return(running_losses, performance, predictions)