# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:09:06 2020

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
    seqlen = hparams["history"]

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
        
        model = models.ConvN(model_design["dimensions"], 
                             model_design["dim_channels"], 
                             model_design["kernel_size"],
                             seqlen, 
                             model_design["activation"])
        
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"], weight_decay=0.001)
        criterion = nn.MSELoss()
            
        for epoch in range(epochs):
    
            x, y = utils.create_inout_sequences(X_train, Y_train, batchsize, seqlen, model="cnn")

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
                pred_train = model(utils.reshaping(X_train, seqlen, model="cnn"))
                pred_test = model(utils.reshaping(X_test, seqlen, model="cnn"))
                rmse_train[split, epoch] = np.sqrt(criterion(pred_train, Y_train[seqlen+1:]))
                rmse_val[split, epoch] = np.sqrt(criterion(pred_test, Y_test[seqlen+1:]))
                mae_train[split, epoch] = metrics.mean_absolute_error(Y_train[seqlen+1:], pred_train)
                mae_val[split, epoch] = metrics.mean_absolute_error(Y_test[seqlen+1:], pred_test)
            
        with torch.no_grad():
            preds_train = model(utils.reshaping(X_train, seqlen, model="cnn"))
            preds_test = model(utils.reshaping(X_test, seqlen, model="cnn"))
            performance.append([np.sqrt(criterion(preds_train, Y_train[seqlen+1:]).numpy()),
                                np.sqrt(criterion(preds_test, Y_test[seqlen+1:]).numpy()),
                                metrics.mean_absolute_error(Y_train[seqlen+1:], preds_train),
                                metrics.mean_absolute_error(Y_test[seqlen+1:], preds_test)])
      
        y_test, preds_test = utils.minmax_rescaler(Y_test.numpy(), Y_mean, Y_std), utils.minmax_rescaler(preds_test.numpy(), Y_mean, Y_std)
        y_tests.append(y_test[seqlen+1:])
        y_preds.append(preds_test)
        
        split += 1
            
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    
    return(running_losses, performance, y_tests, y_preds)

#%%
def conv_selection_parallel(X, Y, hp_list, epochs, splits, searchsize, datadir, q, hp_search = []):
    
    in_features = X.shape[1]
    out_features = Y.shape[1]
        
    search = [random.choice(sublist) for sublist in hp_list]

    # Network training
    hparams = {"batchsize": int(search[1]), 
           "epochs":epochs, 
           "history":int(search[3]), 
           "hiddensize":int(search[0]),
           "optimizer":"adam", 
           "criterion":"mse", 
           "learningrate":search[2]}
    model_design = {"dimensions":[in_features, int(search[0]), out_features],
                    "activation":nn.Sigmoid,
                    "dim_channels":search[4],
                    "kernel_size":search[5]}
   
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
                                figure = searchsize, model="convnet")
    visualizations.plot_nn_predictions(y_tests_nn, 
                                       y_preds_nn, 
                                       history = hparams["history"], 
                                       datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                       figure = searchsize, model="convnet")
    #visualizations.plot_prediction_error(predictions, 
    #                                     history = hparams["history"], 
    #                                     datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"),
    #                                     figure = searchsize, model="convnet")

    q.put(hp_search)
    
#%%
def conv_selected(X, Y, hp_list, epochs, splits, searchsize, datadir):
    
    hp_search = []
    in_features = X.shape[1]
    out_features = Y.shape[1]

    for i in range(searchsize):
        
        #search = [random.choice(sublist) for sublist in hp_list]

        # Network training
         # Network training
        hparams = {"batchsize": int(hp_list[1]), 
                   "epochs":epochs, 
                   "history":int(hp_list[3]), 
                   "hiddensize":int(hp_list[0]),
                   "optimizer":"adam", 
                   "criterion":"mse", 
                   "learningrate":hp_list[2]}
        model_design = {"dimensions":[in_features, int(hp_list[0]), out_features],
                        "activation":nn.Sigmoid,
                        "dim_channels":hp_list[4],
                        "kernel_size":hp_list[5]}
   
        start = time.time()
        running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, X, Y, splits=splits)
        end = time.time()
        # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
        performance = np.mean(np.array(performance), axis=0)
        hp_search.append([item for sublist in [[i, (end-start)], hp_list, performance] for item in sublist])
        print(hp_search)

        #plot_nn_loss(running_losses["rmse_train"], running_losses["rmse_val"], data="profound", history = hparams["history"], figure = i, hparams = hparams)
        #plot_nn_predictions(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
        #plot_prediction_error(predictions, history = hparams["history"], figure = i, data = "Hyytiala profound")
    
    results = pd.DataFrame(hp_search, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "channels", "kernelsize", "rmse_train", "rmse_val", "mae_train", "mae_val"])
        
    print("Best Model Run: \n", results.iloc[results['rmse_val'].idxmin()])    
    
    visualizations.plot_nn_loss(running_losses["rmse_train"], 
                                running_losses["rmse_val"], 
                                hparams = hparams,
                                datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"),
                                figure = "selected", model="convnet")
    
    visualizations.plot_nn_predictions(y_tests, 
                                       y_preds, 
                                       history = hparams["history"], 
                                       datadir = os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn"), 
                                       figure = "selected", model="convnet")
    
    return(running_losses, y_tests, y_preds)