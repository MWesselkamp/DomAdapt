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

import setup.models as models
import setup.utils as utils

import time
import pandas as pd
import os.path

from ast import literal_eval
#%%
def train_model_CV(hparams, model_design, X, Y, eval_set, dropout_prob,
                   data_dir, save, splits=5):
    
    """
    
    
    """
    epochs = hparams["epochs"]
    featuresize = model_design["featuresize"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    #X_mean, X_std = np.mean(X), np.std(X)
    #X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        yt_test = eval_set["Y_test"]
        #Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
        yt_test = torch.tensor(yt_test).type(dtype=torch.float)
        Xt_test = torch.tensor(Xt_test).type(dtype=torch.float)
        #yt_tests = []
        
    i = 0
    
    performance = []
    y_tests = []
    y_preds = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        y_train = torch.tensor(y_train).type(dtype=torch.float)
        
        if featuresize is None:
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"], dropout_prob)
            
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            
            # Training
            model.train()

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
                if eval_set is None:
                    pred_test = model(X_test)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(y_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(y_test, pred_test)  
                else:
                    pred_test = model(Xt_test)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(yt_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(yt_test, pred_test)
                    
         
        # Predict with fitted model
        with torch.no_grad():
            preds_train = model(X_train)
            if eval_set is None:
                preds_test = model(X_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(y_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(y_test, preds_test.numpy())])
            else:
                preds_test = model(Xt_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(yt_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(yt_test, preds_test.numpy())])
    
        if save:
            torch.save(model.state_dict(), os.path.join(data_dir, f"model{i}.pth"))
        
        if eval_set is None:
            y_tests.append(y_test.numpy())
        else:
            y_tests.append(yt_test.numpy())
            
        y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}

    return(running_losses, performance, y_tests, y_preds)
    
#%% Train the model with hyperparameters selected after random grid search:
    
def selected(X, Y, model,typ, best_model, epochs, splits, change_architecture = False,
             traindata_perc = None, simtype = None, featuresize = None, dropout_prob = 0.0, 
             data_dir = None, save = False, eval_set = None):
    
    """
    Takes the best found model parameters and trains a MLP with it.
    
    Args:
        X, Y (numpy array): Feature and Target data. \n
        model_params (dict): dictionary containing all required model parameters. \n
        epochs (int): epochs to train the model. \n
        splits (int): How many splits will be used in the CV. \n
        eval_set (numpy array): if provided, used for model evaluation. Default to None.
        
    Returns:
        running_losses: epoch-wise training and validation errors (rmse and mae) per split.\n
        y_tests: Target test set on which the model was evaluated on per split.\n
        y_preds: Network predictions per split.\n
        performance (pd.DataFrame): Data frame of model parameters and final training and validation errors.\n
    """
    
    hidden_dims = literal_eval(best_model["hiddensize"])
    
    if featuresize is None:
        dimensions = [X.shape[1]]
    else:
        dimensions = []
    for hdim in hidden_dims:
        dimensions.append(hdim)
    dimensions.append(Y.shape[1])
    
    hparams = {"batchsize": int(best_model["batchsize"]), 
               "epochs":epochs, 
               "history": int(best_model["history"]), 
               "hiddensize":hidden_dims,
               "learningrate":best_model["learningrate"]}

    if change_architecture:
      activation = best_model["activation"]
    else:
      activation = eval(best_model["activation"][8:-2])
      
    model_design = {"dimensions": dimensions,
                    "activation": activation}
   
    start = time.time()
    if not data_dir is None:
        data_dir = os.path.join(os.path.join(data_dir, "models"), f"{model}{typ}")
            
    running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, 
                                                                  X, Y, 
                                                                  splits, eval_set, featuresize, dropout_prob,
                                                                  data_dir, save)
    end = time.time()
    
    # Save: Results
    if not simtype is None:
      data_dir = os.path.join(data_dir, f"{simtype}")
        
    if not featuresize is None:
      data_dir = os.path.join(data_dir, r"adaptive_pooling")

    if dropout_prob == 0.0:
      data_dir = os.path.join(data_dir, r"nodropout")
    else:
      data_dir = os.path.join(data_dir, r"dropout")
      
    if not traindata_perc is None:
        data_dir = os.path.join(data_dir, f"data{traindata_perc}perc")
    
    if change_architecture:
        data_dir = os.path.join(data_dir, f"sigmoidActivation")
      
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    rets = [(end-start), 
            best_model["hiddensize"], best_model["batchsize"], best_model["learningrate"], best_model["history"], best_model["activation"], 
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["execution_time", "hiddensize", "batchsize", "learningrate", "history", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
    
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
    #return(running_losses, y_tests, y_preds)
    