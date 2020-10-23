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
def train_model_CV(hparams, model_design, X, Y, splits, eval_set, 
                   data_dir, save):
    
    """
    
    
    """
    epochs = hparams["epochs"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    X_mean, X_std = np.mean(X), np.std(X)
    X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
        yt_test = eval_set["Y_test"]
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
        
        model = models.MLPmod(model_design["dimensions"], model_design["activation"])
            
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

    
#%%
def finetuning_CV(hparams, model_design, X, Y, splits, eval_set, data_dir,
                   save, feature_extraction):
    
    """
    
    
    """
    
    epochs = hparams["epochs"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    X_mean, X_std = np.mean(X), np.std(X)
    X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
        yt_test = eval_set["Y_test"]
        yt_test = torch.tensor(yt_test).type(dtype=torch.float)
        Xt_test = torch.tensor(Xt_test).type(dtype=torch.float)
        yt_tests = []
        
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
            
        print("Loading pretrained Model.")
        model = models.MLP(model_design["dimensions"], model_design["activation"])
        model.load_state_dict(torch.load(os.path.join(data_dir, f"model{i}.pth")))
        model.eval()
        if feature_extraction:
            print("Extracting features.")
            for param in model.parameters():
                param.requires_grad = False

            
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            
            # Training
            model.train()

            x, y = utils.create_batches(X_train, y_train, hparams["batchsize"], hparams["history"])
            
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
            
            output = model(x.unsqueeze(1)).squeeze_(1)
            
            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(X_train.unsqueeze(1)).squeeze_(1)
                if eval_set is None:
                    pred_test = model(X_test.unsqueeze(1)).squeeze_(1)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(y_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(y_test, pred_test)  
                else:
                    pred_test = model(Xt_test.unsqueeze(1)).squeeze_(1)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(yt_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(yt_test, pred_test)
                    
         
        # Predict with fitted model
        with torch.no_grad():
            preds_train = model(X_train.unsqueeze(1)).squeeze_(1)
            if eval_set is None:
                preds_test = model(X_test.unsqueeze(1)).squeeze_(1)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(y_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(y_test, preds_test.numpy())])
            else:
                preds_test = model(Xt_test.unsqueeze(1)).squeeze_(1)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(yt_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(yt_test, preds_test.numpy())])
    
        if save:
            torch.save(model.state_dict(), os.path.join(data_dir, f"model{i}.pth"))
        
        y_tests.append(y_test.numpy())
        y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}

    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, yt_tests, y_preds)


#%% Random Grid search: Paralellized
def mlp_selection_parallel(X, Y, hp_list, epochs, splits, searchsize, 
                           data_dir, q, hp_search = [], 
                           eval_set = None, save = False):
    
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
               "learningrate":search[2]}
    model_design = {"dimensions": dimensions,
                    "activation": search[4]}
   
    start = time.time()
    running_losses,performance, y_tests_nn, y_preds_nn = train_model_CV(hparams, model_design, X, Y, splits, eval_set, data_dir, 
                                                                  save)
    end = time.time()
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    hp_search.append([item for sublist in [[searchsize, (end-start)], [hparams["hiddensize"]], search[1:], performance] for item in sublist])

    print("Model fitted!")
    
    q.put(hp_search)
    
#%% Train the model with hyperparameters selected after random grid search:
    
def selected(X, Y, model,typ, model_params, epochs, splits, data_dir = None, 
             save = False, eval_set = None):
    
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
    
    hidden_dims = literal_eval(model_params["hiddensize"])
    
    dimensions = [X.shape[1]]
    for hdim in hidden_dims:
        dimensions.append(hdim)
    dimensions.append(Y.shape[1])
    
    hparams = {"batchsize": int(model_params["batchsize"]), 
               "epochs":epochs, 
               "history": int(model_params["history"]), 
               "hiddensize":hidden_dims,
               "learningrate":model_params["learningrate"]}

    model_design = {"dimensions": dimensions,
                    "activation": eval(model_params["activation"][8:-2])}
   
    start = time.time()
    if not data_dir is None:
        data_dir = os.path.join(os.path.join(data_dir, "models"), f"{model}{typ}")
    running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, X, Y, splits, eval_set, data_dir, 
                                                                  save)
    end = time.time()
    
    # Save: Results
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    rets = [(end-start), 
            model_params["hiddensize"], model_params["batchsize"], model_params["learningrate"], model_params["history"], model_params["activation"], 
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["execution_time", "hiddensize", "batchsize", "learningrate", "history", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
    
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
    #return(running_losses, y_tests, y_preds)