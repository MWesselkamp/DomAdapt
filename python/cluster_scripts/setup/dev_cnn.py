# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:09:06 2020

@author: marie
"""

#%%
import setup.models as models
import setup.utils as utils

from sklearn import metrics
from sklearn.model_selection import KFold
import os.path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from ast import literal_eval

#%%

def train_model_CV(hparams, model_design, X, Y, eval_set, dropout_prob, data_dir, 
                   save, splits=5):

    batchsize = hparams["batchsize"]
    epochs = hparams["epochs"]
    seqlen = hparams["history"]
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))

    performance = []
    y_tests = []
    y_preds = []

    #X_mean, X_std = np.mean(X), np.std(X)
    #X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        #Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
        Yt_test = eval_set["Y_test"]
        Yt_test = torch.tensor(Yt_test).type(dtype=torch.float)
        Xt_test = torch.tensor(Xt_test).type(dtype=torch.float)
        xt_test, yt_test = utils.create_inout_sequences(Xt_test, Yt_test, "full", seqlen, model="cnn")
        yt_tests = []
        
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
                             model_design["channels"], 
                             model_design["kernelsize"],
                             seqlen, 
                             model_design["activation"])

        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        criterion = nn.MSELoss()
        
        x_train, y_train = utils.create_inout_sequences(X_train, Y_train, "full", seqlen, model="cnn")
        x_test, y_test = utils.create_inout_sequences(X_test, Y_test, "full", seqlen, model="cnn")
        
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
                pred_train = model(x_train)
                rmse_train[split, epoch] = utils.rmse(y_train, pred_train)
                mae_train[split, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                if eval_set is None:
                    pred_test = model(x_test)
                    rmse_val[split, epoch] = utils.rmse(y_test, pred_test)
                    mae_val[split, epoch] = metrics.mean_absolute_error(y_test, pred_test)
                else:
                    pred_test = model(xt_test)
                    rmse_val[split, epoch] = utils.rmse(yt_test,pred_test)
                    mae_val[split, epoch] = metrics.mean_absolute_error(yt_test, pred_test)
            
        with torch.no_grad():
            preds_train = model(x_train)   
            if eval_set is None:
                preds_test = model(x_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(y_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train),
                                    metrics.mean_absolute_error(y_test, preds_test)])
            else:
                preds_test = model(xt_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(yt_test,preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train),
                                    metrics.mean_absolute_error(yt_test, preds_test)])
        
        if save:
            #torch.save(model.state_dict(), os.path.join(data_dir, f"model{split}.pth"))
            torch.save(model, os.path.join(data_dir, f"model{split}.pth"))
            
        y_tests.append(y_test.numpy())
        y_preds.append(preds_test.numpy())
        if not eval_set is None:
            yt_tests.append(yt_test.numpy())
            
        split += 1
            
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    
    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, yt_tests, y_preds)

    
#%%
def selected(X, Y, model,typ, model_params, epochs, splits, simtype = None, featuresize = None, dropout_prob = 0.0, data_dir = None, 
             save = False, eval_set = None):
    
    in_features = X.shape[1]
    out_features = Y.shape[1]

    # Network training
    hparams = {"batchsize": int(model_params["batchsize"]), 
               "epochs":epochs, 
               "history":int(model_params["history"]), 
               "hiddensize":int(model_params["hiddensize"]),
               "learningrate":model_params["learningrate"]}
    model_design = {"dimensions":[in_features, int(model_params["hiddensize"]), out_features],
                    "activation":eval(model_params["activation"][8:-2]),
                    "channels":literal_eval(model_params["channels"]),
                    "kernelsize":model_params["kernelsize"]}
   
    start = time.time()
    if not data_dir is None:
        data_dir = os.path.join(os.path.join(data_dir, "models"), f"{model}{typ}")
    running_losses,performance, y_tests, y_preds = train_model_CV(hparams, model_design, 
                                                                  X, Y, 
                                                                  splits, eval_set,
                                                                  data_dir, save)
    end = time.time()
    
    if dropout_prob == 0.0:
      data_dir = os.path.join(data_dir, r"nodropout")
    else:
      data_dir = os.path.join(data_dir, r"dropout")
      
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    rets = [(end-start), 
            model_params["hiddensize"], model_params["batchsize"], model_params["learningrate"], model_params["history"], model_params["channels"], model_params["kernelsize"], model_params["activation"],
            performance[0], performance[1], performance[2], performance[3]]
    
    results = pd.DataFrame([rets], columns=["execution_time", "hiddensize", "batchsize", "learningrate", "history", "channels", "kernelsize","activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
    
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)