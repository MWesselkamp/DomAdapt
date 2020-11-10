# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold
import os.path
import pandas as pd
from ast import literal_eval

import setup.models as models
import setup.utils as utils


#%%
def training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, featuresize, data_dir,
                   save, splits = 5):
    
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
    #X_mean, X_std = np.mean(X), np.std(X)
    #X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        #Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
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
        
        if isinstance(model_design, dict):
            print("Loading pretrained Model.")
            model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
            model.load_state_dict(torch.load(os.path.join(data_dir, f"model{i}.pth")))
        else:
            model = model_design
        model.eval()
        
        if not feature_extraction is None:
            print("Freezing all weights.")
            for child in model.children():
                for name, parameter in child.named_parameters():
                    if not name in feature_extraction:
                        parameter.requires_grad = False
                    #else:
                    #    parameter.requires_grad = False
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        
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
            if not feature_extraction is None:
                torch.save(model.state_dict(), os.path.join(data_dir, f"tuned\setting1\model{i}.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(data_dir, f"tuned\setting0\model{i}.pth"))
        
        y_tests.append(y_test.numpy())
        y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}

    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, yt_tests, y_preds)
        
#%%
def finetune(X, Y, epochs, model, pretrained_type, searchpath, featuresize, save=False, 
             feature_extraction = None, eval_set = None,
             data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\mlp\grid_search_results_{model}1.csv"))
    
    setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()

    dimensions = literal_eval(setup["hiddensize"])
    dimensions.append(1) # adds the output dimension!

    hparams = {"batchsize": int(setup["batchsize"]), 
               "epochs":epochs, 
               "history": int(setup["history"]), 
               "hiddensize":literal_eval(setup["hiddensize"]),
               "learningrate":setup["learningrate"]}

    model_design = {"dimensions":dimensions,
                    "activation":nn.ReLU}
    
    data_dir = os.path.join(data_dir, f"python\outputs\models\{model}{pretrained_type}\{searchpath}")

    running_losses,performance, y_tests, y_preds = training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, featuresize,
                                                               data_dir, 
                                                               save)
        
    if not feature_extraction is None:
        data_dir = os.path.join(data_dir, "tuned\setting1")
    else:
        data_dir = os.path.join(data_dir, "tuned\setting0")
    
    performance = np.mean(np.array(performance), axis=0)
    rets = [epochs, pretrained_type, 
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["epochs", "pretrained_type", 
                                    "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
        
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
    return(running_losses,performance, y_tests, y_preds)

