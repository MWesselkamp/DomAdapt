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
def training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, data_dir,
                   save, sparse=None, dropout_prob = 0.0, dropout = False, splits = 5):
    
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
        
        if not featuresize is None:
          if isinstance(model_design, dict):
              print("Loading pretrained Model.")
              model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"], dropout_prob, dropout)
              model.load_state_dict(torch.load(os.path.join(data_dir, f"model{i}.pth")))
          else:
              model = model_design
        else:
          model = models.MLP(model_design["dimensions"], model_design["activation"])
          model.load_state_dict(torch.load(os.path.join(data_dir, f"model{i}.pth")))
        model.eval()
        
        if not feature_extraction is None:
            print("Freezing all weights.")
            if featuresize is None:
              try:
                print("backpropagation of third layer parameters.")
                model.hidden3.weight.requires_grad = False
                model.hidden3.bias.requires_grad = False
              except:
                print("backpropagation of first layer parameters.")
                model.hidden1.weight.requires_grad = False
                model.hidden1.bias.requires_grad = False
            else:
              for child in model.children():
                  print("Entering child node")
                  for name, parameter in child.named_parameters():
                      #print(name)
                      if not name in feature_extraction:
                          print("disable backprob for", name)
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
                if not sparse is None:
                    torch.save(model.state_dict(), os.path.join(data_dir, f"tuned/setting1/sparse//{sparse}/model{i}.pth"))
                else:
                    torch.save(model.state_dict(), os.path.join(data_dir, f"tuned/setting1/model{i}.pth"))
            else:
                if not sparse is None:
                    torch.save(model.state_dict(), os.path.join(data_dir, f"tuned/setting0/sparse//{sparse}/model{i}.pth"))
                else:
                    torch.save(model.state_dict(), os.path.join(data_dir, f"tuned/setting0/model{i}.pth"))
        
        y_tests.append(y_test.numpy())
        y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}

    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, yt_tests, y_preds)
        
        

#%%
def finetune(X, Y, epochs, model, pretrained_type, searchpath, 
             feature_extraction = None, eval_set = None, save=True, 
             sparse = None, dummies = False, dropout_prob = 0.0, dropout = False, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
    if ((pretrained_type == 9) | (pretrained_type == 10) | (pretrained_type == 13)):
      gridsearch_results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
    elif ((pretrained_type == 4) | (pretrained_type == 11) | (pretrained_type == 12) | (pretrained_type == 14)):
      gridsearch_results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
      gridsearch_results = gridsearch_results[(gridsearch_results.nlayers == 3)].reset_index()
    elif ((pretrained_type == 5) | (pretrained_type == 7)):
      gridsearch_results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/adaptive_pooling/grid_search_results_{model}2.csv"))
    else:
      gridsearch_results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/simulations/grid_search_results_{model}2.csv"))
    
    setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()
    if ((pretrained_type == 4) | (pretrained_type == 9) | (pretrained_type == 10) | (pretrained_type == 11) | (pretrained_type == 12) | (pretrained_type == 13) | (pretrained_type == 14)):
        setup["featuresize"] = None
        
    if dummies:
      
        Xf = np.zeros((X.shape[0], 12))
        Xf_test = np.zeros((eval_set["X_test"].shape[0], 12))
        Xf[:,:7] = X
        X = Xf
        Xf_test[:,:7] = eval_set["X_test"]
        eval_set["X_test"] = Xf_test
      
    dimensions = [X.shape[1]]
    for dim in literal_eval(setup["hiddensize"]):
        dimensions.append(dim)
    dimensions.append(Y.shape[1])

    hparams = {"batchsize": int(setup["batchsize"]), 
               "epochs":epochs, 
               "history": int(setup["history"]), 
               "hiddensize":literal_eval(setup["hiddensize"]),
               "learningrate":setup["learningrate"]}

    model_design = {"dimensions":dimensions,
                    "activation":nn.ReLU,
                    "featuresize":setup["featuresize"]}
                    
    data_dir = os.path.join(data_dir, f"output/models/{model}{pretrained_type}/{searchpath}")
    
    if not sparse is None:
      if ((pretrained_type == 10) | (pretrained_type == 13)):
        ind = np.load(f"/home/fr/fr_fr/fr_mw263/output/models/{model}0//relu/sparse//{sparse}/ind.npy") #, allow_pickle=True).item()
        X, Y = X[ind], Y[ind]
      elif ((pretrained_type == 12) | (pretrained_type == 14)): 
        ind = np.load(f"/home/fr/fr_fr/fr_mw263/output/models/{model}4//relu/sparse//{sparse}/ind.npy") #, allow_pickle=True).item()
        X, Y = X[ind], Y[ind]
      elif ((pretrained_type == 5) | (pretrained_type == 7)): 
        ind = np.load(f"/home/fr/fr_fr/fr_mw263/output/models/{model}5//relu/sparse//{sparse}/ind.npy") #, allow_pickle=True).item()
        X, Y = X[ind], Y[ind]
      
    running_losses,performance, y_tests, y_preds = training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, 
                                                               data_dir , 
                                                               save, sparse, 
                                                               dropout_prob, dropout)
    
      
    if not feature_extraction is None:
        data_dir = os.path.join(data_dir, "tuned/setting1")
    else:
        data_dir = os.path.join(data_dir, "tuned/setting0")
        
    if not sparse is None:
      data_dir = os.path.join(data_dir, f"sparse//{sparse}")
        
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

