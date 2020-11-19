# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:09:07 2020

@author: marie
"""

import numpy as np

import torch
import torch.nn as nn

import setup.preprocessing as preprocessing

import time
import pandas as pd
import os.path

from ast import literal_eval



#%%
def set_model_parameters(model, typ, epochs, change_architecture, adaptive_pooling,
                         X, Y, featuresize = None, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
    if (typ == 4):
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}4.csv"))
    elif (typ == 2):
      if adaptive_pooling:
        results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/adaptive_pooling/grid_search_results_{model}2.csv"))
        featuresize = results["featuresize"]
      else:
        results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
    else:
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
      
    best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
    
    if change_architecture == True:
      best_model["activation"] == nn.Sigmoid
      
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
      
    model_design = {"dimensions": dimensions,
                    "activation": eval(best_model["activation"][8:-2]),
                    "featuresize":featuresize}
    
    return hparams, model_design

    
#%% Train the model with hyperparameters selected after random grid search:
    
def train_network(model,typ, site, epochs, q, adaptive_pooling, dropout_prob = 0.0, change_architecture = False, 
             traindata_perc = None, featuresize = None,  
             save = True, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
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
    X, Y = preprocessing.get_splits(sites = [site],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

    X_test, Y_test = preprocessing.get_splits(sites = [site],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
    
    eval_set = {"X_test":X_test, "Y_test":Y_test}
        
    hparams, model_design = set_model_parameters(model, typ, epochs, change_architecture, adaptive_pooling, X, Y)
   
    start = time.time()
    
    data_dir = os.path.join(data_dir, f"output/models/{model}{typ}")
    
    if adaptive_pooling:
        data_dir = os.path.join(data_dir, r"adaptive_pooling")

        if dropout_prob == 0.0:
            data_dir = os.path.join(data_dir, r"nodropout")
        else:
            data_dir = os.path.join(data_dir, r"dropout")
      
    if not traindata_perc is None:
        data_dir = os.path.join(data_dir, f"data{traindata_perc}perc")
    
    if change_architecture:
        data_dir = os.path.join(data_dir, f"sigmoidActivation")
        
    dev = __import__(f"setup.dev_{model}", fromlist=["selected"])
            
    running_losses,performance, y_tests, y_preds = dev.train_model_CV(hparams, model_design, 
                                                                  X, Y, 
                                                                  eval_set, dropout_prob,
                                                                  data_dir, save)
    end = time.time()
      
    # performance returns: rmse_train, rmse_test, mae_train, mae_test in this order.
    performance = np.mean(np.array(performance), axis=0)
    rets = [(end-start), 
            hparams["hiddensize"], hparams["batchsize"], hparams["learningrate"], hparams["history"], model_design["activation"], 
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["execution_time", "hiddensize", "batchsize", "learningrate", "history", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
    
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
    #return(running_losses, y_tests, y_preds)
    