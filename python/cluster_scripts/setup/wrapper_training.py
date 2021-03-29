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
def set_model_parameters(model, typ, epochs, adaptive_pooling,
                         X, Y, change_architecture=False, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
    if (int(typ) == 4):
      print("Typ 4: using multiple layer")
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
      results = results[(results.nlayers == 3)].reset_index()
      best_model = results.iloc[results["mae_val"].idxmin()]
      featuresize = None
    elif (int(typ) == 5):
      print("Using architecture with adaptive Pooling.")
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/adaptive_pooling/grid_search_results_{model}2.csv"))
      best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
      featuresize = best_model["featuresize"]
    elif ( (int(typ) == 7) | ( int(typ) == 8) ):
      print("Using architecture with adaptive Pooling.")
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/simulations/grid_search_results_{model}2.csv"))
      best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
      featuresize = best_model["featuresize"]
    else:
      print("Last choice: architecture 1, no pooling")
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
      best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
      featuresize = None
    
    if change_architecture == True:
      best_model["activation"] == nn.Sigmoid
    
    try:
      hidden_dims = literal_eval(best_model["hiddensize"])
    except ValueError:
      hidden_dims = [best_model["hiddensize"]]
      
    dimensions = [X.shape[1]]
    for hdim in hidden_dims:
        dimensions.append(hdim)
    dimensions.append(Y.shape[1])
    
    try: 
      activation = eval(best_model["activation"][8:-2])
    except: 
      activation = torch.relu
      
    hparams = {"batchsize": int(best_model["batchsize"]), 
               "epochs":epochs, 
               "history": int(best_model["history"]), 
               "hiddensize":hidden_dims,
               "learningrate":best_model["learningrate"]}
      
    model_design = {"dimensions": dimensions,
                    "activation": activation,
                    "featuresize":featuresize}
    
    return hparams, model_design

    
#%% Train the model with hyperparameters selected after random grid search:
    
def train_network(model,typ, site, epochs, q, adaptive_pooling, dropout_prob, dropout, sparse = None, 
             traindata_perc = None, 
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
                                
    if not sparse is None:
        ind = np.random.choice(X.shape[0], int(np.floor(X.shape[0]/100*sparse)), replace = False)
        X, Y = X[ind], Y[ind]

    X_test, Y_test = preprocessing.get_splits(sites = [site],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
    
    eval_set = {"X_test":X_test, "Y_test":Y_test}
        
    hparams, model_design = set_model_parameters(model, typ, epochs, adaptive_pooling, X, Y)

    start = time.time()
    
    data_dir = os.path.join(data_dir, f"output/models/{model}{typ}")
    
    data_dir = os.path.join(data_dir, f"relu")
    
    if not sparse is None:
        data_dir = os.path.join(data_dir, f"sparse//{sparse}")
        
    dev = __import__(f"setup.dev_{model}", fromlist=["selected"])
            
    running_losses,performance, y_tests, y_preds = dev.train_model_CV(hparams, model_design, 
                                                                  X, Y, 
                                                                  eval_set, dropout_prob, dropout,
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
    if not sparse is None:
      np.save(os.path.join(data_dir, "ind.npy"), ind)
    
    #return(running_losses, y_tests, y_preds)
    