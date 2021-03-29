# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:01:17 2020

@author: marie
"""
import numpy as np

import setup.dev_mlp as dev_mlp
import setup.preprocessing as preprocessing

import time
import pandas as pd
import os.path
import torch
from ast import literal_eval



#%%
    
def set_model_parameters(model, typ, epochs, featuresize, 
                         X, Y, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
    results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}2.csv"))
    if ((typ == 11) | (typ==12)| (typ==14)):
      results = results[(results.nlayers == 3)].reset_index()
    
    best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
    try:
        featuresize = best_model["featuresize"]
    except:
        featuresize = None
        
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

def pretraining(model, typ, epochs, dropout_prob, dropout, sims_fraction, q, simtype = None, featuresize = 7, 
                  save = True, eval_set = None, data_dir = r"/home/fr/fr_fr/fr_mw263"):
    
    """
    
    Args:
        simtype: One out of None (for models 7 and 8) or ParamsFix, normal_params, uniform_params
    """
    # Load the simulations
    
    if typ == 7:
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"scripts/data/simulations/normal_params"), drop_parameters=False)
    if typ == 8:
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"scripts/data/simulations/uniform_params"), drop_parameters=False)
    if typ ==6:
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"scripts/data/simulations/paramsFix"), drop_parameters=True)
    # Architectures selected for Observations.
    if typ == 5:
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"scripts/data/simulations/paramsFix"), drop_parameters=True)
    if ((typ == 9) | (typ == 11)):
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"scripts/data/simulations/paramsFix"), drop_parameters=True)
    if ((typ == 10) | (typ == 12)):
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"scripts/data/simulations/uniform_params"), drop_parameters=True)
    if ((typ == 13) | (typ == 14)):
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"scripts/data/simulations/uniform_params"), drop_parameters=False)
    
    # Use full simulations or only parts of it?
    if not sims_fraction is None:
        
        print("Using only", sims_fraction, "% of the Simulations")
        ind = int(np.floor(X.shape[0]/100*sims_fraction))
        X, Y = X[:ind], Y[:ind]
    
    # Set the hyperparameters and architecture of the network from preselection.
    hparams, model_design = set_model_parameters(model, typ, epochs, featuresize, X, Y)
    # Directory where fitted network is saved to
    if dropout_prob == 0.0:
      data_dir = os.path.join(data_dir, f"output/models/{model}{typ}/nodropout")
    else:
      data_dir = os.path.join(data_dir, f"output/models/{model}{typ}/dropout")
      
    if not sims_fraction is None:
      data_dir = os.path.join(data_dir, f"sims_frac{sims_fraction}")
    else:
      data_dir = os.path.join(data_dir, f"sims_frac100")
     
    # Load script from which to use train_model_CV
    dev = __import__(f"setup.dev_{model}", fromlist=["selected"])
    
    # Train model
    start = time.time()
    
    running_losses,performance, y_tests, y_preds = dev.train_model_CV(hparams, model_design, 
                                                                  X, Y, 
                                                                  eval_set, dropout_prob, dropout,
                                                                  data_dir, save)
    end = time.time()
    
    # Save: Results
    if not simtype is None:
      data_dir = os.path.join(data_dir, f"{simtype}")
      
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
    