# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""
#import sys
#sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import setup.preprocessing as preprocessing
import os.path

import pandas as pd
import numpy as np
import multiprocessing as mp

#%%
def train_selected_simulations(model, typ, epochs, dropout_prob, param_distr, q, splits = 5, featuresize = 7, change_architecture = None, 
                  save = True, eval_set = None, data_dir = r"/home/fr/fr_fr/fr_mw263"):
                  
    if param_distr == "normal":
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"scripts/data/simulations/normal_params"), drop_parameters=False)
    else:
      X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"scripts/data/simulations/uniform_params"), drop_parameters=False)
    
    if typ >= 5:
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}1.csv"))
    else:
      results = pd.read_csv(os.path.join(data_dir, f"output/grid_search/grid_search_results_{model}{typ}.csv"))

    best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
    
    if not change_architecture is None:
      
      for item in change_architecture.items():
        best_model[item[0]] = item[1]
    
    dev = __import__(f"setup.dev_{model}", fromlist=["selected"])
    
    #X, Y = X[index], Y[index]
    
    print(f"Fitting {model} {typ} with dropout prob {dropout_prob}")
    
    dev.selected(X, Y, model, typ, best_model, epochs, splits, featuresize,  dropout_prob, 
                 os.path.join(data_dir, "output"), save, eval_set)
    
    
#%%
model = "mlp"
typ = [9, 10]
dropout_prob = [0.0, 0.0]
param_distr = ["normal", "uniform"]
epochs = 200000
# change architecture: {"learningrate":1e-4, "batchsize":512} # {"nlayers":2, "hiddensize":str([32, 64]), "batchsize":512}


if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(typ)):
        p = mp.Process(target=train_selected_simulations, args=(model, typ[i], epochs, dropout_prob[i], param_distr, q))
        processes.append(p)
        p.start()
