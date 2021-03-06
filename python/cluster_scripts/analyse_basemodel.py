# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:20:39 2020

@author: marie
"""


import setup.preprocessing as preprocessing
import os.path

import multiprocessing as mp
import pandas as pd
import numpy as np

#%% Load Data: Profound in and out.
data_dir = r"/home/fr/fr_fr/fr_mw263"

X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

def subset_data(data, perc):
    
    n_subset = int(np.floor(data.shape[0]/100*perc))
    subset = data[:n_subset,:]
    
    return(subset)
    
#%%
def train_selected(X, Y, model, typ, epochs, q, 
                   traindata_perc = 100, dropout_prob = 0.0, simtype=None, save = True, change_architecture = False,
                   eval_set = {"X_test":X_test, "Y_test":Y_test}, 
                   data_dir = os.path.join(data_dir, "output")):

    results = pd.read_csv(os.path.join(data_dir, f"models/mlp0/adaptive_pooling/nodropout/sigmoid/selected_results.csv"))
      
    best_model = results.iloc[results['mae_val'].idxmin()].to_dict()
    # featuresize somehow not in the results. see best model, architecture 3.
    best_model["featuresize"] = 10

    dev = __import__(f"setup.dev_{model}", fromlist=["selected"])
    
    X_subset, Y_subset = subset_data(X, traindata_perc), subset_data(Y, traindata_perc)
    
    dev.selected(X_subset, Y_subset, model, typ, best_model, epochs, change_architecture,
                 traindata_perc, simtype, best_model["featuresize"], dropout_prob, 
                 data_dir, save, eval_set)
    

#%%
model = "mlp"
typ = 0
epochs = 10000

datasteps = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,75, 80,85, 90,95]

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(datasteps)):
        p = mp.Process(target=train_selected, args=(X, Y, model, typ, epochs, q, datasteps[i]))
        processes.append(p)
        p.start()
