# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""

import preprocessing
import os.path

import pandas as pd
import multiprocessing as mp
import itertools
import json

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007, 2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None,
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR","DOY_sin", "DOY_cos"],
                                to_numpy = True)

#%%
def train_selected(X, Y, model, typ, epochs, q,
                   datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\plots\data_quality_evaluation"):

    results = pd.read_csv(os.path.join(datadir, f"grid_search_results_{model}{typ}.csv"))

    best_model = results.iloc[results['rmse_val'].idxmin()].to_dict()

    dev = __import__(f"dev_{model}")
    
    running_losses, y_tests, y_preds = dev.selected(X, Y, 
                                                    best_model, 
                                                    epochs, splits=6, datadir=datadir)
    
    out = [running_losses, y_tests, y_preds]
    
    return(out)
    
#%%
models = ["mlp", "cnn"]
typ = 1
epochs = 2

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(models)):
        p = mp.Process(target=train_selected, args=(X, Y, models[i], typ, epochs, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
    
    with open("test.txt", "w") as f:
        f.write(json.dumps(rets))