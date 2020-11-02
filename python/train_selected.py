# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""
#import sys
#sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import preprocessing
import os.path

import multiprocessing as mp
import pandas as pd

#%% Load Data: Profound in and out.
data_dir = r"/home/fr/fr_fr/fr_mw263"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002, 2003, 2004],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                #simulations = "preles", drop_cols=True)

#%%
def train_selected(X, Y, model, typ, epochs, q, 
                   splits = 5, featuresize = None, dropout_prob = 0.0, save = True, eval_set = None, 
                   data_dir = os.path.join(data_dir, "output")):

    if (typ >= 4) | (typ==0):
      results = pd.read_csv(os.path.join(data_dir, f"grid_search/grid_search_results_{model}1.csv"))
    else:
      results = pd.read_csv(os.path.join(data_dir, f"grid_search/grid_search_results_{model}{typ}.csv"))
      
    best_model = results.iloc[results['mae_val'].idxmin()].to_dict()

    dev = __import__(f"dev_{model}")
    
    dev.selected(X, Y, model, typ, best_model, epochs, splits, featuresize, dropout_prob, data_dir, save, eval_set)
    
#%%
models = ["mlp"]
typ = 0
epochs = 10000

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(models)):
        p = mp.Process(target=train_selected, args=(X, Y, models[i], typ, epochs, q))
        processes.append(p)
        p.start()
