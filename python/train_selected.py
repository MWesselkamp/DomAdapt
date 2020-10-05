# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import preprocessing
import os.path

import pandas as pd
import multiprocessing as mp
import itertools

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%%
def train_selected(X, Y, model, typ, epochs, splits, save, eval_set, finetuning, feature_extraction, q,
                   data_dir):

    results = pd.read_csv(os.path.join(data_dir, f"grid_search\grid_search_results_{model}{typ}.csv"))

    best_model = results.iloc[results['rmse_val'].idxmin()].to_dict()

    dev = __import__(f"dev_{model}")
    
    dev.selected(X, Y, model, best_model, epochs, splits, data_dir, save, eval_set, finetuning)
    
    #out = [running_losses, y_tests, y_preds]
    
    #q.put(out)
    
#%%
models = ["mlp"]
typ = 1
epochs = 2
splits = 6
save=True
eval_set = {"X_test":X_test, "Y_test":Y_test}
finetuning = False
feature_extraction=False
data_dir = os.path.join(datadir, "python\outputs")

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(models)):
        p = mp.Process(target=train_selected, args=(X, Y, models[i], typ, epochs, splits, 
                                                    save, eval_set, finetuning, feature_extraction, q, data_dir))
        processes.append(p)
        p.start()

    #for p in processes:
    #    ret = itertools.chain(*q.get())
    #    rets.append(list(ret))
    #    p.join()
    
    #for i in range(len(models)):
    #    print(rets[i])
        #with open(os.path.join(data_dir, f"{models[i]}\\running_losses.txt"), "w") as f:
            #f.write(json.dumps(rets[i]))