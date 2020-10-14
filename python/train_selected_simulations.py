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
#%% Load Data: Profound in and out.
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, "data\preles\simulations"))

#%%
def train_selected(X, Y, model, typ, epochs, splits, save, eval_set, finetuning, feature_extraction,data_dir, q
                   ):

    results = pd.read_csv(os.path.join(data_dir, f"grid_search\{model}\grid_search_results_{model}{typ}.csv"))

    best_model = results.iloc[results['rmse_val'].idxmin()].to_dict()

    dev = __import__(f"dev_{model}")
    
    dev.selected(X, Y, model, typ, best_model, epochs, splits, data_dir, save, eval_set, finetuning)
    
    #out = [running_losses, y_tests, y_preds]
    
    #return(out)
    #q.put(out)
    
#%%
models = ["mlp"]
typ = 5
epochs = 4000
splits = 5
save=True
eval_set = None #{"X_test":X_test, "Y_test":Y_test}
finetuning = False
feature_extraction=False
data_dir = os.path.join(data_dir, "python\outputs")

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(models)):
        p = mp.Process(target=train_selected, args=(X_sims, Y_sims, models[i], typ, epochs, splits, 
                                                    save, eval_set, finetuning, feature_extraction, data_dir, q))
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