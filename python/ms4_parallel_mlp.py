# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:36:42 2020

@author: marie
"""

#%% Set working directory
import os.path
import preprocessing

import pandas as pd
import time

from dev_mlp import mlp_selection_parallel
import multiprocessing as mp
import itertools
#%% Load Data
data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"

X, Y = preprocessing.get_splits(sites = ['le_bray', 'bily_kriz','collelongo'],
                                years = [2001,2003,2004,2005,2006],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['le_bray', 'bily_kriz','collelongo'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256, 512]
batchsize = [8, 64, 128, 256, 512]
learningrate = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
history = [0,1,2]
hp_list = [hiddensize, batchsize, learningrate, history]
epochs = 3000
splits= 15
searchsize = 50
hp_search = []
eval_set = {"X_test":X_test, "Y_test":Y_test}

#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=mlp_selection_parallel, args=(X, Y, hp_list, epochs, splits, searchsize, data_dir, q, 
                                                            hp_search, eval_set))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
        
    print('NN fitting took {} seconds'.format(time.time() - starttime))
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"outputs\grid_search\grid_search_results_mlp4.csv"), index = False)