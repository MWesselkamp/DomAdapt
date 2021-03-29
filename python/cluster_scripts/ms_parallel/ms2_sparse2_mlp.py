# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:36:42 2020

@author: marie
"""

#%% Set working directory
import setup_ms.preprocessing as preprocessing
from setup_ms.dev_mlp import _selection_parallel

import pandas as pd
import time
import os.path
import multiprocessing as mp
import itertools
import torch.nn as nn

#%% Load Data
data_dir = r"/home/fr/fr_fr/fr_mw263"
X, Y = preprocessing.get_splits(sites = ["bily_kriz"], 
                                years = [2005, 2006],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['bily_kriz'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

#%% Grid search of hparams
hiddensize = [8, 16, 32, 64, 128, 256]
batchsize = [8, 16, 32, 64, 128, 256]
learningrate = [1e-4, 1e-3, 5e-3, 1e-2]
history = [0,1,2]
n_layers = [1,2,3]
activation = [nn.ReLU]
featuresize = None #[5, 7,10]
hp_list = [hiddensize, batchsize, learningrate, history, activation, n_layers]

epochs = 500
eval_set = {"X_test":X_test, "Y_test":Y_test}
splits=5
searchsize = 50
hp_search = []


#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=_selection_parallel, args=(X, Y, hp_list, epochs, splits, searchsize, 
                           data_dir, q, hp_search, eval_set))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
        
    print('NN fitting took {} seconds'.format(time.time() - starttime))
    
    if featuresize == None:
      results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history","activation", "nlayers", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    else:
      results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history","activation", "nlayers", "featuresize", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r"/home/fr/fr_fr/fr_mw263/output/sparse/grid_search/grid_search_sparse2_mlp2.csv", index = False)