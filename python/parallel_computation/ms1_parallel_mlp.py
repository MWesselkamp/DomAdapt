# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:47:11 2020

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
import torch.nn as nn
#%% Load Data
datadir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ["le_bray"], 
                                years = [2001,2002,2003,2004,2005,2006, 2007, 2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256, 512]
batchsize = [16, 64, 128, 256, 512]
learningrate = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
history = [0,1,2]
activation = [nn.Sigmoid, nn.ReLU]
n_layers = [1,2,3]

hp_list = [hiddensize, batchsize, learningrate, history, activation, n_layers]
epochs = 4000
splits=6
searchsize = 50

#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=mlp_selection_parallel, args=(X, Y, hp_list, epochs, splits, i, datadir, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history","activation", "nlayers", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\mlp\grid_search_results_mlp_ex.csv"), index = False)