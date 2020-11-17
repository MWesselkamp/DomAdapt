# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:05:12 2020

@author: marie
"""

#%% Set working directory
import os.path
import setup.preprocessing as preprocessing

import pandas as pd
import numpy as np
import time

from setup.dev_lstm import _selection_parallel
import multiprocessing as mp
import itertools
import torch.nn.functional as F
#%% Load Data
data_dir = r"/home/fr/fr_fr/fr_mw263"
X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"scripts/data/simulations/uniform_params"), drop_parameters=False)
# Use only first 10 percent of data.
ind = int(np.floor(X.shape[0]/100*10))
X, Y = X[:ind], Y[:ind]

#%% Grid search of hparams
hiddensize = [8, 16, 32, 64, 128, 256]
batchsize = [8, 16, 32, 64, 128, 256]
learningrate = [1e-4, 1e-3, 5e-3, 1e-2]
history = [3, 7, 14]
activation = [F.relu]
hp_list = [hiddensize, batchsize, learningrate, history, activation]

epochs = 20000
splits = 5
searchsize = 30
hp_search = []
eval_set = None #{"X_test":X_test, "Y_test":Y_test}

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
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history","activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r"/home/fr/fr_fr/fr_mw263/output/grid_search/simulations/grid_search_results_lstm2.csv", index = False)