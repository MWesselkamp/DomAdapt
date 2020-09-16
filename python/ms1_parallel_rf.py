# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:50:21 2020

@author: marie
"""

#%% Set working directory
import os.path

import preprocessing

import pandas as pd
import time

from dev_rf import rf_selection_parallel
import multiprocessing as mp
import itertools
import utils

#%% Load Data
datadir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"
X, Y = preprocessing.get_splits(sites = ["le_bray"], 
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulation = None)

#%% 
cv_splits = [6]
shuffled = [False]
n_trees = [200,300,400,500]
depth = [4,5,6,7]

p_list = utils.expandgrid(cv_splits, shuffled, n_trees, depth)

searchsize = len(p_list[0])

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=rf_selection_parallel, args=(X, Y, p_list, i, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
        
    print('RF fitting took {} seconds'.format(time.time() - starttime)) 
    
    results = pd.DataFrame(rets, columns=["run", "cv_splits", "shuffled", "n_trees", "depth", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(datadir, r"plots\data_quality_evaluation\fits_rf\grid_search_results_rf1.csv"), index = False)
