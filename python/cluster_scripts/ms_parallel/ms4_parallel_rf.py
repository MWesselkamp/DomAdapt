# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:50:21 2020

@author: marie
"""

#%% Set working directory
import os.path

import setup.preprocessing as preprocessing

import pandas as pd
import time

from setup.dev_rf import rf_selection_parallel
import multiprocessing as mp
import itertools
import setup.utils as utils

#%% Load Data
data_dir = r"/home/fr/fr_fr/fr_mw263"
X, Y = preprocessing.get_splits(sites = ["bily_kriz", "collelongo", "soro"], 
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound", 
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ["bily_kriz", "collelongo", "soro"],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

#%% 
cv_splits = [5]
shuffled = [False]
n_trees = [200,300,400,500]
depth = [4,5,6,7]
eval_set = None

p_list = utils.expandgrid(cv_splits, shuffled, n_trees, depth)

searchsize = len(p_list[0])

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=rf_selection_parallel, args=(X, Y, p_list, eval_set, i, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
    
    results = pd.DataFrame(rets, columns=["run", "cv_splits", "shuffled", "n_trees", "depth", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv("/home/fr/fr_fr/fr_mw263/output/grid_search/grid_search_results_rf4.csv", index = False)
