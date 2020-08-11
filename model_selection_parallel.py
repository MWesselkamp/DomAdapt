# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import os
os.getcwd()
#os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing

import pandas as pd
import time

from experiment import nn_selection_parallel
from random_forest import rf_selection_parallel
import multiprocessing as mp
import itertools
import utils

#%% Load Data
X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256]
batchsize = [2, 8, 64, 128, 256]
learningrate = [7e-3, 1e-2, 3e-2, 8e-2]
history = [0,1,2]
hp_list = [hiddensize, batchsize, learningrate, history]
splits=6
searchsize = 25

#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=nn_selection_parallel, args=(X, Y, hp_list,splits, i, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
        
    print('NN fitting took {} seconds'.format(time.time() - starttime))
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(r'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn\grid_search_results.csv', index = False)
#%%
#results = pd.DataFrame(hp_search, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "rmse_train", "rmse_val", "mae_train", "mae_val"])
#results.to_csv(r'plots\data_quality_evaluation\fits_nn\grid_search_results.csv', index = False)
#print("Best Model Run: \n", results.iloc[results['RSME_val'].idxmin()]) 
#results.iloc[results['RSME_val'].idxmin()].to_dict()
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
    results.to_csv(r'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf\grid_search_results.csv', index = False)
