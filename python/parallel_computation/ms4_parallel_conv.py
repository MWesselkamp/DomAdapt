# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import os
import os.path
#os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing

import pandas as pd
import time

from dev_cnn import conv_selection_parallel
import multiprocessing as mp
import itertools
import torch.nn as nn

#%% Load Data
data_dir = r"/home/fr/fr_fr/fr_mw263/scripts/"
X, Y = preprocessing.get_splits(sites = ["collelongo", "soro", "le_bray"], 
                                years = [2001,2002,2003,2004,2005,2006, 2007, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound", 
                                simulations = None)

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256, 512]
batchsize = [16, 64, 128, 256, 512]
learningrate = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
history = [5,10,15,20]
channels = [[7,14], [10,20], [14,28]]
kernelsize = [2,3,4]
activation = [nn.Sigmoid, nn.ReLU]

hp_list = [hiddensize, batchsize, learningrate, history, channels, kernelsize, activation]
epochs = 5000
splits = 5
searchsize = 50

#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=conv_selection_parallel, args=(X, Y, hp_list, epochs, splits, searchsize, 
                           data_dir, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "channels", "kernelsize", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    print(results)
    results.to_csv(r"/home/fr/fr_fr/fr_mw263/output/grid_search/grid_search_results_cnn4.csv", index = False)
