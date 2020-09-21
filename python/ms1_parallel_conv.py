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

from dev_convnet import conv_selection_parallel
import multiprocessing as mp
import itertools
import torch.nn as nn
#%% Load Data
datadir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ["le_bray"], 
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound", 
                                simulations = None)

#%% Grid search of hparams
hiddensize = [16, 64, 128, 256, 512]
batchsize = [16, 64, 128, 256, 512]
learningrate = [1e-3, 5e-3, 1e-2, 5e-2]
history = [5,10,15,20]
channels = [[7,14], [10,20], [14,28]]
kernelsize = [2,3,4]
activation = [nn.Sigmoid, nn.ReLU]

hp_list = [hiddensize, batchsize, learningrate, history, channels, kernelsize, activation]
epochs = 20
splits = 6
searchsize = 20

#%% multiprocessed model selection with searching random hparam combinations from above.

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    starttime = time.time()
    processes = []
    rets =[]
    
    for i in range(searchsize):
        p = mp.Process(target=conv_selection_parallel, args=(X, Y, hp_list, epochs, splits, i, datadir, q))
        processes.append(p)
        p.start()

    for p in processes:
        ret = itertools.chain(*q.get())
        rets.append(list(ret))
        p.join()
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "channels", "kernelsize", "activation", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(datadir, r"python\plots\data_quality_evaluation\fits_nn\convnet\grid_search_results_cnn1.csv"), index = False)
