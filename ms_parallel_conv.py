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
#%% Load Data
datadir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ["hyytiala"], 
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound")

#%% Grid search of hparams
hiddensize = [64, 128, 256]
batchsize = [8, 64, 128]
learningrate = [1e-3, 5e-3, 1e-2, 5e-2]
history = [10, 20, 30]
channels = [[7,14], [10,20], [14,28]]
kernelsize = [2,3]

hp_list = [hiddensize, batchsize, learningrate, history, channels, kernelsize]
epochs = 300
splits=6
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
        
    print('ConvNet fitting took {} seconds'.format(time.time() - starttime))
    
    results = pd.DataFrame(rets, columns=["run", "execution_time", "hiddensize", "batchsize", "learningrate", "history", "channels", "kernelsize", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(datadir, r"plots\data_quality_evaluation\fits_nn\convnet\grid_search_results.csv"), index = False)
