# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""
#import sys
#sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import setup.preprocessing as preprocessing
import setup.wrapper_pretraining as wp
import os.path

import pandas as pd
import multiprocessing as mp
    
    
#%%

typ = 5
dropout_prob = [0.0, 0.0, 0.0, 0.0]
sims_frac = [30, 50, 70, None]
epochs = [10000, 20000, 30000, 40000]


if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(sims_frac)):
        p = mp.Process(target=wp.pretraining, args=("mlp", 5, epochs[i], dropout_prob[i], sims_frac[i], q))
        processes.append(p)
        p.start()
