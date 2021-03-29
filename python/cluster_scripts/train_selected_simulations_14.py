# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie
"""
#import sys
#sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import setup.preprocessing as preprocessing
import setup.wrapper_pretraining_9_12 as wp
import os.path

import pandas as pd
import multiprocessing as mp
    
    
#%%

typ = 14
dropout_prob = [0.0, 0.0, 0.0, 0.0]
sims_frac = [30, 50, 70, None]
epochs = [20000, 40000, 60000, 70000]
dropout = False


if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(sims_frac)):
        p = mp.Process(target=wp.pretraining, args=("mlp", 14, epochs[i], dropout_prob[i], dropout, sims_frac[i], q))
        processes.append(p)
        p.start()

