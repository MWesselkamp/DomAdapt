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
dropout_prob = [6, 7, 8 ,9]
sims_frac =  None
epochs = 40000
dropout = True


if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(dropout_prob)):
        p = mp.Process(target=wp.pretraining, args=("mlp", 5, epochs, dropout_prob[i], dropout, sims_frac, q))
        processes.append(p)
        p.start()
