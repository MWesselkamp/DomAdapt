# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie

This script calls the function pretraining in the wrapper file for pretraining the networks with selected architecture on simulations.
The function takes the arguments:
    
    model (str): network type
    typ (int): model type
    epochs (int): number of epochs
    dropout_prob (float): dropout probability.
    params_distr (str): simulated parameter distribution (normal or uniform). Required for loading the data.
    sims_fraction (int): how much percent of the simulated data (50000 data points) will be used?
    
    q: adds job to Queue.
"""
import multiprocessing as mp

import setup.wrapper_pretraining as wp
    
    
#%%
dropout_prob = 0.0
sims_frac = [30, 50, 70, None]
epochs = [10000,20000,30000,40000]
dropout = False
typ = 7

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(sims_frac)):
        p = mp.Process(target=wp.pretraining, args=("mlp", typ, epochs[i], dropout_prob, dropout, sims_frac[i], q))
        processes.append(p)
        p.start()
