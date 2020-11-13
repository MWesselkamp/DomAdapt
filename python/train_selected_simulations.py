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
dropout_prob = [0.0, 0.1]

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(dropout_prob)):
        p = mp.Process(target=wp.pretraining, args=("mlp", 7, 100000, dropout_prob[i], 50, q))
        processes.append(p)
        p.start()
