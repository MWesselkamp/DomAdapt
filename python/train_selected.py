# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:33:52 2020

@author: marie

This script calls the function train_network from the training wrapper script.
This function takes the arguments:
  model
  typ
  site
  epochs
  q
  adaptive_pooling
  dropout_prob
  change_architecure

"""
import multiprocessing as mp
import setup.wrapper_training as wt

#%%
dropout_prob = [0.1, 0.0]

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(dropout_prob)):
        p = mp.Process(target=wt.train_network, args=("mlp", 0, "hyytiala", 10000, q, True, dropout_prob[i], False))
        processes.append(p)
        p.start()
