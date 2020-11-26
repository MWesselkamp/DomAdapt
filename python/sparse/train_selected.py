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
models = [0, 0, 0, 2, 2, 2]
sites = ["bily_kriz", "bily_kriz", "bily_kriz", "hyytiala", "hyytiala", "hyytiala"]
sparse = [1, 2, 3, 1, 2, 3]
epochs = 500

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(models)):
        p = mp.Process(target=wt.train_network, args=("mlp", models[i], sparse[i], sites[i], epochs, q))
        processes.append(p)
        p.start()
