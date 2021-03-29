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
adaptive_pooling = False
dropout_prob = 0.0
dropout = False
sparse = [1, 2, 3,5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,100]
#dropout_prob = 0.0 #[0.0, 0.0, 0.1, 0.1, 0.2, 0.2]
#change_architecture = [False, True, False, True, False, True]

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(sparse)):
        p = mp.Process(target=wt.train_network, args=("mlp", 0, "hyytiala", 5000, q, adaptive_pooling, dropout_prob, dropout, sparse[i]))
        processes.append(p)
        p.start()
