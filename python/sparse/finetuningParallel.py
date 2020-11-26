# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:54:27 2020

@author: marie
"""
import setup.preprocessing as preprocessing
from finetuning import finetune

import multiprocessing as mp
import os.path
import numpy as np
import pandas as pd

data_dir = r"/home/fr/fr_fr/fr_mw263"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                
X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                
    
setting1 = {"epochs":20000, "pretrained_type":7, "params_distr":"normal", "feature_extraction":None}
setting2 = {"epochs":300000, "pretrained_type":7, "params_distr":"normal", "feature_extraction":["hidden2.weight", "hidden3.weight"]}

settings = [None, ["hidden2.weight", "hidden3.weight"]]
eval_set = {"X_test":X_test, "Y_test":Y_test}

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for i in range(len(settings)):
        p = mp.Process(target=finetune, args=(X, Y, 40000, "mlp", 7, r"adaptive_pooling/nodropout", 7, True, settings[i],
                                                eval_set, data_dir))
        processes.append(p)
        p.start()
        