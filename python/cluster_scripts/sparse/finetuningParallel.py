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
                                years = [2004, 2005,2006],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                
X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                
setting1 = {"epochs":5000, "pretrained_type":6, "sparse":3, "feature_extraction":None}
setting3 = {"epochs":40000, "pretrained_type":6, "sparse":3, "feature_extraction":["hidden2.weight", "hidden2.bias"]}
setting6 = {"epochs":5000, "pretrained_type":7, "sparse":3, "feature_extraction":None}
setting9 = {"epochs":40000, "pretrained_type":7, "sparse":3, "feature_extraction":["hidden2.weight", "hidden2.bias"]}
setting12 = {"epochs":5000, "pretrained_type":8, "sparse":3, "feature_extraction":None}
setting15 = {"epochs":40000, "pretrained_type":8, "sparse":3, "feature_extraction":["hidden2.weight", "hidden2.bias"]}


settings = [setting1, setting3, setting6, setting9, setting12, setting15]
eval_set = {"X_test":X_test, "Y_test":Y_test}

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for setting in settings:
        p = mp.Process(target=finetune, args=(X, Y, setting["epochs"], "mlp", setting["pretrained_type"], setting["sparse"], setting["feature_extraction"],
                                                eval_set))
        processes.append(p)
        p.start()
        