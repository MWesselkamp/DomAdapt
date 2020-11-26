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
                                
setting1 = {"epochs":10000, "pretrained_type":6, "simsfrac":30, "feature_extraction":None}
setting2 = {"epochs":10000, "pretrained_type":6, "simsfrac":50, "feature_extraction":None}
setting3 = {"epochs":40000, "pretrained_type":6, "simsfrac":30, "feature_extraction":["hidden2.weight", "hidden2.bias"]}
setting4 = {"epochs":40000, "pretrained_type":6, "simsfrac":50, "feature_extraction":["hidden2.weight", "hidden2.bias"]}
setting5 = {"epochs":10000, "pretrained_type":6, "simsfrac":70, "feature_extraction":None}
setting6 = {"epochs":10000, "pretrained_type":6, "simsfrac":100, "feature_extraction":None}
setting7 = {"epochs":40000, "pretrained_type":6, "simsfrac":70, "feature_extraction":["hidden2.weight", "hidden2.bias"]}
setting8 = {"epochs":40000, "pretrained_type":6, "simsfrac":100, "feature_extraction":["hidden2.weight", "hidden2.bias"]}


settings = [setting1, setting2, setting3, setting4, setting5, setting6, setting7, setting8]
eval_set = {"X_test":X_test, "Y_test":Y_test}

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for setting in settings:
        p = mp.Process(target=finetune, args=(X, Y, setting["epochs"], "mlp", setting["pretrained_type"], f"nodropout/sims_frac{setting['simsfrac']}", setting["feature_extraction"],
                                                eval_set))
        processes.append(p)
        p.start()
        