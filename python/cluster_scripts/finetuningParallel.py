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


setting1 = {"epochs":10000, "pretrained_type":5, "simsfrac":100, "feature_extraction":["hidden1.weight", "hidden1.bias"], "dummies":True}  
#setting2 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":None, "dummies":False} 
#setting3 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":None, "dummies": True} 
#setting4 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":None, "dummies":True} 
#setting5 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":None, "dummies":False}
#setting6 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":None, "dummies":True}  

#setting7 = {"epochs":5000, "pretrained_type":10, "simsfrac":100, "feature_extraction":["hidden1.weight", "hidden1.bias"], "dummies":False}  
#setting8 = {"epochs":5000, "pretrained_type":12, "simsfrac":100, "feature_extraction":["hidden3.weight", "hidden3.bias"], "dummies":False} 
#setting9 = {"epochs":5000, "pretrained_type":13, "simsfrac":100, "feature_extraction":["hidden1.weight", "hidden1.bias"], "dummies":True} 
#setting10 = {"epochs":5000, "pretrained_type":14, "simsfrac":100, "feature_extraction":["hidden3.weight", "hidden3.bias"], "dummies":True} 
#setting11 = {"epochs":5000, "pretrained_type":7, "simsfrac":100, "feature_extraction":["hidden0.weight", "hidden0.bias","hidden1.weight", "hidden1.bias"], "dummies":False}
#setting12 = {"epochs":5000, "pretrained_type":5, "simsfrac":100, "feature_extraction":["hidden0.weight", "hidden0.bias", "hidden1.weight", "hidden1.bias"], "dummies":True} 


#settings = [setting1, setting2, setting3, setting4, setting5, setting6, setting7, setting8, setting9, setting10, setting11, setting12 ]
#, setting13, setting14, setting15, setting16] # , #]#,setting9, setting10, setting11, setting12, 

eval_set = {"X_test":X_test, "Y_test":Y_test}
sparse = None#[1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
dropout_prob = [0,1,2,3,4,5,6,7,8,9]
dropout = False

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for dp in dropout_prob:
        p = mp.Process(target=finetune, args=(X, Y, setting1["epochs"], "mlp", setting1["pretrained_type"], f"dropout/0{dp}/sims_frac{setting1['simsfrac']}", setting1["feature_extraction"], eval_set, 
                                              True, sparse, setting1["dummies"], dropout_prob, dropout))
        processes.append(p)
        p.start()
        