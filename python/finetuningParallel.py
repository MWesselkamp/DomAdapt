# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:54:27 2020

@author: marie
"""


import finetuning
import setup.preprocessing as preprocessing

import multiprocessing as mp
import os.path
import numpy as np
import pandas as pd


def train_pretrained(settings, q, data_dir):
    
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007,2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    
    running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 
                                                                       epochs = settings["epochs"], 
                                                                       model="mlp",
                                                                       pretrained_type=settings["pretrained_type"], 
                                                                       feature_extraction=settings["feature_extraction"])

    performance = np.mean(np.array(performance), axis=0)
    rets = [settings["epochs"], 
            settings["pretrained_type"],
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["epochs", "pretrained_type", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
    
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
setting1 = {"epochs":3000, "pretrained_type":7, "feature_extraction":None}
setting2 = {"epochs":3000, "pretrained_type":7, "feature_extraction":["hidden.weights2","hidden.weights3"]}

settings = [setting1, setting2]

if __name__ == '__main__':
    #freeze_support()
    
    q = mp.Queue()
    
    processes = []
    rets =[]
    
    for setting in settings:
        p = mp.Process(target=train_pretrained, args=(setting, q))
        processes.append(p)
        p.start()
        