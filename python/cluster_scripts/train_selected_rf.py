# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:21:54 2020

@author: marie
"""
import os.path
import setup.preprocessing as preprocessing
import pandas as pd
import numpy as np
import setup.dev_rf as dev_rf

#%%
data_dir = r"/home/fr/fr_fr/fr_mw263"
rets_rf = pd.read_csv(os.path.join(data_dir, r"output/grid_search/grid_search_results_rf2.csv"))
rfp = rets_rf.iloc[rets_rf['mae_val'].idxmin()].to_dict()
#%%

X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002, 2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "scripts/data"), 
                                dataset = "profound",
                                simulations = None)
                                
y_preds, y_tests, errors = dev_rf.random_forest_CV(X, Y, 
                                                   rfp["cv_splits"], False, rfp["n_trees"], rfp["depth"], 
                                                   eval_set = {"X_test":X_test, "Y_test":Y_test},
                                                   selected = False)

np.save(os.path.join(data_dir, r"output/models/rf0/errors.npy"), errors)
np.save(os.path.join(data_dir, r"output/models/rf0/y_tests.npy"), y_tests)
np.save(os.path.join(data_dir, r"output/models/rf0/y_preds.npy"), y_preds)
