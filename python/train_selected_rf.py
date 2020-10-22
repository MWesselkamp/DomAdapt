# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:21:54 2020

@author: marie
"""
import os.path
import preprocessing
import pandas as pd
import numpy as np
import dev_rf

#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
rets_rf = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\rf\grid_search_results_rf1.csv"))
rfp = rets_rf.iloc[rets_rf['rmse_val'].idxmin()].to_dict()
#%%

X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005, 2006, 2007, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

y_preds, y_tests, errors = dev_rf.random_forest_CV(X, Y, 
                                                   rfp["cv_splits"], False, rfp["n_trees"], rfp["depth"])

np.save(os.path.join(data_dir, r"python\outputs\models\rf1\errors.npy"), errors)
np.save(os.path.join(data_dir, r"python\outputs\models\rf1\y_tests.npy"), y_tests)
np.save(os.path.join(data_dir, r"python\outputs\models\rf1\y_preds.npy"), y_preds)

#%%
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
                                #simulations = "preles", drop_cols=True)
                                
y_preds, y_tests, errors = dev_rf.random_forest_CV(X, Y, 
                                                   rfp["cv_splits"], False, rfp["n_trees"], rfp["depth"],
                                                   eval_set ={"X_test":X_test, "Y_test":Y_test})

np.save(os.path.join(data_dir, r"python\outputs\models\rf2\errors.npy"), errors)
np.save(os.path.join(data_dir, r"python\outputs\models\rf2\y_tests.npy"), y_tests)
np.save(os.path.join(data_dir, r"python\outputs\models\rf2\y_preds.npy"), y_preds)