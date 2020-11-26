# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:01:38 2020

@author: marie
"""

#%%
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import numpy as np
import pandas as pd
import os.path
import visualizations

import finetuning
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

#%%
rets_sparse1_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse1_mlp2.csv"))
rets_sparse2_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse2_mlp2.csv"))
rets_sparse3_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse3_mlp2.csv"))
rets_sparse4_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse4_mlp2.csv"))
rets_sparse5_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\sparse\grid_search\500\grid_search_sparse5_mlp2.csv"))

visualizations.hparams_optimization_errors([rets_sparse1_mlp, rets_sparse2_mlp, rets_sparse3_mlp, rets_sparse4_mlp, rets_sparse5_mlp], 
                                           ["1year", "2years",  "3years", "4years",  "5years"], 
                                           error="mae",
                                           train_val = True)

#%%
l = visualizations.losses("mlp", 0, r"sparse1", sparse=True)
l = visualizations.losses("mlp", 0, r"sparse2", sparse=True)
l = visualizations.losses("mlp", 0, r"sparse3", sparse=True)
l = visualizations.losses("mlp", 0, r"sparse4", sparse=True)
l = visualizations.losses("mlp", 0, r"sparse5", sparse=True)
visualizations.losses("mlp", 2, r"sparse1", sparse=True)
visualizations.losses("mlp", 2, r"sparse2", sparse=True)
visualizations.losses("mlp", 2, r"sparse3", sparse=True)
visualizations.losses("mlp", 2, r"sparse4", sparse=True)
visualizations.losses("mlp", 2, r"sparse5", sparse=True)
#%% OLS with sparse data 

predictions_test, errors = finetuning.featureExtractorC("mlp", 6, None, 30, classifier = "ols", 
                      years = [2006])
errors1 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 6, None, 30, classifier = "ols", 
                      years = [2005, 2006])
errors2 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 6, None, 30, classifier = "ols", 
                      years = [2004, 2005, 2006])
errors3 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 6, None, 30, classifier = "ols", 
                      years = [2003, 2004, 2005, 2006])
errors4 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 6, None, 30, classifier = "ols", 
                      years = [2001, 2003, 2004, 2005, 2006])
errors5 = np.mean(np.array(errors), axis=1)
#%%
predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = [2006])
errors1 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = [2005, 2006])
errors2 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = [2004, 2005, 2006])
errors3 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = [2003, 2004, 2005, 2006])
errors4 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = [2001, 2003, 2004, 2005, 2006])
errors5 = np.mean(np.array(errors), axis=1)
#%%
predictions_test, errors = finetuning.featureExtractorC("mlp", 8, None, 30, classifier = "ols", 
                      years = [2006])
errors1 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 8, None, 30, classifier = "ols", 
                      years = [2005, 2006])
errors2 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 8, None, 30, classifier = "ols", 
                      years = [2004, 2005, 2006])
errors3 = np.mean(np.array(errors), axis=1)

predictions_test, errors = finetuning.featureExtractorC("mlp", 8, None, 30, classifier = "ols", 
                      years = [2003, 2004, 2005, 2006])
errors4 = np.mean(np.array(errors), axis=1)


#%% full backprob and frozen weights.

l = visualizations.losses("mlp", 6, r"sparse1/setting0", sparse=True)
l = visualizations.losses("mlp", 7, r"sparse1/setting0", sparse=True)
l = visualizations.losses("mlp", 8, r"sparse1/setting0", sparse=True)

l = visualizations.losses("mlp", 6, r"sparse1/setting1", sparse=True)
l = visualizations.losses("mlp", 7, r"sparse1/setting1", sparse=True)
l = visualizations.losses("mlp", 8, r"sparse1/setting1", sparse=True)

#%%
