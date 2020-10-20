# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')
import dev_mlp
import pandas as pd
import os.path
import preprocessing
import torch.nn as nn
import visualizations
# Finetuning

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2002,2003,2004,2005,2006,2007, 2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)


#%%
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))

res_mlp = rets_mlp.iloc[rets_mlp['rmse_val'].idxmin()].to_dict()

#%%
model = "mlp"
typ = 5
epochs = 5000
splits = 6
dimensions = [12,literal_eval(res_mlp["hiddensize"])[0],Y.shape[1]]

hparams = {"batchsize": int(res_mlp["batchsize"]), 
           "epochs":epochs, 
           "history": int(res_mlp["history"]), 
           "hiddensize":literal_eval(res_mlp["hiddensize"]),
           "learningrate":res_mlp["learningrate"]}

model_design = {"dimensions": dimensions,
                "activation": nn.ReLU}

save = False
eval_set = None
finetuning=True
feature_extraction=False
   
#%%
running_losses,performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, X, Y, splits, 
                                                                      eval_set,os.path.join(datadir, f"python\outputs\models\mlp5") , 
                                                                      save, finetuning, feature_extraction)


#%%
visualizations.plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"], "anything", "mlp")
print(np.mean(np.array(performance), axis=0))