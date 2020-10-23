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
from ast import literal_eval
import numpy as np

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)


#%%
rets_mlp = pd.read_csv(os.path.join(datadir, r"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
res_mlp = rets_mlp.iloc[rets_mlp['mae_val'].idxmin()].to_dict()
results = visualizations.losses("mlp", 6, "") 
#%%
model = "mlp"
typ = 6
splits = 5
dimensions = [X.shape[1]]
for hs in literal_eval(res_mlp["hiddensize"]):
    dimensions.append(hs)
dimensions.append(Y.shape[1])

hparams = {"batchsize": int(res_mlp["batchsize"]), 
           "epochs":10000, 
           "history": int(res_mlp["history"]), 
           "hiddensize":literal_eval(res_mlp["hiddensize"]),
           "learningrate":res_mlp["learningrate"]}

model_design = {"dimensions": dimensions,
                "activation": nn.ReLU}

   
#%%
running_losses,performance, y_tests, y_preds = dev_mlp.finetuning_CV(hparams, model_design, X, Y, splits, 
                                                                      eval_set=None, data_dir = os.path.join(datadir, f"python\outputs\models\mlp6") , 
                                                                      save=False, feature_extraction=False)


#%%
visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], "", "mlp")
print(np.mean(np.array(performance), axis=0))

res_mlp = visualizations.losses("mlp", 0, "") 
