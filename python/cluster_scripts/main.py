# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import finetuning
import setup.preprocessing as preprocessing
import visualizations

import os.path
import numpy as np

#%% Load Data: Profound in and out.
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)


#%%
pretrained_model = visualizations.losses("mlp", 7, "") 

running_losses,performance, y_tests, y_preds  = finetuning.finetune(X, Y, epochs = 100, model="mlp", pretrained_type=7)
#%%
visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], "", "mlp")
print(np.mean(np.array(performance), axis=0))

res_mlp = visualizations.losses("mlp", 0, "") 


#%%
import setup.models as models
import torch
model = models.MLPmod(7, [64,64,16,1], nn.ReLU)
model.load_state_dict(torch.load(os.path.join(datadir, f"python\outputs\models\mlp6\model0.pth")))
