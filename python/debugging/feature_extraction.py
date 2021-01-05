# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:02:20 2020

@author: marie
"""

import os.path
import numpy as np
import pandas as pd
import setup.models as models
import torch.nn as nn
import torch
from ast import literal_eval
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
rets_mlp = pd.read_csv(os.path.join(data_dir, r"python\outputs\grid_search\observations\mlp\grid_search_results_mlp2.csv"))
rets_mlp = rets_mlp[(rets_mlp.nlayers == 3)].reset_index()
bm = rets_mlp.iloc[rets_mlp['mae_val'].idxmin()].to_dict()

model = models.MLP([7,128, 16, 256,1], nn.ReLU)
model.load_state_dict(torch.load(os.path.join(data_dir, r"python\outputs\models\mlp4\nodropout\sims_frac100\model0.pth")))
#%%
for child in model.children():
    print(child)
    for name, parameter in child.named_parameters():
        print(name)
        print(parameter)
        if not name in ["hidden3.weight", "hidden3.bias"]:
            print("disable backprob for", name)
            parameter.requires_grad = False
#%%
pars=[hidden3.weight, hidden3.bias]
model.hidden3.weight.requires_grad = False
