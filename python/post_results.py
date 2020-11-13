# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:32:31 2020

@author: marie
"""

import numpy as np
import pandas as pd
import os.path
import setup.models as models
from ast import literal_eval
import torch.nn as nn
import torch

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"

#%% Part I: Reduced Amount of data
mae_vals = []
mae_trains = []

for i in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95]:
    
    preds = np.load(os.path.join(data_dir, f"outputs\models\mlp0\AdaptPool\dropout\data{i}perc\y_preds.npy"))
    tests = np.load(os.path.join(data_dir, f"outputs\models\mlp0\AdaptPool\dropout\data{i}perc\y_tests.npy"))
    res = pd.read_csv(os.path.join(data_dir, f"outputs\models\mlp0\AdaptPool\dropout\data{i}perc\selected_results.csv"))
    
    mae_vals.append(res["mae_val"].item())
    mae_trains.append(res["mae_train"].item())

import matplotlib.pyplot as plt
plt.plot(mae_vals)

#%% Part II: Weight Analysis of MLP 0.

res = pd.read_csv(os.path.join(data_dir, r"outputs\models\mlp0\noPool\relu\selected_results.csv"))
dimensions = [7]
for hdim in literal_eval(res["hiddensize"].item()):
    dimensions.append(hdim)
dimensions.append(1)

model = models.MLP(dimensions, nn.ReLU)
model.load_state_dict(torch.load(os.path.join(data_dir, r"outputs\models\mlp0\noPool\relu\model0.pth")))

hidden0 = model[0].weight.detach().numpy()
hidden1 = model[2].weight.detach().numpy()

#%%
weight_sums1 = []
weight_sums2 = []

for i in range(7):
    weight_sums1.append(np.sum(hidden0[:,i]))
    weight_sums2.append(np.sum(hidden0[:,i] + np.transpose(hidden1)[:,0]) )

weight_sums1
weight_sums2
