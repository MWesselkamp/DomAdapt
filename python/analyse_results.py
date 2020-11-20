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
import setup.preprocessing as preprocessing
import setup.dev_mlp as dev_mlp
import collect_results

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

X, Y = preprocessing.get_splits(sites = ["hyytiala"],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None, to_numpy=False)

#%% Part I: Reduced Amount of data
mae_vals = []
mae_trains = []

for i in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95]:
    
    preds = np.load(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\y_preds.npy"))
    tests = np.load(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\y_tests.npy"))
    res = pd.read_csv(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\selected_results.csv"))
    
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

weights1 = []
weights2 = []
for i in range(5):
    model = models.MLP(dimensions, nn.ReLU)
    model.load_state_dict(torch.load(os.path.join(data_dir, f"outputs\models\mlp0\\noPool\\relu\model{i}.pth")))

    hidden0 = model[0].weight.detach().numpy()
    hidden1 = model[2].weight.detach().numpy()

    weight_sums1 = []
    weight_sums2 = []

    for i in range(7):
        weight_sums1.append(np.sum(hidden0[:,i]))
        weight_sums2.append(np.sum(hidden0[:,i] + np.transpose(hidden1)[:,0]) )

    weights1.append(weight_sums1)
    weights2.append(weight_sums2)

#%%
weights2m = np.mean(np.array(weights2),0)
weights2sd = np.std(np.array(weights2),0)

weights2error_upper = weights2m + 2*weights2sd
weights2error_lower = weights2m - 2*weights2sd
variables = ("PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos")

plt.errorbar(variables, y = weights2m, yerr = weights2sd, fmt='', marker = 'o')


#%% Part III: train models on Boreal forest data.
# A) new training
X, Y = preprocessing.get_borealsites(year = "train")
X_test, Y_test = preprocessing.get_borealsites(year = "test")

hparams = {"epochs":500,
           "batchsize":int(res["batchsize"]),
           "history":int(res["history"]),
           "learningrate":res["learningrate"].item()}
model_design = {"dimensions":dimensions,
                "activation":nn.ReLU,
                "featuresize":None}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, 
                                                                       {"X_test":X_test, "Y_test":Y_test}, 
                                                                       0.0, data_dir, False)

visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], legend=True)

#%% B) predict with fitted model (architecture 2)
prediction_errors = collect_results.borealsites_predictions()
