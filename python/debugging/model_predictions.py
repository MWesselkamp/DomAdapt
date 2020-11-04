# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:19:04 2020

@author: marie
"""

#%%
import matplotlib.pyplot as plt
import pandas as pd
import setup.models as models
import torch.nn as nn
import torch
import setup.utils as utils
import setup.preprocessing as preprocessing
import os.path
from sklearn import metrics
#%%

datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%%
gridsearch_results = pd.read_csv(os.path.join(datadir, f"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()
model = models.MLP([7, 64, 64, 16, 1], nn.ReLU)
model.load_state_dict(torch.load(os.path.join(datadir, f"python\outputs\models\mlp2\model0.pth")))

X_scaled = utils.minmax_scaler(X)
X_scaled = torch.tensor(X_scaled).type(dtype=torch.float)

y_preds = model(X_scaled).detach().numpy()

plt.plot(Y)
plt.plot(y_preds)

metrics.mean_absolute_error(Y,y_preds)
#%%
model = models.MLP([7, 64, 64, 16, 1], nn.ReLU)

model.load_state_dict(torch.load(os.path.join(datadir, f"python\outputs\models\mlp2\model0.pth")))
y_preds = model(X_scaled).detach().numpy()

plt.plot(Y)
plt.plot(y_preds)

metrics.mean_absolute_error(Y,y_preds)
