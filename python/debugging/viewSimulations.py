# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:11 2020

@author: marie
"""

import setup.preprocessing as preprocessing
import setup.dev_mlp as dev_mlp
import pandas as pd
import os.path
import numpy as np
#%%
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(datadir, r"data\simulations\uniform_params"))

#%%
results = pd.read_csv(os.path.join(datadir, f"python/outputs/grid_search/mlp/grid_search_results_mlp1.csv"))
best_model = results.iloc[results['mae_val'].idxmin()].to_dict()

#%%
dev_mlp.selected(X_sims, Y_sims, "mlp", "7", best_model, 20, 5, 7)

#%%
data_norm = (X_sims - np.mean(X_sims, axis=0))/np.std(X_sims, axis=0)
ss = pd.DataFrame(data_norm).describe()
ss = pd.DataFrame(X_sims).describe()
