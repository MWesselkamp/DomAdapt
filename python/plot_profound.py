# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:01:09 2020

@author: marie
"""

import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing
import numpy as np

import matplotlib.pyplot as plt

#%%
X, Y = preprocessing.get_splits(sites = ["hyytiala"], 
                                datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\data", 
                                dataset = "profound",
                                to_numpy=False)
#%%
fig, ax = plt.subplots(5, figsize=(8,9), sharex='col')
fig.suptitle("Preles input data")
for i in range(5):
    ax[i].plot(X.to_numpy()[:365,i])
    ax[i].set_ylabel(X.columns[i])
fig.text(0.5,0.04, "Day of Year")

    