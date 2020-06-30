# -*- coding: utf-8 -*-
"""
This script reads in and processes data that have been generated
and preprocessed in Rstudio.
a) observed input (@profound database) and corresponding simulated output.
"""

import os
import os.path
os.getcwd()
os.chdir('OneDrive/Dokumente/Sc_Master/Masterthesis/Project/DomAdapt')

import pandas as pd
import numpy as np

def get_simulations(data_dir = 'data\preles\exp'):
        
    data = {}

    filesnum = int(len([name for name in os.listdir(data_dir)])/2)
    filenames = [f'sim{i}' for i in range(1,filesnum+1)]

    for filename in filenames:
        data[filename] = {}
        path_in = os.path.join(data_dir, f"{filename}_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
        data[f"{filename}"]['y'] = pd.read_csv(path_out, sep=";").to_numpy()
        data[f"{filename}"]['X'] = pd.read_csv(path_in, sep=";").drop(columns=['date']).to_numpy()
        
    return data

sims = get_simulations()

def merge_XY(data):
    """
    This function concatenates the target and the feature values to one large np.array.
    These are again saved in a dictionary ("sim1",...).
    """
    data = {sim[0]: np.concatenate([v for k,v in sim[1].items()], 1) for sim in data.items()}
    return data

sim = sims['sim1']

# Data scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (-1,1))
sims_norm = scaler.fit_transform(sim.reshape(-1,12))
