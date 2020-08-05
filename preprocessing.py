# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:55:08 2020

@author: marie
"""

#%% Input normalization
import os
import os.path
import pandas as pd
import numpy as np

import random
from sklearn.utils import shuffle
from math import floor

import utils


#%% load data

def load_data(dataset, data_dir = r'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\data', simulations = None):
    
    path_in = os.path.join(data_dir, f"{dataset}\{dataset}_in")
    if (simulations=="preles"):
        path_out = os.path.join(data_dir, f"{dataset}\{simulations}_out")
    else:
        path_out = os.path.join(data_dir, f"{dataset}\{dataset}_out")
        
    X = pd.read_csv(path_in, sep=";")
    Y = pd.read_csv(path_out, sep=";")
    
    # Remove nows with na values
    rows_with_nan = pd.isnull(X).any(1).nonzero()[0]
    X = X.drop(rows_with_nan)
    Y = Y.drop(rows_with_nan)
    
    return X, Y

def get_splits(sites, dataset = "profound", 
    colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos"]):
    
    X, Y = load_data(dataset = dataset)
    
    X["DOY_sin"], X["DOY_cos"] = utils.encode_doy(X["DOY"])

    if all([site in X["site"].values for site in sites]):
        row_ind = X['site'].isin(sites)
        print(f"Returns {sites} from \n", X["site"].unique())
    else: 
        print("Not all sites in dataset!")
    
    try:
        X = X[colnames]
    except:
        print("Columns are missing!")
        
    try:
        Y= Y.drop(columns=["ET"])
    except:
        None
        
    X, Y = X.to_numpy(), Y.to_numpy()
    
    return X[row_ind], Y[row_ind]

#%% 
def create_batches(X, Y, batchsize, history):
    
    subset = [j for j in random.sample(range(X.shape[0]), batchsize) if j > history]
    subset_h = [item for sublist in [list(range(j-history,j)) for j in subset] for item in sublist]
    x = np.concatenate((X[subset], X[subset_h]), axis=0)
    y = np.concatenate((Y[subset], Y[subset_h]), axis=0)
    
    return x, y

#%%
def get_simulations(data_dir = 'data\preles\exp', ignore_env = True):

    filesnum = int(len([name for name in os.listdir(data_dir)])/2)
    filenames = [f'sim{i}' for i in range(1,filesnum+1)]
    
    X = [None]*filesnum
    Y = [None]*filesnum
    
    for i in range(filesnum):
        filename = filenames[i]
        path_in = os.path.join(data_dir, f"{filename}_in")
        path_out = os.path.join(data_dir, f"{filename}_out")
        X[i] = pd.read_csv(path_in, sep=";")
        if(ignore_env):
            X[i] = X[i].drop(columns=['date']).to_numpy()
        Y[i] = pd.read_csv(path_out, sep=";").to_numpy()
        
    return X, Y#, filenames

