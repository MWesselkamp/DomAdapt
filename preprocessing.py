# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:55:08 2020

@author: marie
"""

#%% Input normalization
import os
import os.path
import pandas as pd

import utils


#%% load data

def load_data(dataset, data_dir, simulations = None):
    
    path_in = os.path.join(data_dir, f"{dataset}_in")
    if (simulations=="preles"):
        path_out = os.path.join(data_dir, f"{simulations}_out")
    else:
        path_out = os.path.join(data_dir, f"{dataset}_out")
        
    X = pd.read_csv(path_in, sep=";")
    Y = pd.read_csv(path_out, sep=";")
    
    # Remove nows with na values
    rows_with_nan = pd.isnull(X).any(1).to_numpy().nonzero()[0]
    X = X.drop(rows_with_nan)
    Y = Y.drop(rows_with_nan)
    
    return X, Y

def get_splits(sites, datadir, dataset = "profound", 
    colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos"],
    to_numpy = True):
    
    datadir = os.path.join(datadir, f"{dataset}")
    X, Y = load_data(dataset = dataset, data_dir = datadir)
    
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
    
    if to_numpy:
        X, Y = X.to_numpy(), Y.to_numpy()
    
    return X[row_ind], Y[row_ind]



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

