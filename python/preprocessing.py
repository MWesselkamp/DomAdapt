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

def load_data(dataset, data_dir, simulations):
    
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

def get_splits(sites, years, datadir, dataset = "profound", simulations = None, drop_cols = False,
    colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos"],
    to_numpy = True):
    
    datadir = os.path.join(datadir, f"{dataset}")
    
    X, Y = load_data(dataset = dataset, data_dir = datadir, simulations = simulations)
    
    X["date"] = X["date"].str[:4].astype(int) # get years as integers
    X["DOY_sin"], X["DOY_cos"] = utils.encode_doy(X["DOY"]) # encode day of year as sinus and cosinus

    if all([site in X["site"].values for site in sites]):
        row_ind = X['site'].isin(sites)
        print(f"Returns {sites} from \n", X["site"].unique())
        X, Y = X[row_ind], Y[row_ind]
    else: 
        print("Not all sites in dataset!")
        
    try:
        row_ind = X["date"].isin(years)
        print(f"Returns valid years from {years} in \n", X["date"].unique())
        X, Y = X[row_ind], Y[row_ind]
    except: 
        print(" years specification invalid. Returns all years.")
    
    try:
        X = X[colnames]
    except:
        print("Columns are missing!")
    
    if simulations != None:    
        
        if drop_cols:
            Y= Y.drop(columns=["ET", "SW"])
        else:
            X["ET"] = Y["ET"]
            X["SW"] = Y["SW"]
            Y= Y.drop(columns=["ET", "SW"])
    else:
        try:
            Y= Y.drop(columns=["ET"])
        except:
            None
            
    if to_numpy:
        X, Y = X.to_numpy(), Y.to_numpy()
    
    return X, Y



#%%
def get_simulations(data_dir, ignore_env = True):

    
    path_in = os.path.join(data_dir, f"sims_in.csv")
    path_out = os.path.join(data_dir, f"sims_out.csv")
    
    X = pd.read_csv(path_in, sep=";")

    Y = pd.read_csv(path_out, sep=";")
        
    return X, Y#, filenames

