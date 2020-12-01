# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:58:32 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import os.path
import pandas as pd
import numpy as np
import collect_results
import matplotlib.pyplot as plt
import seaborn
import matplotlib

import visualizations

#%%
fulltab = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv", index_col=False)
fulltab.drop(fulltab.columns[0], axis=1, inplace=True)

#%%
def plot1a(colors = ["blue","blue","blue", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                #[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                #[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'o', "*", '*']
    s = [60, 60, 60,60, 200, 200]
    labs = ["selected", None,None, None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("log(MAE)")
            plt.ylabel("log(RMSE)")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error")
            plt.ylabel("Root Mean Squared Error")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
#%%
plot1a(log=False) 

#%%
def plot1b(colors = ["blue","blue","blue",  "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o',  's', 'o', "*", '*']
    s = [60, 60, 60, 60,  60, 200, 200]
    labs = ["selected", None,None,"pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("log(MAE)")
            plt.ylabel("log(RMSE)")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error")
            plt.ylabel("Root Mean Squared Error")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")

#%%
plot1b(log=True)    

#%%
def plot1c(colors = ["blue","blue","blue", "orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'x', 's', 'o', "*", '*']
    s = [60, 60, 60, 60, 60, 60, 200, 200]
    labs = ["selected", None,None, "finetuned", "pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("log(MAE)")
            plt.ylabel("log(RMSE)")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error")
            plt.ylabel("Root Mean Squared Error")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")

#%%
plot1c(log=True)    
plot1c(colors = ["lightgrey", "lightgrey", "lightgrey", "orange", "lightgrey", "lightgrey", "lightgrey", "lightgrey"], log=True)
