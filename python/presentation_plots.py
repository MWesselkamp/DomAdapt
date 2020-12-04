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
plt.rcParams.update({'font.size': 14})
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
    labs = ["Selected", None,None, None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("log(MAE)")
            plt.ylabel("log(RMSE)")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
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
    labs = ["Selected", None,None,"Pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
    
    if log:
        xlocs, xlabels = plt.xticks() 
        xlocs[1] = 0.5           # Get locations and labels
        plt.xticks(xlocs[1:4], [0.5,1, 10])
        ylocs, ylabels = plt.yticks() 
        ylocs[1] = 0.5           # Get locations and labels
        plt.yticks(ylocs[1:4], [0.5,1, 10])
#%%
plot1b(log=True)    

#%%
def plot1c(colors = ["blue","blue","blue", "orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'x', 's', 'o', "*", '*']
    s = [60, 60, 60, 60, 60, 60, 200, 200]
    labs = ["Selected", None,None, "Finetuned", "Pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
    
    if log:
        xlocs, xlabels = plt.xticks() 
        xlocs[1] = 0.5           # Get locations and labels
        plt.xticks(xlocs[1:4], [0.5,1, 10])
        ylocs, ylabels = plt.yticks() 
        ylocs[1] = 0.5           # Get locations and labels
        plt.yticks(ylocs[1:4], [0.5,1, 10])

#%%
plot1c(log=True)    

#%%
def plot1d(colors = ["blue","blue","blue", "orange","orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'x','x', 's', 'o', "*", '*']
    s = [60, 60, 60, 60, 60,60, 60, 200, 200]
    labs = ["Selected", None,None, "Finetuned",None, "Pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
    
    if log:
        xlocs, xlabels = plt.xticks() 
        xlocs[1] = 0.5           # Get locations and labels
        plt.xticks(xlocs[1:4], [0.5,1, 10])
        ylocs, ylabels = plt.yticks() 
        ylocs[1] = 0.5           # Get locations and labels
        plt.yticks(ylocs[1:4], [0.5,1, 10])

#%%
plot1d(log=True)

#%%
def plot1e(colors = ["blue","blue","blue", "orange","orange","orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'x','x','x', 's', 'o', "*", '*']
    s = [60, 60, 60, 60, 60,60,60, 60, 200, 200]
    labs = ["Selected", None,None, "Finetuned",None, None,"Pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
    
    if log:
        xlocs, xlabels = plt.xticks() 
        xlocs[1] = 0.5           # Get locations and labels
        plt.xticks(xlocs[1:4], [0.5,1, 10])
        ylocs, ylabels = plt.yticks() 
        ylocs[1] = 0.5           # Get locations and labels
        plt.yticks(ylocs[1:4], [0.5,1, 10])

#%%
plot1e(log=True)

#%%
def plot1f(colors = ["blue","blue","blue", "orange","orange","orange","orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "D-MLP2") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "D-MLP2")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o','o', 'o', 'x','x','x', 'x', 's', 'o', "*", '*']
    s = [60, 60, 60, 60, 60,60,60,60, 60, 200, 200]
    labs = ["Selected", None,None, "Finetuned",None, None,None,"Pretrained",None, "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
        else:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
            plt.ylabel("Root Mean Squared Error [g C m$^{-2}$ day$^{-1}$]")
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)

        plt.legend(loc="lower right")
    
    if log:
        xlocs, xlabels = plt.xticks() 
        xlocs[1] = 0.5           # Get locations and labels
        plt.xticks(xlocs[1:4], [0.5,1, 10])
        ylocs, ylabels = plt.yticks() 
        ylocs[1] = 0.5           # Get locations and labels
        plt.yticks(ylocs[1:4], [0.5,1, 10])

#%%
plot1f(log=True)

#%%
plot1f(colors = ["lightgrey", "lightgrey", "lightgrey", "orange","orange","orange","orange","lightgrey", "lightgrey", "lightgrey", "lightgrey"], 
       log=True)

#%% PLOT 1.3
def plot13(fulltab):
    
    fulltab = fulltab[:155].astype({'typ':'int64'})
    fulltab = fulltab.loc[(fulltab.typ != 6)]# & (fulltab.finetuned_type != "D-MLP2")]
    plt.figure(num=None, figsize=(10,7), facecolor='w', edgecolor='k')
    seaborn.boxplot(x = "finetuned_type",
            y = "mae_val",
            hue = "typ",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")],
            width=0.7,
            linewidth = 0.8)
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    locs, labels = plt.xticks()
    plt.xticks(locs, ["OLS", "Add. \nlayer", "Full \nback-prop.", "Feature \nextraction"])
    plt.ylim(0.4,1.1)
    
    bm = fulltab.loc[(fulltab.typ == 0) & (fulltab.architecture == 2)].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 4,colors="orange", linestyles="dashed", label="best MLP", linewidth=1.2)
    L=plt.legend(loc="upper left")
    #L.get_texts()[0].set_text("A1 Fixed")
    L.get_texts()[0].set_text("A1 normal (exp)")
    L.get_texts()[1].set_text("A1 uniform (exp)")
    L.get_texts()[2].set_text("A0 fixed")
    L.get_texts()[3].set_text("A0 uniform (imp)")
    L.get_texts()[4].set_text("A0 best MLP")
#%%
plot13(fulltab)

#%% PLOT 1.4
def plot14(fulltab):
    
    fulltab = fulltab[:155].astype({'typ':'int64'})
    fulltab = fulltab.loc[(fulltab.typ != 6)]
    plt.figure(num=None, figsize=(10, 7), facecolor='w', edgecolor='k')
    seaborn.boxplot(y = "mae_val", 
                x = "simsfrac",
                hue = "typ",
                palette = "pastel",
                data = fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")] ,
                width=0.7,
                linewidth = 0.8,
                showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"6"})
    plt.xlabel("Percentage of simulations used for training")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    plt.ylim(0.4,1.2)
    
    bm = fulltab.loc[(fulltab.typ == 0) & (fulltab.architecture == 2)].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 4,colors="orange", linestyles="dashed", label="best MLP", linewidth=1.2)
    L=plt.legend(loc="upper left")
    #L.get_texts()[0].set_text("A1 Fixed")
    L.get_texts()[0].set_text("A1 normal (exp)")
    L.get_texts()[1].set_text("A1 uniform (exp)")
    L.get_texts()[2].set_text("A0 fixed")
    L.get_texts()[3].set_text("A0 uniform (imp)")
    L.get_texts()[4].set_text("A0 best MLP")
#%%
plot14(fulltab)
#%% Plot 2:
# Make sure to have the same reference full backprob model!
def plot2(typ, frac, epochs = 2000):

    bm = fulltab.loc[(fulltab.task == "finetuning") & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bm.iloc[bm['mae_val'].idxmin()].to_dict()
    # now load the model losses from file.
    rl = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting0\\running_losses.npy", allow_pickle=True).item()
    visualizations.plot_running_losses(rl["mae_train"][:, :epochs], rl["mae_val"][:, :epochs], False, False)

    bm = fulltab.loc[(fulltab.typ == 0) & (fulltab.architecture == 2)].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #rf = fulltab.loc[(fulltab.model == "rf")]["mae_val"].item()
    
    plt.hlines(bestmlp0, 0, epochs,colors="orange", linestyles="dashed", label="Best MLP", linewidth=1.2)
    #plt.hlines(rf, 0, 2000,colors="yellow", linestyles="dashed", label="Random Forest", linewidth=1.2)
    
    bm = fulltab.loc[(fulltab.typ == typ) & (fulltab.simsfrac == frac) & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bestols = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    posols = np.max(np.where(rl["mae_val"][:, :epochs] > bestols))
    plt.arrow(x=posols, y=1.3, dx=0, dy=-(1.3-bestols), linewidth=0.8, color="gray")
    plt.text(x=posols, y=1.35, s="OLS", fontstyle="italic")
    
    try:
        bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "B-fW2")].reset_index()
        bestfw2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
        posfw2 = np.max(np.where(rl["mae_val"][:, :epochs] > bestfw2))
        plt.arrow(x=posfw2, y=2, dx=0, dy=-(2-bestfw2), linewidth=0.8, color="gray")
        plt.text(x=posfw2, y=2.05, s="Feature \nextraction", fontstyle="italic")
    except:
        print("no fw2.")
    
    bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "D-MLP2")].reset_index()
    bestmlp2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    posmlp2 = np.max(np.where(rl["mae_val"][:, :epochs] > bestmlp2))
    plt.arrow(x=posmlp2, y=1.2, dx=0, dy=-(1.2-bestmlp2), linewidth=0.8, color="gray")
    plt.text(x=posmlp2, y=1.25, s="Add. \nlayer", fontstyle="italic")

    prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
    posprel = np.max(np.where(rl["mae_val"][:, :epochs] > prel))
    plt.arrow(x=posprel, y=1.6, dx=0, dy=-(1.6-prel), linewidth=0.8, color="gray")
    plt.text(x=posprel, y=1.65, s="PRELES", fontstyle="italic")

    plt.legend()
    #plt.text(x= 50, y= 3.5, s=f"MAE = {np.round(bm.iloc[bm['mae_val'].idxmin()]['mae_val'],4)}")
#%%
plot2(6,50, 1500) 
