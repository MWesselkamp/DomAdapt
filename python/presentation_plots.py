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
from numpy.polynomial.polynomial import polyfit
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn
import finetuning

import visualizations
import setup.preprocessing as preprocessing
#%%
fulltab = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv", index_col=False)
fulltab.drop(fulltab.columns[0], axis=1, inplace=True)

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
def plot1b(colors = ["blue","blue",  "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                #[fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o', 'o',  's', 'o', "*", '*']
    s = [ 60, 60, 60,  60, 200, 200]
    labs = ["Reference NN",None,"Pretrained NN",None, "PRELES", "RandomForest"]
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
def plot1c(colors = ["blue","blue", "orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o', 'o', 'x', 's', 'o', "*", '*']
    s = [ 60, 60, 60, 60, 60, 200, 200]
    labs = ["Reference NN", None, "Finetuned", "Pretrained",None, "PRELES", "RandomForest"]
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
def plot1f(colors = ["blue","blue", "orange","orange","orange","orange", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
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
    yi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
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
                
    m = ['o', 'o', 'x','x','x', 'x', 's', 'o', "*", '*']
    s = [ 60, 60, 60, 60,60,60,60, 60, 200, 200]
    labs = ["Reference NN",None, "Finetuned NN",None, None,None,"Pretrained NN",None, "PRELES", "RandomForest"]
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
plot1f(colors = ["lightgrey", "lightgrey", "lightgrey", "orange","orange","lightgrey","lightgrey","lightgrey", "lightgrey", "lightgrey", "lightgrey"], 
       log=True)


#%%
def plot1g(colors = ["blue","blue", "orange", "orange", "green", "green", "blue", "red", "yellow"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")  & ((fulltab.typ == 8) | (fulltab.typ == 10)) ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") & ((fulltab.typ == 8) | (fulltab.typ == 10))]["mae_val"]], 
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A")& (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")  & ((fulltab.typ == 8) | (fulltab.typ == 10)) ]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") & ((fulltab.typ == 8) | (fulltab.typ == 10))]["rmse_val"]],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
                
    m = ['o', 'o', 'x','x', 's','s', 'o', "*", '*']
    s = [60, 60, 60, 60,60, 60,60, 200, 200]
    labs = ["Selected", None, "Finetuned",None, "Pretrained",None, None,  "PRELES", "RandomForest"]
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
plot1g(colors = ["lightgrey", "lightgrey", "lightgrey", "orange", "lightgrey", "green", "lightgrey", "lightgrey", "lightgrey"],
       log=True)
        
#%% PLOT 1.3
def plot13(fulltab):
    
    fulltab = fulltab[:155].astype({'typ':'int64'})
    fulltab = fulltab.loc[(fulltab.typ != 6) & (fulltab.typ != 7) & (fulltab.typ != 9)]# & (fulltab.finetuned_type != "D-MLP2")]
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
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
    plt.xticks(locs, ["OLS", "MLP", "Full \nbackprop", "Feature \nextraction"])
    plt.ylim(0.4,1.1)
    
    bm = fulltab.loc[(fulltab.typ == 0) & (fulltab.architecture == 2)].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 4,colors="orange", linestyles="dashed", label="best MLP", linewidth=1.2)
    L=plt.legend(loc="upper left")
    #L.get_texts()[0].set_text("A1 Fixed")
    #L.get_texts()[0].set_text("A1 normal (exp)")
    L.get_texts()[0].set_text("Explicit parameters (A1)")
    #L.get_texts()[2].set_text("A0 fixed")
    L.get_texts()[1].set_text("Implicit parameters (A0)")
    L.get_texts()[2].set_text("Best MLP (A0)")
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
    
    plt.hlines(bestmlp0, 0, epochs,colors="orange", linestyles="dashed", label="Best selected network", linewidth=1.2)
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
    plt.text(x=posmlp2, y=1.25, s="MLP", fontstyle="italic")

    prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
    posprel = np.max(np.where(rl["mae_val"][:, :epochs] > prel))
    plt.arrow(x=posprel, y=1.6, dx=0, dy=-(1.6-prel), linewidth=0.8, color="gray")
    plt.text(x=posprel, y=1.65, s="PRELES", fontstyle="italic")

    plt.legend()
    #plt.text(x= 50, y= 3.5, s=f"MAE = {np.round(bm.iloc[bm['mae_val'].idxmin()]['mae_val'],4)}")
#%%
plot2(10,50, 1500) 

#%%
def plot3a():
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

    #Y_preles = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008def"), sep=";")
    #Y_preles_calib = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008calib"), sep=";")

    fig, ax = plt.subplots(figsize=(7,7))
    fig.suptitle("Hyytiälä (2008)")
    ax.plot(Y, color="green", label="Ground Truth", marker = "o", linewidth=0.8, alpha=0.9, markerfacecolor='green', markersize=4)
    #ax.plot(Y_preles, color="blue", label="PRELES \nPredictions", marker = "", alpha=0.5)
    #ax.plot(Y_preles_calib, color="green", label="PRELES \nPredictions", marker = "", alpha=0.5)
    ax.set(xlabel="Day of Year", ylabel="GPP [g C m$^{-2}$ day$^{-1}$]")
    plt.legend()
#%%
plot3a()
#%%
def plot3b():
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

    #Y_preles = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008def"), sep=";")
    Y_preles_calib = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008calib"), sep=";")
    
    fig, ax = plt.subplots(figsize=(7,7))
    fig.suptitle("Hyytiälä (2008)")
    ax.plot(Y, color="lightgrey", label="Ground Truth", marker = "o", linewidth=0.8, alpha=0.9, markerfacecolor='lightgrey', markersize=4)
    #ax.plot(Y_preles, color="blue", label="PRELES \nPredictions", marker = "", alpha=0.5)
    ax.plot(Y_preles_calib, color="green", label="PRELES \nPredictions", marker = "", alpha=0.5)
    ax.set(xlabel="Day of Year", ylabel="GPP [g C m$^{-2}$ day$^{-1}$]")
    plt.legend(loc="upper right")
    
    mae = metrics.mean_absolute_error(Y, Y_preles_calib)
    plt.text(10,10, f"MAE = {np.round(mae, 4)}")
#%%
plot3b()

#%%
def plot3c(sparse = False):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    if sparse:
        Y_preds = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\sparse\models\mlp0\sparse1\y_preds.npy", allow_pickle=True)
    else:
        Y_preds = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp0\noPool\sigmoid\y_preds.npy", allow_pickle=True)
    visualizations.plot_prediction(Y, Y_preds, "Hyytiälä (2008)")
    plt.legend(loc="upper right")
    
    mae = metrics.mean_absolute_error(Y, np.mean(Y_preds, axis = 0))
    plt.text(10,10, f"MAE = {np.round(mae, 4)}")
#%%
plot3c(sparse=True)
plot3c(sparse=False)

#%%
def plot3d(sparse = False):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    Y_preds = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\sparse\models\mlp8\sparse1\setting1\y_preds.npy", allow_pickle=True)
    
    visualizations.plot_prediction(Y, Y_preds, "Hyytiälä (2008)")
    plt.legend(loc="upper right")
#%%
plot3d()

#%%
def plot3e(sparse = False):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    Y_preds = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\lstm0\y_preds.npy", allow_pickle=True)
    
    visualizations.plot_prediction(Y, Y_preds, "Hyytiälä (2008)")
    plt.legend(loc="upper right")
    
    mae = metrics.mean_absolute_error(Y[:Y_preds.shape[1]], np.mean(Y_preds, 0))
    plt.text(10,10, f"MAE = {np.round(mae, 4)}")
#%%
plot3e()

#%%
def plot3f(years=[2001,2002,2003, 2004, 2005, 2006, 2007]):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    
    predictions_test, errors = finetuning.featureExtractorC("mlp", 10, None, 50,
                      years = years)
    Y_preds = np.array(predictions_test)
    
    visualizations.plot_prediction(Y, Y_preds, "Hyytiälä (2008)")
    plt.legend(loc="upper right")
    
    mae = metrics.mean_absolute_error(Y, np.mean(Y_preds, 0))
    plt.text(10,10, f"MAE = {np.round(mae, 4)}")
#%%
plot3f()
plot3f(years = [2003, 2004, 2005, 2006])
plot3f(years = [2004, 2005, 2006])
plot3f(years = [2005, 2006])
plot3f(years = [2004])
#%%
def plot4(w, model, years=[2001,2002,2003, 2004, 2005, 2006, 2007]):
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    Y_preles = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008calib"), sep=";")
    Y_nn = np.transpose(np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp0\noPool\sigmoid\y_preds.npy", allow_pickle=True).squeeze(2))
    predictions_test, errors = finetuning.featureExtractorC("mlp", 10, None, 50,
                      years = years)
    Y_nn_f = np.transpose(np.array(predictions_test).squeeze(2))
    
    mt = moving_average(Y.squeeze(1), w)
    mp = moving_average(Y_preles.squeeze(1), w)
    mn = moving_average(np.mean(Y_nn, axis=1), w)
    mnf = moving_average(np.mean(Y_nn_f, axis=1), w)
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    plt.plot(mt, label="Groundtruth", color="lightgrey")
    if model=="preles":
        plt.plot(mp, label="PRELES \npredictions", color="green")
        maep = metrics.mean_absolute_error(mt, mp)
        plt.text(10,9, f"MAE = {np.round(maep, 4)}")
    elif model=="mlp0":
        plt.plot(mn, label="MLP \npredictions", color="green")
        maen = metrics.mean_absolute_error(mt, mn)
        plt.text(10,9, f"MAE = {np.round(maen, 4)}")
    elif model=="mlp10":
        plt.plot(mnf, label="Finetuned MLP \npredictions", color="green")
        maen = metrics.mean_absolute_error(mt, mnf)
        plt.text(10,9, f"MAE = {np.round(maen, 4)}")
    plt.xlabel("Day of Year")
    plt.ylabel("Average GPP over 7 days [g C m$^{-2}$ day$^{-1}$]")
    plt.legend()
    
#%%
plot4(w = 7, model= "preles")
plot4(w = 7, model= "mlp10")
plot4(w = 7, model= "mlp0")
#%%
# Polynomial Regression
def rsquared(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

def plot5(model, w = None, years=[2001,2002,2003, 2004, 2005, 2006, 2007]):
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    X, Y = preprocessing.get_splits(sites = ['hyytiala'],   
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

    Y_preles = pd.read_csv(os.path.join(data_dir ,r"data\profound\outputhyytiala2008calib"), sep=";")
    Y_nn = np.transpose(np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp0\noPool\sigmoid\y_preds.npy", allow_pickle=True).squeeze(2))
    predictions_test, errors = finetuning.featureExtractorC("mlp", 10, None, 50,
                      years = years)
    Y_nn_f = np.transpose(np.array(predictions_test).squeeze(2))
    
    if not w is None:
        Y = moving_average(Y.squeeze(1), w)
        Y_preles = moving_average(Y_preles.squeeze(1), w)
        Y_nn = moving_average(np.mean(Y_nn, axis=1), w)
        Y_nn_f = moving_average(np.mean(Y_nn_f, axis=1), w)
    else:
        Y = Y.squeeze(1)
        Y_preles = Y_preles.squeeze(1)
        Y_nn = np.mean(Y_nn, axis=1)
        Y_nn_f = np.mean(Y_nn_f, axis=1)

    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    if model == "preles":
        plt.scatter(Y_preles, Y, color="darkblue")
        # Fit with polyfit
        b, m = polyfit(Y_preles, Y,  1)
        r2_p = rsquared(Y_preles, Y,  1)["determination"]
        plt.plot(Y_preles, b + m * Y_preles, '-', color="darkred", label = "y = a + b $\hat{y}$ ")
        maep = metrics.mean_absolute_error(Y, Y_preles)
        plt.text(0,10, f"MAE = {np.round(maep, 4)}")
        plt.text(0,9, f"R$^2$ = {np.round(r2_p, 4)}")
    elif model == "mlp0":
        plt.scatter(Y_nn, Y, color="darkblue")
        # Fit with polyfit
        b, m = polyfit(Y_nn, Y, 1)
        r2_nn = rsquared(Y_nn, Y,  1)["determination"]
        plt.plot(Y_nn, b + m *Y_nn, '-', color="darkred", label = "y = a + b $\hat{y}$ ")
        maen = metrics.mean_absolute_error(Y, Y_nn)
        plt.text(0,10, f"MAE = {np.round(maen, 4)}")
        plt.text(0,9, f"R$^2$ = {np.round(r2_nn, 4)}")
    elif model == "mlp10":
        plt.scatter(Y_nn_f, Y, color="darkblue")
        b, m = polyfit(Y_nn_f, Y, 1)
        r2_nnf = rsquared(Y_nn_f, Y,  1)["determination"]
        plt.plot(Y_nn_f, b + m * Y_nn_f, '-', color="darkred", label = "y = a + b $\hat{y}$ ")
        maenf = metrics.mean_absolute_error(Y, Y_nn_f)
        plt.text(0,10, f"MAE = {np.round(maenf, 4)}")
        plt.text(0,9, f"R$^2$ = {np.round(r2_nnf, 4)}")
    
    plt.plot(np.arange(11), 0 + 1 *np.arange(11), '--', color="gray", label = "y = $\hat{y}$")
    plt.xlim((-1,11))
    plt.ylim((-1,11))
    plt.ylabel("True GPP Test [g C m$^{-2}$ day$^{-1}$]")
    plt.xlabel("Estimated GPP Test [g C m$^{-2}$ day$^{-1}$]")
    
    plt.legend(loc="lower right")

#%%
plot5("preles", w=7)
plot5("mlp10", w=7)
plot5("mlp0", w=7)

#%%
sl0s = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\sparse\models\mlp0\sparse1\selected_results.csv")
sl5 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp5\nodropout\sims_frac100\selected_results.csv")
sl6 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp6\nodropout\sims_frac100\selected_results.csv")
sl7 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\sims_frac100\selected_results.csv")
sl8 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp8\nodropout\sims_frac100\selected_results.csv")
sl9 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp9\nodropout\sims_frac100\selected_results.csv")
sl10 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp10\nodropout\sims_frac100\selected_results.csv")
