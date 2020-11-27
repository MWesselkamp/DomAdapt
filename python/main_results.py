# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:42:45 2020

@author: marie

The feature extractors take the arguments:
    
    model (str): pretrained network class to load (use mlp)
    typ (int): what type of pretrained network to load. 7 has seen normal parameters, 8 uniform parameters.
    epochs (int): if feature extraction involves retraining, enter number of epochs. Else None.
    simsfrac (int): on how much of the simuluated data has the network been pretrained? (E.g. 30 for 30%. Available: 30,50,80,100)
    
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
subtab1, running_losses, predictions = collect_results.feature_extraction_results(types = [6,7,8], simsfrac = [30, 50, 70, 100])

subtab2 = collect_results.selected_networks_results(types = [7,8], simsfrac = [30,50,70,100])

subtab1 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\featureextraction.csv", index_col=False)
subtab1.drop(subtab1.columns[0], axis=1, inplace=True)
#subtab2 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\selectednetworks.csv", index_col=False)
#subtab2.drop(subtab2.columns[0], axis=1, inplace=True)
fulltab = pd.concat([subtab1, subtab2])
fulltab.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\results_full.xlsx")
fulltab.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv")
#
fulltab = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv", index_col=False)

#%% PLOT 1
def plot1(colors = ["blue","blue","blue", "orange", "green", "red", "yellow", "blue"], log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.id =="MLP0nP2D0S")]["rmse_val"].item()]]
    m = ['o','o', 'o', 'x', 's', "*", '*', 'o']
    s = [60, 60, 60, 60, 60, 200, 200, 60]
    labs = ["selected", None,None, "finetuned", "pretrained", "PRELES", "RandomForest", None]
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
plot1(log=True) 
plot1(log=False)
#plot1(colors = ["lightgrey", "lightgrey", "lightgrey", "orange", "lightgrey", "lightgrey", "lightgrey"])    
#plot1(colors = ["blue", "blue", "blue", "lightgrey", "lightgrey", "lightgrey", "lightgrey"])  
#plot1(colors = ["blue", "red", "red", "lightgrey", "lightgrey", "lightgrey", "lightgrey"])  

#%% PLOT 1A
def plot1A(colors = ["blue","blue","blue", "orange", "green", "red", "yellow", "blue"] ):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="borealsitesprediction")& (fulltab.model =="preles")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="borealsitesprediction")& (fulltab.model =="mlp")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") ]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="borealsitesprediction") & (fulltab.model =="preles")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()],
                 [fulltab.loc[(fulltab.task =="borealsitesprediction") & (fulltab.model =="mlp")]["rmse_val"].item()]]
    m = ['o','o', 'o', 'x', 's', "*", '*', 'o']
    s = [60, 60, 60, 60, 60, 200, 200, 60]
    labs = ["selected", None,None, "finetuned", "pretrained", "PRELES", "RandomForest", None]
    for i in range(len(xi)):
        plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
        plt.ylim(0, 2.0)    
        plt.xlim(0, 2.0)
        plt.legend(loc="upper right")
        plt.xlabel("Mean Absolute Error")
        plt.ylabel("Root Mean Squared Error")
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
#%%
plot1(["lightgrey","lightgrey","lightgrey", "lightgrey", "lightgrey", "red", "lightgrey", "blue"])
plot1A(["lightgrey","lightgrey","lightgrey", "lightgrey", "lightgrey", "red", "lightgrey", "blue"])
#%% Plot 1.1
def plot11():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    xi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS")]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2")]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "D-MLP2")]["mae_val"]]]
    yi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "C-OLS")]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fb")]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "B-fW2")]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "D-MLP2")]["rmse_val"]]]

    s = [60, 60,60, 60]
    labs = ["OLS", "Full Backpropagation", "Retrain Last Layer", "MLP"]
    for i in range(len(xi)):
        plt.scatter(xi[i], yi[i], alpha = 0.8, marker="x", s = s[i], label=labs[i])
    plt.ylim(0, 2.0)    
    plt.xlim(0, 2.0)
    plt.legend()
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Root Mean Squared Error")
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
#%%
plot11()
#%% Plot 1.2
def plot12():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    xi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 7)]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 8)]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 6)]["mae_val"]]]
    yi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 7)]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 8)]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 6)]["rmse_val"]]]

    s = [60, 60,60, 60]
    labs = ["Normal Parameter Samples", "Uniform Parameter Samples", "Default Parameter Values"]
    for i in range(len(xi)):
        plt.scatter(xi[i], yi[i], alpha = 0.8, marker="x", s = s[i], label=labs[i])
    plt.ylim(0, 2.0)    
    plt.xlim(0, 2.0)
    plt.legend()
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Root Mean Squared Error")
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
#%%
plot12()

#%% PLOT 1.3
def plot13():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    seaborn.boxplot(x = "finetuned_type",
            y = "mae_val",
            hue = "typ",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")],
            width=0.5,
            linewidth = 0.8)
    plt.xlabel("Type of finetuning")
    plt.ylabel("Mean Absolute Error")
    plt.ylim(0.4,1.0)
    
    bm = fulltab.loc[(fulltab.typ == 0)].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 4,colors="orange", linestyles="dashed", label="Best MLP", linewidth=1.2)
#%%
plot13()
    
#%% PLOT 1.4
def plot14():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    seaborn.boxplot(y = "mae_val", 
                x = "simsfrac",
                hue = "typ",
                palette = "pastel",
                data = fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")] ,
                width=0.5,
                linewidth = 0.8,
                showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"6"})
    plt.xlabel("Percentage of simulations used for training")
    plt.ylabel("Mean Absolute Error")
    plt.ylim(0.4,1.2)
#%%
plot14()

#%% Plot 2:
# Make sure to have the same reference full backprob model!

bm = fulltab.loc[(fulltab.task == "finetuning") & (fulltab.finetuned_type == "C-OLS")].reset_index()
bm.iloc[bm['mae_val'].idxmin()].to_dict()
# now load the model losses from file.
rl = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp6\nodropout\sims_frac30\tuned\setting0\running_losses.npy", allow_pickle=True).item()

visualizations.plot_running_losses(rl["mae_train"][:, :1000], rl["mae_val"][:, :1000], False, False)

bm = fulltab.loc[(fulltab.typ == 0)].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
#rf = fulltab.loc[(fulltab.model == "rf")]["mae_val"].item()

plt.hlines(bestmlp0, 0, 1000,colors="orange", linestyles="dashed", label="Best MLP", linewidth=1.2)
#plt.hlines(rf, 0, 2000,colors="yellow", linestyles="dashed", label="Random Forest", linewidth=1.2)

bm = fulltab.loc[(fulltab.typ == 6) & (fulltab.simsfrac == 30) & (fulltab.finetuned_type == "C-OLS")].reset_index()
bestols = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posols = np.max(np.where(rl["mae_val"] > bestols))
plt.arrow(x=posols, y=1.7, dx=0, dy=-(1.7-bestols), linewidth=0.8, color="gray")
plt.text(x=posols, y=1.75, s="OLS")

bm = fulltab.loc[(fulltab.typ == 6)& (fulltab.simsfrac == 30)  & (fulltab.finetuned_type == "B-fW2")].reset_index()
bestfw2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posfw2 = np.max(np.where(rl["mae_val"] > bestfw2))
plt.arrow(x=posfw2, y=2, dx=0, dy=-(2-bestfw2), linewidth=0.8, color="gray")
plt.text(x=posfw2, y=2.1, s="fW2")
           
bm = fulltab.loc[(fulltab.typ == 6)& (fulltab.simsfrac == 30)  & (fulltab.finetuned_type == "D-MLP2")].reset_index()
bestmlp2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posmlp2 = np.max(np.where(rl["mae_val"] > bestmlp2))
plt.arrow(x=posmlp2, y=1.2, dx=0, dy=-(1.2-bestmlp2), linewidth=0.8, color="gray")
plt.text(x=posmlp2, y=1.25, s="MLP")

prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
posprel = np.max(np.where(rl["mae_train"] > prel))
plt.arrow(x=posprel, y=1.5, dx=0, dy=-(1.5-prel), linewidth=0.8, color="gray")
plt.text(x=posprel-100, y=1.55, s="PRELES")

plt.legend()
#%% Plot 4: plt.errorbar! linestyle='None', marker='^'
df = collect_results.analyse_basemodel_results([10,15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

#%%
params = pd.read_csv(os.path.join(data_dir, r"data\parameter_default_values.csv"), sep =";", names=["name", "value"])
#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
def plot4(parameter):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    si_n = pd.read_csv(os.path.join(data_dir, r"data\simulations\normal_params\sims_in.csv"), sep =";")
    si_u = pd.read_csv(os.path.join(data_dir, r"data\simulations\uniform_params\sims_in.csv"), sep =";")
    
    fig, ax = plt.subplots(figsize=(7,7))
    plt.hist(si_u[parameter], bins=50, histtype="stepfilled", density=True, color="lightblue",alpha=0.7, linewidth=1.2, label="uniform")
    plt.hist(si_n[parameter], bins=50, histtype="stepfilled", density=True, color="lightgreen", alpha=0.7, linewidth=1.2, label="truncated normal")
    plt.xlabel(parameter, fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylim(0,1)
    plt.legend(prop={"size":14})

plot4("beta")
