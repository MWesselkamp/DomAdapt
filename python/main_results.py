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

import pandas as pd
import numpy as np
import collect_results
import matplotlib.pyplot as plt
import seaborn
import matplotlib

import visualizations
#%%
#subtab1, running_losses, predictions = collect_results.feature_extraction_results(types = [7,8], simsfrac = [30, 50, 70, 100])

subtab2 = collect_results.selected_networks_results(types = [7,8], simsfrac = [30,50,70,100])

subtab1 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\featureextraction.csv", index_col=False)
subtab1.drop(subtab1.columns[0], axis=1, inplace=True)
#subtab2 = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\selectednetworks.csv", index_col=False)
#subtab2.drop(subtab2.columns[0], axis=1, inplace=True)
fulltab = pd.concat([subtab1, subtab2])
fulltab.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\results_full.xlsx")
fulltab.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv")

#%% PLOT 1
def plot1(colors = ["blue","blue","blue", "orange", "green", "red", "yellow"] ):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    xi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["mae_val"]], 
                [fulltab.loc[fulltab.task =="pretraining"]["mae_val"]],
                [fulltab.loc[fulltab.task =="processmodel"]["mae_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()]]
    yi = [[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["rmse_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")]["rmse_val"]],
                [fulltab.loc[fulltab.task =="pretraining"]["rmse_val"]],
                [fulltab.loc[fulltab.task =="processmodel"]["rmse_val"]],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["rmse_val"].item()]]
    m = ['o','o', 'o', 'x', 's', "*", '*']
    s = [60, 60, 60, 60, 60, 200, 200]
    labs = ["selected", None,None, "finetuned", "pretrained", "PRELES", "RandomForest"]
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
plot1() 
plot1(colors = ["lightgrey", "lightgrey", "lightgrey", "orange", "lightgrey", "lightgrey", "lightgrey"])    
plot1(colors = ["blue", "blue", "blue", "lightgrey", "lightgrey", "lightgrey", "lightgrey"])  
plot1(colors = ["blue", "red", "red", "lightgrey", "lightgrey", "lightgrey", "lightgrey"])  
  
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
    labs = ["OLS", "Full Backpropagation", "Freeze Last Layer", "MLP"]
    for i in range(len(xi)):
        plt.scatter(xi[i], yi[i], alpha = 0.8, marker="x", s = s[i], label=labs[i])
    plt.ylim(0, 2.0)    
    plt.xlim(0, 2.0)
    plt.legend()
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Root Mean Squared Error")
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)

#%% Plot 1.2
def plot12():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    xi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 7)]["mae_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 8)]["mae_val"]]]
    yi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 7)]["rmse_val"]],
                   [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.typ == 8)]["rmse_val"]]]

    s = [60, 60,60, 60]
    labs = ["Normal Parameter Samples", "Uniform Parameter Samples"]
    for i in range(len(xi)):
        plt.scatter(xi[i], yi[i], alpha = 0.8, marker="x", s = s[i], label=labs[i])
    plt.ylim(0, 2.0)    
    plt.xlim(0, 2.0)
    plt.legend()
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Root Mean Squared Error")
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)

#%% PLOT 1.3
def plot13():
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')
    cols = seaborn.color_palette(palette="pastel")
    seaborn.boxplot(x = "finetuned_type",
            y = "mae_val",
            hue = "typ",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")],
            width=0.5,
            linewidth = 0.8)
    plt.xlabel("Type of finetuning")
    plt.ylabel("Mean Absolute Error")
    plt.ylim(0,1.2)
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
    plt.ylim(0,1.2)
    
#%% Plot 2:
# Make sure to have the same reference full backprob model!
bm = fulltab.loc[(fulltab.task == "finetuning") & (fulltab.finetuned_type == "B-fb")].reset_index()
bm.iloc[bm['mae_val'].idxmin()].to_dict()
# now load the model losses from file.
rl = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\sims_frac30\tuned\setting0\running_losses.npy", allow_pickle=True).item()

visualizations.plot_running_losses(rl["mae_train"][:, :2000], rl["mae_val"][:, :2000], False)

bm = fulltab.loc[(fulltab.typ == 0)].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
#rf = fulltab.loc[(fulltab.model == "rf")]["mae_val"].item()

plt.hlines(bestmlp0, 0, 2000,colors="orange", linestyles="dashed", label="Best MLP", linewidth=1.2)
#plt.hlines(rf, 0, 2000,colors="yellow", linestyles="dashed", label="Random Forest", linewidth=1.2)

bm = fulltab.loc[(fulltab.typ == 7) & (fulltab.simsfrac == 30) & (fulltab.finetuned_type == "C-OLS")].reset_index()
bestols = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posols = np.max(np.where(rl["mae_val"] > bestols))
plt.arrow(x=posols, y=3, dx=0, dy=-(3-bestols), linewidth=0.8)
plt.text(x=posols, y=3.1, s="OLS")

bm = fulltab.loc[(fulltab.typ == 7)& (fulltab.simsfrac == 30)  & (fulltab.finetuned_type == "B-fW2")].reset_index()
bestfw2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posfw2 = np.max(np.where(rl["mae_val"] > bestfw2))
plt.arrow(x=posfw2, y=3, dx=0, dy=-(3-bestfw2), linewidth=0.8)
plt.text(x=posfw2, y=3.1, s="fW2")
           
bm = fulltab.loc[(fulltab.typ == 7)& (fulltab.simsfrac == 30)  & (fulltab.finetuned_type == "D-MLP2")].reset_index()
bestmlp2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
posmlp2 = np.max(np.where(rl["mae_val"] > bestmlp2))
plt.arrow(x=posmlp2, y=3, dx=0, dy=-(3-bestmlp2), linewidth=0.8)
plt.text(x=posmlp2, y=3.1, s="MLP")

prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
posprel = np.max(np.where(rl["mae_train"] > prel))
plt.arrow(x=posprel, y=3, dx=0, dy=-(3-prel), linewidth=0.8)
plt.text(x=posprel, y=3.1, s="PRELES")

plt.legend()
#%% Plot 4: plt.errorbar! linestyle='None', marker='^'
