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
import visualizations
import finetuning
from sklearn import metrics
import setup.preprocessing as preprocessing

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({'font.size': 18})
#%%
subtab1, running_losses, predictions = collect_results.feature_extraction_results(types = [5,6, 7,8,9,10,11,12], simsfrac = [30, 50, 70, 100])
subtab2 = collect_results.selected_networks_results(simsfrac = [30,50,70,100])
fulltab = pd.concat([subtab1, subtab2])

subtab3 = collect_results.sparse_networks_results([0, 1,2,3, 4, 5])
fulltab = pd.concat([fulltab, subtab3])

fulltab.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\results_full.xlsx")
fulltab.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv")
#
fulltab = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv", index_col=False)
fulltab.drop(fulltab.columns[0], axis=1, inplace=True)
#fulltab = fulltab[:155].astype({'typ':'int64'})

#%% PLOT 1
def plot1(colors, log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    
    xi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_val"]],
                #[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_val"].item()],
                #[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")  & (fulltab.finetuned_type != "D-MLP2") & (fulltab.simsfrac == 100) ]["mae_val"]], 
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A")& (fulltab.simsfrac == 100) ]["mae_val"]],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_val"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_val"].item()],
                [fulltab.loc[(fulltab.id =="MLP4nP1D0R")]["mae_val"].item()]]
    yi = [#[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "mlp")]["mae_train"]],
                #[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "cnn")]["mae_train"].item()],
                #[fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "lstm")]["mae_train"].item()],
                [fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")& (fulltab.finetuned_type != "D-MLP2")& (fulltab.simsfrac == 100)]["mae_train"]],
                [fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type == "A") & (fulltab.simsfrac == 100)]["mae_train"]],
                [fulltab.loc[(fulltab.task =="processmodel") & (fulltab.typ == 0)]["mae_train"].item()],
                [fulltab.loc[(fulltab.task =="selected") & (fulltab.model == "rf")]["mae_train"].item()],
                [fulltab.loc[(fulltab.id =="MLP4nP1D0R")]["mae_train"].item()]]
    #m = ['o','o', 'o', 'x', 's', "*", '*', 'o']
    m = ['s', 's', "*", '*', 'o']
    #s = [60, 60, 60, 60, 60, 200, 200, 60]
    s = [ 100, 100, 280, 280, 100]
    # labs = ["selected", None,None, "finetuned", "pretrained", "PRELES", "RandomForest", None]
    labs = ["Fine-tuned", "Pretrained", "PRELES", "RandomForest", "Best MLP"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            plt.xlabel("log(MAE test)", size = 20)
            plt.ylabel("log(MAE training)", size=20)
        else:
            plt.rcParams.update({'font.size': 20})
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("MAE Test [g C m$^{-2}$ day$^{-1}$]", size=20)
            plt.ylabel("MAE Training [g C m$^{-2}$ day$^{-1}$]", size=20)
            plt.ylim(0,8.5)
            plt.xlim(0,8.5)
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)

        plt.legend(loc="lower right", prop={'size': 20})
#%%
plot1(colors = ["orange", "green", "red", "purple", "blue"], log=True) 
plot1(colors = ["orange", "green", "red", "purple", "blue"], log=False)

#%% PLOT 1.3
def plot13a(fulltab):
    
    fulltab = fulltab[:256].astype({'typ':'int64'})
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    #plt.suptitle("Direct transfer")
    cols = ["green", "red", "blue", "orange", "purple"]

    bplot = seaborn.boxplot(x = "typ",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type == "A") & (fulltab.typ != 9)  & (fulltab.typ != 4)],
            width=0.6,
            linewidth = 1) #labels = ["5","6", "7","8","9","10"]
    for i in range(5):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #plt.ylim(0.3,3)
    #bplot.set_xticklabels(["A2 \nunif","A3 \nfix","A3 \nnorm","A3 \nunif","A1 \nunif"])
    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1,5,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.legend(loc="upper right")

def plot13b(fulltab):
    
    fulltab = fulltab[:256].astype({'typ':'int64'})
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    #plt.suptitle("Ordinary least squares")
    
    cols = ["green", "red", "blue", "orange", "purple"]
    bplot = seaborn.boxplot(x = "typ",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type == "C-OLS")& (fulltab.simsfrac != 30) & (fulltab.typ != 9)  & (fulltab.typ != 4) ],
            width=0.6,
            linewidth = 1) #labels = ["5","6", "7","8","9","10"])
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #bplot.set_xticklabels(["A2 \nunif","A3 \nfix","A3 \nnorm","A3 \nunif","A1 \nunif"])
    for i in range(5):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 5,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.legend(loc="upper right")

def plot13c(fulltab):
    
    fulltab = fulltab[:256].astype({'typ':'int64'})
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    #plt.suptitle("Full backpropagation")
    
    cols = ["green", "red", "blue", "orange", "purple"]
    bplot = seaborn.boxplot(x = "typ",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type == "B-fb")& (fulltab.typ != 9) & (fulltab.typ != 4)],
            width=0.6,
            linewidth = 1) #labels = ["5","6", "7","8","9","10"])
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #bplot.set_xticklabels(["A2 \nunif","A3 \nfix","A3 \nnorm","A3 \nunif","A1 \nunif"])
    for i in range(5):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 5,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.legend(loc="upper right")

def plot13d(fulltab):
    
    fulltab = fulltab[:256].astype({'typ':'int64'})
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    #plt.suptitle("Feature Extraction")
    
    cols = ["green", "red", "blue", "orange", "purple"]
    bplot = seaborn.boxplot(x = "typ",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type == "B-fW2")& (fulltab.typ != 9)& (fulltab.typ != 4)],
            width=0.6,
            linewidth = 1) #labels = ["5","6", "7","8","9","10"])
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #bplot.set_xticklabels(["A2 \nunif","A3 \nfix","A3 \nnorm","A3 \nunif","A1 \nunif"])
    for i in range(4):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 4,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.legend(loc="upper right")

def plot13e(fulltab):
    
    fulltab = fulltab[:256].astype({'typ':'int64'})
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    #plt.suptitle("Modified Classifier")
    
    cols = ["green", "red", "blue", "orange", "purple"]
    bplot = seaborn.boxplot(x = "typ",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type == "D-MLP2")& (fulltab.typ != 9)& (fulltab.typ != 4)],
            width=0.6,
            linewidth = 1) #labels = ["5","6", "7","8","9","10"])
    plt.xlabel("")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #bplot.set_xticklabels(["A2 \nunif","A3 \nfix","A3 \nnorm","A3 \nunif","A1 \nunif"])
    for i in range(5):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    plt.hlines(bestmlp0, -1, 5,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.legend(loc="upper right")
    
#%%
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.constrained_layout.use'] = True
plot13a(fulltab)
plot13b(fulltab)
plot13c(fulltab)
plot13d(fulltab)
plot13e(fulltab)

#%% PLOT 1.3: ONLY for full simulations.

models=[8, 10,12]
maes = []
types = []
for mod in models:
    predictions_test, errors = finetuning.featureExtractorC("mlp", mod, None, 100)
    maes.append(errors[3])
    types.append([mod]*5)
    
maes = [item for sublist in maes for item in sublist]
types = [item for sublist in types for item in sublist]
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])
#%%
plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
cols = ["blue", "purple", "green"]
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.constrained_layout.use'] = True
bplot = seaborn.boxplot(y="mae_val",
                x = "typ",
                data = df,
                width=0.6)
for i in range(len(models)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
plt.ylim(bottom=0.2)
bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
plt.hlines(bestmlp0, -1, 3,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]", size=20)
plt.xlabel("", size = 20)
bplot.set_xticklabels(["Adaptive\nPooling","Wide","Deep"])
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc="upper right", prop={"size":18})
#%%
models=[8, 10,12]
maes = []
types = []
for mod in models:
    predictions_test, errors, y_test = finetuning.featureExtractorA("mlp", mod, None, 100)
    maes.append(errors[3])
    types.append([mod]*5)
    
maes = [item for sublist in maes for item in sublist]
types = [item for sublist in types for item in sublist]
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])

#%%
plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
cols = ["blue", "purple", "green"]
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.constrained_layout.use'] = True
bplot = seaborn.boxplot(y="mae_val",
                x = "typ",
                data = df,
                width=0.6)
for i in range(len(models)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
plt.ylim(bottom=0.2)
bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
plt.hlines(bestmlp0, -1, 3,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]", size=20)
plt.xlabel("", size = 20)
bplot.set_xticklabels(["Adaptive\nPooling","Wide","Deep"])
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc="upper right", prop={"size":18})

#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X_test, y_tests = preprocessing.get_splits(sites = ['hyytiala'],
                                          years = [2008],
                                          datadir = os.path.join(data_dir, "data"), 
                                          dataset = "profound",
                                          simulations = None)
models = [8,10,12]
maes = []
types = []
for mod in models:
    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\nodropout\sims_frac100\\tuned\setting0\\y_preds.npy"), allow_pickle=True)
    for i in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[i,:])))
        types.append(mod)
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])   
#%%
plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
cols = ["blue", "purple", "green"]
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.constrained_layout.use'] = True
bplot = seaborn.boxplot(y="mae_val",
                x = "typ",
                data = df,
                width=0.6)
for i in range(len(models)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
plt.ylim(bottom=0.4)
bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
plt.hlines(bestmlp0, -1, 3,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]", size=20)
plt.xlabel("", size = 20)
bplot.set_xticklabels(["Adaptive\nPooling","Wide","Deep"])
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc="upper right", prop={"size":18})

#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X_test, y_tests = preprocessing.get_splits(sites = ['hyytiala'],
                                          years = [2008],
                                          datadir = os.path.join(data_dir, "data"), 
                                          dataset = "profound",
                                          simulations = None)
models = [5,7,8,12]
maes = []
types = []
for mod in models:
    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\nodropout\sims_frac100\\tuned\setting1\\y_preds.npy"), allow_pickle=True)
    for i in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[i,:])))
        types.append(mod)
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])   
#%%
seaborn.boxplot(y="mae_val",
                x = "typ",
                data = df)
plt.ylim(bottom=0.4)
bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
plt.hlines(bestmlp0, -1, 5,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
#%% PLOT 1.4
def plot14(fulltab):
    
    fulltab = fulltab.astype({'typ':'int64'})
    plt.figure(num=None, figsize=(10, 7), facecolor='w', edgecolor='k')
    seaborn.boxplot(y = "mae_val", 
                x = "simsfrac",
                hue = "typ",
                palette = "pastel",
                data = fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")] ,
                width=0.7,
                linewidth = 0.8,
                showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"6"})
    plt.xlabel("Percentage of simulations used for training")
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    #plt.ylim(0.4,1.2)
    
#%%
plot14(fulltab)

#%% Plot 2:
# Make sure to have the same reference full backprob model!
def plot2(typ, frac, epochs = 5000):

    bm = fulltab.loc[(fulltab.task == "finetuning") & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bm.iloc[bm['mae_val'].idxmin()].to_dict()
    # now load the model losses from file.
    rl = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting0\\running_losses.npy", allow_pickle=True).item()
    visualizations.plot_running_losses(rl["mae_train"][:, :epochs], rl["mae_val"][:, :epochs], False, False)

    bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #rf = fulltab.loc[(fulltab.model == "rf")]["mae_val"].item()
    
    plt.hlines(bestmlp0, 0, epochs,colors="darkblue", linestyles="dashed", label="Best MLP", linewidth=2)
    #plt.hlines(rf, 0, 2000,colors="yellow", linestyles="dashed", label="Random Forest", linewidth=1.2)
    
    bm = fulltab.loc[(fulltab.typ == typ) & (fulltab.simsfrac == frac) & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bestols = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    posols = np.max(np.where(rl["mae_val"][:, :epochs] > bestols))
    plt.arrow(x=posols, y=0.7, dx=0, dy=-(0.7-bestols), linewidth=1.5, color="gray")
    # mlp10: plt.text(x=posols, y=1.05, s="OLS", fontstyle="italic", fontsize=16)
    # mlp7: plt.text(x=posols, y=1.05, s="OLS", fontstyle="italic", fontsize=16)
    plt.text(x=posols, y=0.71, s="OLS", fontstyle="italic", fontsize=16)
    
    try:
        bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "B-fW2")].reset_index()
        bestfw2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
        posfw2 = np.max(np.where(rl["mae_val"][:, :epochs] > bestfw2))
        plt.arrow(x=posfw2, y=0.8, dx=0, dy=-(0.8-bestfw2), linewidth=1.0, color="gray")
        # mlp7: plt.text(x=posfw2, y=1.25, s="Partial\nback-prop", fontstyle="italic", fontsize=16)
        plt.text(x=posfw2, y=0.81, s="Partial\nback-prop", fontstyle="italic", fontsize=16)
    except:
        print("no fw2.")
    
    #bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "D-MLP2")].reset_index()
    #bestmlp2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #posmlp2 = np.max(np.where(rl["mae_val"][:, :epochs] > bestmlp2))
    #plt.arrow(x=posmlp2, y=1.2, dx=0, dy=-(1.2-bestmlp2), linewidth=0.8, color="gray")
    #plt.text(x=posmlp2, y=1.25, s="MLP")

    prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
    posprel = np.max(np.where(rl["mae_val"][:, :epochs] > prel))
    plt.arrow(x=posprel, y=0.9, dx=0, dy=-(0.9-prel), linewidth=1.5, color="gray")
    # mlp7: plt.text(x=posprel-30, y=1.55, s="PRELES", fontstyle="italic", fontsize=16)
    plt.text(x=posprel-30, y=0.91, s="PRELES", fontstyle="italic", fontsize=16)
    
    plt.legend()
#%%

plot2(12,100, 5000)
plot2(10,100, 5000)  
plot2(8,100, 5000)
plot2(7,100, 5000)     
plot2(6,100, 1000)   
#%% Plot 4: plt.errorbar! linestyle='None', marker='^'
df = collect_results.sparse_networks_results([1])
df = df.loc[(df.sparse == 1)]

visualizations.losses("mlp", 0, "sparse1", sparse=True)
visualizations.losses("mlp", 6, "sparse1", sparse=True)
visualizations.losses("mlp", 6, "sparse1\setting0", sparse=True)
visualizations.losses("mlp", 6, "sparse1\setting1", sparse=True)
visualizations.losses("mlp", 8, "sparse1", sparse=True)
visualizations.losses("mlp", 8, "sparse1\setting0", sparse=True)
visualizations.losses("mlp", 8, "sparse1\setting1", sparse=True)
#%%
data_dir = 'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt'
params = pd.read_csv(os.path.join(data_dir, r"data\parameter_default_values.csv"), sep =";", names=["name", "value"])
#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
def plot4(parameter, lim = True):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    si_n = pd.read_csv(os.path.join(data_dir, r"data\simulations\normal_params\sims_in.csv"), sep =";")
    si_u = pd.read_csv(os.path.join(data_dir, r"data\simulations\uniform_params\sims_in.csv"), sep =";")
    
    fig, ax = plt.subplots(figsize=(7,7))
    plt.hist(si_u[parameter], bins=50, histtype="stepfilled", density=False, color="lightblue",alpha=0.7, linewidth=1.2, label="uniform")
    plt.hist(si_n[parameter], bins=50, histtype="stepfilled", density=False, color="lightgreen", alpha=0.7, linewidth=1.2, label="truncated normal")
    plt.xlabel(parameter, fontsize=18)
    plt.ylabel("Counts", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    if lim:
        plt.ylim(0,1)
    plt.legend(prop={"size":14})
#%%
plot4("beta", False)
plot4("X0", False)
plot4("alpha", False)
plot4("gamma", False)

#%% Analyse OLS feature matrix
import finetuning

predictions_test, errors, out_test = finetuning.featureExtractorC("mlp", 8, None, 30)
corr1 = np.corrcoef(out_test, rowvar=False)
predictions_test, errors, out_test = finetuning.featureExtractorC("mlp", 8, None, 50)
corr2 = np.corrcoef(out_test, rowvar=False)
predictions_test, errors, out_test = finetuning.featureExtractorC("mlp", 8, None, 70)
corr3 = np.corrcoef(out_test, rowvar=False)
predictions_test, errors, out_test = finetuning.featureExtractorC("mlp", 8, None, 100)
corr4 = np.corrcoef(out_test, rowvar=False)

np.nansum(np.abs(corr1))
np.nansum(np.abs(corr2))
np.nansum(np.abs(corr3))
np.nansum(np.abs(corr4))


#%% Plot X: Errorbar plot for size of source domain

def plot5(plot="architectures"):
    from sklearn import metrics
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    df = pd.DataFrame(None, columns = ["fracs", "models", "run", "mae"])

    plt.rcParams.update({'font.size': 20})
    fig,ax = plt.subplots(figsize=(7.1,7.1))
    seaborn.set(style= "ticks")

    if plot=="architectures":
        mods = [10,12,8]
        labels = ["Architecture 1", "Architecture 2","Architecture 3"]
        cols = ["purple", "green",  "blue"]
    elif plot=="parameters":
        mods = [6,8,7]
        labels = ["fixed", "uniform","normal"]
        cols = ["red", "blue",  "orange"]
    for i in range(len(mods)):
        ndatapoints = [15000,25000,35000,50000]
        fracs = [30,50,70,100]
        sds = [] 
        means = []
        for frac in fracs:
            y_preds = np.load(os.path.join(data_dir, f"python\outputs\models\mlp{mods[i]}\\nodropout\\sims_frac{frac}\y_preds.npy")).squeeze(2)
            y_tests = np.load(os.path.join(data_dir, f"python\outputs\models\mlp{mods[i]}\\nodropout\\sims_frac{frac}\y_tests.npy")).squeeze(2)
            errors = []
            for model in range(5):
                errors.append(metrics.mean_absolute_error(y_tests[model,:], y_preds[model,:]))
                df = df.append({"mae":metrics.mean_absolute_error(y_tests[model,:], y_preds[model,:]),
                            "run":model,
                            "models":mods[i],
                            "fracs":frac},ignore_index=True)
            means.append(np.mean(errors))
            sds.append(np.std(errors)*2)
        #all_errors.append(errors)  
        ax.errorbar(ndatapoints, y = means, yerr = sds,  linewidth = 3, elinewidth=3, capsize=5, capthick=3,
                #markerfacecolor="black", markeredgecolor="black",
                color=cols[i],
                alpha=0.7, #fmt="o",
                marker = 'o', label=f"{labels[i]}")
        #ax.set_ylim((0,1.35))
        ax.legend(bbox_to_anchor=(0.5, 1.17), loc="upper center", ncol=2, prop={'size': 20})
        ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20)
        ax.set_xlabel("Size of source domain [datapoints]", size=20)
        for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
                    ax.set_xticklabels(labels = ["","15000", "", "25000", "", "35000", "", "", "50000"], size=18)
                    #all_errors = np.transpose(np.array(all_errors))
                    #seaborn.boxplot(x="fracs",
                    #                y="mae",
                    #                hue="models",
                    #                data=df)

#%%
plot5(plot="architectures")
plot5(plot="parameters")
#%%
#%%
def running_losses(train_losses, val_losses, 
                   colors1 = ["lightgreen",  "thistle"],
                   colors2 = ["green",  "blueviolet"]):

    #if model=="mlp":
    #    colors = ["blue","lightblue"]
    #elif model=="cnn":
    #    colors = ["darkgreen", "palegreen"]
    #elif model=="lstm":
    #    colors = ["blueviolet", "thistle"]
    #else:
    labels = ["fixed", "uniform"]

    fig, ax = plt.subplots(figsize=(7,7))
    
    for i in range(len(val_losses)):
        #ci_train = np.quantile(train_losses[i], (0.05,0.95), axis=0)
        ci_val = np.quantile(val_losses[i], (0.05,0.95), axis=0)
        #train_loss = np.mean(train_losses[i], axis=0)
        val_loss = np.mean(val_losses[i], axis=0)
       # ax.fill_between(np.arange(len(train_losses[i])), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        ax.fill_between(np.arange(10000), ci_val[0],ci_val[1], color=colors1[i], alpha=0.3)
        
        #ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=1.2)
        ax.plot(val_loss, color=colors2[i], label = labels[i], linewidth=1.2)

    ax.set(xlabel="Epochs", ylabel="Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]")
    plt.ylim(bottom = 0.0)
    plt.rcParams.update({'font.size': 20})
    plt.legend(loc="upper right")
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models"

#%%
rl11 = np.load(os.path.join(data_dir,r"mlp11\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()
rl12 = np.load(os.path.join(data_dir,r"mlp12\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

rl6 = np.load(os.path.join(data_dir,r"mlp6\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()
rl8 = np.load(os.path.join(data_dir,r"mlp8\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

running_losses([rl11["mae_train"][:,:10000], rl12["mae_train"][:,:10000]], [rl11["mae_val"][:,:10000], rl12["mae_val"][:,:10000]])  
running_losses([rl6["mae_train"][:,:10000], rl8["mae_train"][:,:10000]], [rl6["mae_val"][:,:10000], rl8["mae_val"][:,:10000]])  
