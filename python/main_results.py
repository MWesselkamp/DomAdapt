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
import torch
import setup.models as models
from scipy.stats import pearsonr

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({'font.size': 18})

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%%
subtab2 = collect_results.selected_networks_results(simsfrac = [30,50,70,100])

#fulltab = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv", index_col=False)
#fulltab.drop(fulltab.columns[0], axis=1, inplace=True)
#fulltab = fulltab[:155].astype({'typ':'int64'})

#%% REFERENCE MODELS: BOXPLOTS

def base_predictions(mods, architectures=None, types = None, dummies = False,  rf = False):

    maes = []
    typ = []
    archs = []
    
    predictions = []
    
    j = 0
    
    for mod in mods:
        
        hparams, model_design, X, Y, X_test, Y_test = finetuning.settings(mod, None, data_dir, dummies)
    
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(Y_test).type(dtype=torch.float)
        
        if ((mod > 7) | (mod == 0) | (mod == 4)):
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
        
        
        mod_predictions = []
        mod_maes = []
        mod_corr = []
        for i in range(5):
        
            model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\relu\model{i}.pth")))
    
            pred_test = model(X_test)#
            with torch.no_grad():
                maes.append(metrics.mean_absolute_error(y_test, pred_test))
                mod_maes.append(metrics.mean_absolute_error(y_test, pred_test))
                mod_corr.append(pearsonr(Y_test.squeeze(1), pred_test.squeeze(1))[0])
                mod_predictions.append(pred_test.numpy())
            if not types is None:
                typ.append(types)
            else:
                typ.append(f"mlp{mod}")
            if not architectures is None:
                archs.append(architectures[j])
        
        error = np.round(np.mean(mod_maes), 4)
        corr = np.round(np.mean(mod_corr),4)
        print(error)
        visualizations.plot_prediction(Y_test, np.array(mod_predictions).squeeze(2), mae=error)
        visualizations.scatter_prediction(Y_test, np.array(mod_predictions).squeeze(2), corr=corr)
        predictions.append(mod_predictions)
        
        j += 1
            
    if rf:
        y_preds_rf0 = np.load(os.path.join(data_dir, f"python\outputs\models\\rf0\y_preds.npy"))
        mod_maes = []
        for i in range(5):
            maes.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_rf0)[:,i]))
            mod_maes.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_rf0)[:,i]))
            typ.append("rf0")
            archs.append("rf")
        error = np.round(np.mean(mod_maes), 4)
        visualizations.plot_prediction(Y_test, y_preds_rf0, mae=error)


    df = pd.DataFrame(list(zip(typ, maes)),
                 columns=["typ", "mae_val"]) 
    
    if not architectures is None:
        df["archs"] = archs
        
    return(df, predictions)

#%%
df, predictions = base_predictions([0, 4, 5], architectures=None, types = None, dummies = False, rf = True)

res = Y_test - predictions[0][0]
res_s = (res - np.min(res)) / (np.max(res)-np.min(res))
plt.plot(res_s)

utils.error_in_percent(df.groupby("typ")["mae_val"].mean())

#%%
def finetuned_predictions(mods, setting, architectures=None, types = None, dummies = False):

    maes = []
    typ = []
    archs = []
    
    predictions = []
    
    j = 0
    
    for mod in mods:
        
        hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", mod, None, data_dir, dummies)
    
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(Y_test).type(dtype=torch.float)
        
        if ((mod > 7) | (mod == 0) | (mod == 4)):
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
        
        
        mod_predictions = []
        mod_maes = []
        mod_corr = []
        
        for i in range(5):
        
            try:
                model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\nodropout\sims_frac100\\tuned\\setting{setting}\\model{i}.pth")))
            except:
                model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\nodropout\sims_frac100\\tuned\\setting{setting}\\freeze2\\model{i}.pth")))
            
            pred_test = model(X_test)#
            with torch.no_grad():
                maes.append(metrics.mean_absolute_error(y_test, pred_test))
                mod_maes.append(metrics.mean_absolute_error(y_test, pred_test))
                mod_corr.append(pearsonr(Y_test.squeeze(1), pred_test.squeeze(1))[0])
                mod_predictions.append(pred_test.numpy())
            if not types is None:
                typ.append(types)
            else:
                typ.append(f"mlp{mod}")
            if not architectures is None:
                archs.append(architectures[j])
        
        error = np.round(np.mean(mod_maes), 4)
        corr = np.round(np.mean(mod_corr),4)
        print(error)
        visualizations.plot_prediction(Y_test, np.array(mod_predictions).squeeze(2), mae=error)
        visualizations.scatter_prediction(Y_test, np.array(mod_predictions).squeeze(2), corr=corr)
        predictions.append(mod_predictions)
        
        j += 1


    df = pd.DataFrame(list(zip(typ, maes)),
                 columns=["typ", "mae_val"]) 
    
    if not architectures is None:
        df["archs"] = archs
        
    return(df, predictions)
    
#%%
df, predictions = finetuned_predictions([10, 12, 7], setting = 1)
df, predictions = finetuned_predictions([13, 14, 5], setting = 1, dummies=True)

#%% Predictions OLS

predictions_test, errors = finetuning.featureExtractorC(10, None, 100)
visualizations.plot_prediction(Y_test, np.array(predictions_test).squeeze(2), mae=np.mean(errors))

#%% Predictions: Weekly 
Y_test_rm = np.convolve(Y_test.squeeze(1), 14)
Y_preds_rm = []
for i in range(len(predictions_test)):
    Y_preds_rm.append(np.convolve(predictions_test[i].squeeze(1), 14))

visualizations.plot_prediction(Y_test_rm, np.array(Y_preds_rm), mae=np.mean(errors))
#%%
fig, ax = plt.subplots(figsize=(7,7))

bplot = seaborn.boxplot(y="maes", 
                x="typ", 
                data=df,
                width=0.6,
                showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"12"})

cols=["lightgrey", "darkgrey", "grey", "lightblue"]
for i in range(3):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
        mybox.set_edgecolor("black")
        
ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
ax.set_xlabel("", size=20, family='Palatino Linotype')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
    tick.label.set_fontfamily('Palatino Linotype') 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
    tick.label.set_fontfamily('Palatino Linotype') 
ax.set_xticklabels(labels = ["MLP\n(A1 - wide)", "MLP\n(A2 - deep)","MLP\n(A3 - AP)", "Random\nForest"], size=20, family='Palatino Linotype')
#ax.set_ylim(bottom=0.5, top = 1.2)
#%%
maes = []
model = []
y_preds_mlp8 = np.load(os.path.join(data_dir, f"python\outputs\models\mlp8\\relu\y_preds.npy")).squeeze(2)
mae_mean = []
for i in range(5):
    maes.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_mlp8)[:,i]))
    mae_mean.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_mlp8)[:,i]))
    model.append("mlp0")
visualizations.plot_prediction(Y_test,y_preds_mlp8, np.mean(mae_mean),"")

y_preds_rf0 = np.load(os.path.join(data_dir, f"python\outputs\models\\rf0\y_preds.npy"))
mae_mean = []
for i in range(5):
    maes.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_rf0)[:,i]))
    mae_mean.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_rf0)[:,i]))
    model.append("rf0")
visualizations.plot_prediction(Y_test,y_preds_rf0, np.mean(mae_mean), "")

#prelesGPP_calib =  pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\data\profound\outputhyyitala2008calib", sep=";")
#visualizations.plot_prediction(Y_test,prelesGPP_calib, None, "")

df = pd.DataFrame(list(zip(maes, model)),
                  columns=["maes", "model"]) 

#%%
fig, ax = plt.subplots(figsize=(7,7))

bplot = seaborn.boxplot(y="maes", 
                x="model", 
                data=df,
                width=0.6,
                showmeans=True,
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"12"})

cols= ["grey", "red"]
for i in range(2):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
        mybox.set_edgecolor("black")
        
ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
ax.set_xlabel("", size=20, family='Palatino Linotype')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
    tick.label.set_fontfamily('Palatino Linotype') 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
    tick.label.set_fontfamily('Palatino Linotype') 
ax.set_xticklabels(labels = ["MLP\n(A4 - AP)", "Random\nForest"], size=20, family='Palatino Linotype')

#%% PLOT 1
def plot1(colors, log=False):
    
    plt.figure(num=None, figsize=(7, 7), facecolor='w', edgecolor='k')

    
    xi = [[subtab2.loc[(subtab2.id =="MLP0base")]["mae_val"].item()],
                [subtab2.loc[(subtab2.id =="MLP4base")]["mae_val"].item()],
                [subtab2.loc[(subtab2.id =="MLP5base")]["mae_val"].item()],
                [subtab2.loc[(subtab2.task =="pretraining_A")& (subtab2.simsfrac == 100)]["mae_val"]],
                [subtab2.loc[(subtab2.task =="finetuning") & (subtab2.finetuned_type != "A") & (subtab2.simsfrac == 100) ]["mae_val"]], 
                [subtab2.loc[(subtab2.task =="finetuning") & (subtab2.finetuned_type == "A")& (subtab2.simsfrac == 100) ]["mae_val"]],
                [subtab2.loc[(subtab2.task =="processmodel") & (subtab2.typ == 0)]["mae_val"].item()],
                [subtab2.loc[(subtab2.task =="selected") & (subtab2.model == "rf")]["mae_val"].item()]
                ]
    yi = [[subtab2.loc[(subtab2.id =="MLP0base")]["rmse_val"].item()],
                [subtab2.loc[(subtab2.id =="MLP4base")]["rmse_val"].item()],
                [subtab2.loc[(subtab2.id =="MLP5base")]["rmse_val"].item()],
                [subtab2.loc[(subtab2.task =="pretraining_A")& (subtab2.simsfrac == 100)]["rmse_val"]],
                [subtab2.loc[(subtab2.task =="finetuning")& (subtab2.finetuned_type != "A") & (subtab2.simsfrac == 100)]["rmse_val"]],
                [subtab2.loc[(subtab2.task =="finetuning") & (subtab2.finetuned_type == "A") & (subtab2.simsfrac == 100)]["rmse_val"]],
                [subtab2.loc[(subtab2.task =="processmodel") & (subtab2.typ == 0)]["rmse_val"].item()],
                [subtab2.loc[(subtab2.task =="selected") & (subtab2.model == "rf")]["rmse_val"].item()]
                ]
    #m = ['o','o', 'o', 'x', 's', "*", '*', 'o']
    m = [ 'o', 'o', 'o','o', 'o','o', "*", '*']
    #s = [60, 60, 60, 60, 60, 200, 200, 60]
    s = [100, 100, 100, 100, 100, 100, 250, 250]
    a = [1,1,1,0.8, 0.6, 0.8, 0.8, 0.8]
    # labs = ["selected", None,None, "finetuned", "pretrained", "PRELES", "RandomForest", None]
    labs = [ "Base", "", "","Pre-trained", "Fine-tuned", "", "PRELES", "RandomForest"]
    for i in range(len(xi)):
        if log:
            plt.scatter(xi[i], yi[i], alpha = a[i], color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.yscale("log")
            plt.xscale("log")
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            plt.xlabel("log(MAE test)", size = 20)
            plt.ylabel("log(MAE training)", size=20)
        else:
            plt.rcParams.update({'font.size': 20})
            plt.scatter(xi[i], yi[i], alpha = 0.8, color = colors[i], marker=m[i], s = s[i], label=labs[i])
            plt.xlabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20)
            plt.ylabel("Root mean squared error [g C m$^{-2}$ day$^{-1}$]", size=20)
            #plt.ylim(bottom=0)
            #plt.xlim(bottom = 0)
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)

        plt.legend(loc="upper left", prop={'size': 20})
#%%
plot1(colors = ["black", "black", "black","green", "lightblue", "lightblue", "red", "orange"], log=False)


#%% PLOT 1.3: BUT ONLY for full simulations!! (sims_frac100, errors over CV folds)
def DA_performances(models, df, 
                    lowerlim = 0.3, upperlim=3.0,
                    cols = ["gray", "blue", "purple", "green"],
                    log = False
                    #labs = ["Base MLP", "Adaptive\nPooling","Wide","Deep"]
                    ):
    plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
    bplot = seaborn.boxplot(y="mae_val",
                x = "archs",
                hue = "typ",
                hue_order = ["Base", "$\mathcal{D}_{S,7}$", "$\mathcal{D}_{S,12}$"],
                palette = "colorblind",
                data = df,
                width=0.6,
                showmeans=True,
                orient = "v",
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"12"}
                )
    if log:
        bplot.set_yscale("log")
    #for i in range(len(models)):
    #    mybox = bplot.artists[i]
    #    mybox.set_facecolor(cols[i])

    plt.ylim(bottom=lowerlim, top=upperlim)
    plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    plt.xlabel("", size = 20, family='Palatino Linotype')
    #bplot.set_xticklabels(labs)
    plt.xticks(size=20, family='Palatino Linotype')
    plt.yticks(size=20, family='Palatino Linotype')
    #plt.legend([], [], frameon=False)
    plt.legend(loc="upper left", prop = {'size':20, 'family':'Palatino Linotype'})
    #i=0
    #for legpatch in plt.legend().get_patches():
    #    #col = legpatch.get_facecolor()
    #    legpatch.set_edgecolor("black")
    #    legpatch.set_facecolor(cols[i])
    #    i = i+1
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X_test, y_tests = preprocessing.get_splits(sites = ['hyytiala'],
                                          years = [2008],
                                          datadir = os.path.join(data_dir, "data"), 
                                          dataset = "profound",
                                          simulations = None)
#%%
archs = []
types = []
labs = ["A1 - shallow","A2 - deep", "A3 - AP"]
#labs = ["MLP\n(A3 - AP)", "MLP\n(A4 - AP)"]

finetuned_nns = [10,12,7]
maes = []
for i in range(len(finetuned_nns)):
    predictions_test, errors = finetuning.featureExtractorC("mlp", finetuned_nns[i], None, 100, False)
    mae = []
    for err in errors[3]:
        maes.append(err)
        mae.append(err)
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,7}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))
    
finetuned_nns = [13,14,5]

for i in range(len(finetuned_nns)):
    predictions_test, errors = finetuning.featureExtractorC("mlp", finetuned_nns[i], None, 100, True)
    mae = []
    for err in errors[3]:
        #if finetuned_nns[i] == 14: 
        #    print("setting error ", err, " of model ", finetuned_nns[i], " to None")
       #     err = None
        #else:
        print("error ", err, " of model ", finetuned_nns[i])
        maes.append(err)
        mae.append(err)
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,12}$")
    try:
        print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))
    except:
        pass

#base_nns =  [0,4,5] 
#for i in range(len(base_nns)):
#    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{base_nns[i]}\\relu\y_preds.npy"), allow_pickle=True)
#    for j in range(5):
#        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), y_preds.squeeze(2)[j,:]))
#        archs.append(labs[i])
#        types.append("Base") #$\mathcal{D}_{T}$
        
 
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])
df["archs"] = archs

df_bp = base_predictions([0, 4, 5], 
                         architectures = ["A1 - shallow", "A2 - deep", "A3 - AP"], 
                         types = "Base", 
                         dummies = False, 
                         rf = False)

res = pd.concat([df, df_bp])
#%%
mods = [0,10,13,4,12,14, 5,7,5] #[5, 5, 8] # #,3,5]
DA_performances(mods, res, 
                upperlim=1.6,
                lowerlim=0.4,
                cols = ["grey","purple","plum", "grey", "green","lightgreen", "grey", "yellow", "khaki"],#, ["grey", "yellow", "blue"],  "grey", "yellow"],
                log=False)


#%%
archs = []
types = []
labs = ["A1 - shallow","A2 - deep", "A3 - AP"]
#labs = ["MLP\n(A3 - AP)", "MLP\n(A4 - AP)"]

finetuned_nns = [10,12,7]
#finetuned_nns = [13,14,5]
maes = []
for i in range(len(finetuned_nns)):
    predictions_test, errors = finetuning.featureExtractorA("mlp", finetuned_nns[i], None, 100, False)
    mae = []
    for err in errors[3]:
        maes.append(err)
        mae.append(err)
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,7}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))

finetuned_nns = [13,14,5]
for i in range(len(finetuned_nns)):
    predictions_test, errors = finetuning.featureExtractorA("mlp", finetuned_nns[i], None, 100, True)
    mae = []
    for err in errors[3]:
        maes.append(err)
        mae.append(err)
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,12}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))

#base_nns =  [0,4,5] 
#for i in range(len(base_nns)):
#    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{base_nns[i]}\\relu\y_preds.npy"), allow_pickle=True)
#    for j in range(5):
#        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), y_preds.squeeze(2)[j,:]))
#        archs.append(labs[i])
#        types.append("Base")
        
 
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])
df["archs"] = archs

df_bp = base_predictions([0, 4, 5], 
                         architectures = ["A1 - shallow", "A2 - deep", "A3 - AP"], 
                         types = "Base", 
                         dummies = False, 
                         rf = False)

res = pd.concat([df, df_bp])
#%%
#models = [0,13,4,14, 5,5]
mods = [0,10,13,4,12,14, 5,7,5] #[5, 5, 8] # #,3,5]
DA_performances(mods, res, 
                upperlim=2.6,
                lowerlim=0.4,
                cols = ["grey","purple","plum", "grey", "green","lightgreen", "grey", "yellow", "khaki"],#, ["grey", "yellow", "blue"],  "grey", "yellow"],
                log=False)

#%%
archs = []
types = []
labs = ["A1 - shallow","A2 - deep", "A3 - AP"]
#labs = ["MLP\n(A3 - AP)", "MLP\n(A4 - AP)"]

finetuned_nns = [10,12,7]
#finetuned_nns = [13,14,5]
maes = []
for i in range(len(finetuned_nns)):
    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\sims_frac100\\tuned\setting0\\y_preds.npy"), allow_pickle=True)
    mae = []
    for j in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        mae.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,7}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))


finetuned_nns = [13,14,5]
for i in range(len(finetuned_nns)):
    if finetuned_nns[i] == 5:
        y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\dummies\sims_frac100\\tuned\setting0\\y_preds.npy"), allow_pickle=True)
    else:
        y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\sims_frac100\\tuned\setting0\\y_preds.npy"), allow_pickle=True)
    mae = []
    for j in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        mae.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,12}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))

       
#base_nns =  [0,4,5]
#for i in range(len(base_nns)):
#    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{base_nns[i]}\\relu\y_preds.npy"), allow_pickle=True)
#    for j in range(5):
#        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
#        archs.append(labs[i])
#        types.append("Base")
        
 
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])
df["archs"] = archs

df_bp = base_predictions([0, 4, 5], 
                         architectures = ["A1 - shallow", "A2 - deep", "A3 - AP"], 
                         types = "Base", 
                         dummies = False, 
                         rf = False)

res = pd.concat([df, df_bp])
#%%
mods = [0,10,13,4,12,14, 5,7,5] #[5, 5, 8] # #,3,5]
DA_performances(mods, res, 
                upperlim=None,
                lowerlim=0.4,
                cols = ["grey","purple","plum", "grey", "green","lightgreen", "grey", "yellow", "khaki"],#, ["grey", "yellow", "blue"],  "grey", "yellow"],
                log=False)

#%%
archs = []
types = []
labs = ["A1 - shallow","A2 - deep", "A3 - AP"]

finetuned_nns = [10,12,7]
#finetuned_nns = [13,14,5]
maes = []
for i in range(len(finetuned_nns)):
    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\sims_frac100\\tuned\setting1\\y_preds.npy"), allow_pickle=True)
    mae = []
    for j in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        mae.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,7}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))

finetuned_nns = [13,14,5]
for i in range(len(finetuned_nns)):
    if finetuned_nns[i] == 5:
        y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\dummies\sims_frac100\\tuned\setting1\\freeze2\y_preds.npy"), allow_pickle=True)
    else:
        y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{finetuned_nns[i]}\\nodropout\sims_frac100\\tuned\setting1\\y_preds.npy"), allow_pickle=True)
    mae = []
    for j in range(5):
        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        mae.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
        archs.append(labs[i])
        types.append("$\mathcal{D}_{S,12}$")
    print(f"model {finetuned_nns[i]} mean and std: ", np.round(np.mean(mae), 4), np.round(np.std(mae), 2))

   
#base_nns =  [0,4,5]
#for i in range(len(base_nns)):
#    y_preds= np.load(os.path.join(data_dir, f"python\outputs\models\mlp{base_nns[i]}\\relu\y_preds.npy"), allow_pickle=True)
#    for j in range(5):
#        maes.append(metrics.mean_absolute_error(y_tests.squeeze(1), np.transpose(y_preds.squeeze(2)[j,:])))
#        archs.append(labs[i])
#        types.append("Base")
        
 
df = pd.DataFrame(list(zip(types, maes)),
                  columns=["typ", "mae_val"])
df["archs"] = archs 

df_bp = base_predictions([0, 4, 5], 
                         architectures = ["A1 - shallow", "A2 - deep", "A3 - AP"], 
                         types = "Base", 
                         dummies = False, 
                         rf = False)

res = pd.concat([df, df_bp])
#%%
mods = [0,10,13,4,12,14, 5,7,5] #[5, 5, 8] # #,3,5]
DA_performances(mods, res, 
                upperlim=None,
                lowerlim=0.4,
                cols = ["grey","purple","plum", "grey", "green","lightgreen", "grey", "yellow", "khaki"],#, ["grey", "yellow", "blue"],  "grey", "yellow"],
                log=False)

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
def plot2(typ, frac, epochs = 5000,
          pos_preles = 5.0,
          pos_fw2 = 5.0,
          pos_ols = 6.9,
          colors_test_loss = ["purple", "plum"],
          label = ""):

    bm = fulltab.loc[(fulltab.task == "finetuning") & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bm.iloc[bm['mae_val'].idxmin()].to_dict()
    # now load the model losses from file.
    rl = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting0\\running_losses.npy", allow_pickle=True).item()
    #rl4 = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp4\\relu\\running_losses.npy", allow_pickle=True).item()
    running_losses([rl["mae_train"][:,:epochs]], 
               [rl["mae_val"][:,:epochs]],
               length = epochs, lowerlim = 0.3,
               colors1 = [ colors_test_loss[1]],
               colors2 = [colors_test_loss[0]],
               labels = [label])
    #visualizations.plot_running_losses(rl["mae_train"][:, :epochs], rl["mae_val"][:, :epochs], False, False,
    #                                   colors_test_loss = colors_test_loss)


    #bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    #bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #rf = fulltab.loc[(fulltab.model == "rf")]["mae_val"].item()
    
    #plt.hlines(bestmlp0, 0, epochs,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    #plt.hlines(rf, 0, 2000,colors="yellow", linestyles="dashed", label="Random Forest", linewidth=1.2)
    
    bm = fulltab.loc[(fulltab.typ == typ) & (fulltab.simsfrac == frac) & (fulltab.finetuned_type == "C-OLS")].reset_index()
    bestols = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    try:
        posols = np.max(np.where(np.mean(rl["mae_val"][:, :epochs], axis=0) > bestols))
        plt.arrow(x=posols, y=pos_ols, dx=0, dy=-(pos_ols-bestols), linewidth=1.2, color="gray")
        # mlp10: plt.text(x=posols, y=1.05, s="OLS", fontstyle="italic", fontsize=16)
        # mlp7: plt.text(x=posols, y=1.05, s="OLS", fontstyle="italic", fontsize=16)
        plt.text(x=posols, y=pos_ols+0.01, s="OLS", fontfamily="Palatino Linotype", fontsize=18)
    
    except:   
        print("OLS worse than initial NN prediction")

    try:
        bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "B-fW2")].reset_index()
        bestfw2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
        posfw2 = np.max(np.where(np.mean(rl["mae_val"][:, :epochs], axis=0) > bestfw2))
        plt.arrow(x=posfw2, y=pos_fw2, dx=0, dy=-(pos_fw2-bestfw2), linewidth=1.2, color="gray")
        # mlp7: plt.text(x=posfw2, y=1.25, s="Partial\nback-prop", fontstyle="italic", fontsize=16)
        plt.text(x=posfw2, y=pos_fw2+0.01, s="Re-train\nlast layer", fontfamily="Palatino Linotype", fontsize=18)
    except:
        print("no fw2.")
    
    #bm = fulltab.loc[(fulltab.typ == typ)& (fulltab.simsfrac == frac)  & (fulltab.finetuned_type == "D-MLP2")].reset_index()
    #bestmlp2 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #posmlp2 = np.max(np.where(rl["mae_val"][:, :epochs] > bestmlp2))
    #plt.arrow(x=posmlp2, y=1.2, dx=0, dy=-(1.2-bestmlp2), linewidth=0.8, color="gray")
    #plt.text(x=posmlp2, y=1.25, s="MLP")

    prel = fulltab.loc[(fulltab.model == "preles") & (fulltab.typ == 0)]["mae_train"].item()
    posprel = np.max(np.where(np.mean(rl["mae_val"][:, :epochs], axis=0) > prel))
    plt.arrow(x=posprel, y=pos_preles, dx=0, dy=-(pos_preles-prel), linewidth=1.2, color="gray")
    # mlp7: plt.text(x=posprel-30, y=1.55, s="PRELES", fontstyle="italic", fontsize=16)
    plt.text(x=posprel-30, y=pos_preles+0.01, s="PRELES", fontfamily="Palatino Linotype", fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend(prop = {'size':20, 'family':'Palatino Linotype'})
#%%
plot2(5,100, 3000,
          pos_preles = 1.5,
          pos_fw2 = 2.0,
          pos_ols = 1.8,
          colors_test_loss = ["yellow", "khaki"],
          label = "MLP (A3 - AP)")
plot2(7,100, 1000,
          pos_preles = 2.5,
          pos_fw2 = 4.0,
          pos_ols = 3.5)
plot2(8,100, 1000,
          pos_preles = 5.0,
          pos_fw2 = 7.5,
          pos_ols = 10.0,
          colors_test_loss = ["blue", "lightblue"],
          label = "MLP (A4 - AP)")
plot2(9,100, 2000,
          pos_preles = 1.0,
          pos_fw2 = 0.7,
          pos_ols = 0.8)
plot2(10,100, 2000,
          pos_preles = 1.2,
          pos_fw2 = 0.9,
          pos_ols = 1.0,
          colors_test_loss = ["purple", "plum"],
          label = "MLP (A1 - Wide)")
plot2(11,100, 5000,
          pos_preles = 0.9,
          pos_fw2 = 0.7,
          pos_ols = 1.0)
plot2(12,100, 5000,
          pos_preles = 1.2,
          pos_fw2 = 0.9,
          pos_ols = 1.0,
          colors_test_loss = ["green", "lightgreen"],
          label = "MLP (A1 - Deep)")
   
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


#%% Plot X: Errorbar plot for size of source domain

def plot5(plot="architectures"):
    
    df = pd.DataFrame(None, columns = ["fracs", "models", "run", "mae"])

    plt.rcParams.update({'font.size': 20})
    fig,ax = plt.subplots(figsize=(7.1,7.1))
    seaborn.set(style= "ticks")

    if plot=="architectures":
        mods = [10,12,5]
        labels = ["A1", "A2","A3"]
        cols = ["purple", "green","yellow"]
    for i in range(len(mods)):
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
                            "models":labels[i],
                            "fracs":frac},ignore_index=True)
            means.append(np.mean(errors))
            sds.append(np.std(errors)*2)
    
    bplot = seaborn.boxplot(x = "fracs",
                    y = "mae",
                    hue = "models",
                    width=0.4,
                    data = df,
                    dodge = True,
                    showmeans = True,
                    meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"6"})
    cols_long = cols*len(fracs)
    for i in range(len(mods)*len(fracs)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols_long[i])
        mybox.set_edgecolor("black")
        
    ax.legend(loc="lower right", prop={'size': 20, 'family':'Palatino Linotype'})
    i=0
    for legpatch in ax.get_legend().get_patches():
        #col = legpatch.get_facecolor()
        legpatch.set_edgecolor("black")
        legpatch.set_facecolor(cols_long[i])
        i = i+1
        
    
    ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    ax.set_xlabel("Size of source domain [datapoints]", size=20, family='Palatino Linotype')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    ax.set_xticklabels(labels = ["15000", "25000", "35000",  "50000"], size=20, family='Palatino Linotype')
    ax.set_ylim(bottom=0.5, top = 1.2)
            

#%%
plot5(plot="architectures")
plot5(plot="parameters")

#%% Plot 6: ERRORBARS plot for DROPOUT A3 (WITH ADAPTIVE POOLING)
def plot6():
    
    df = pd.DataFrame(None, columns = ["dropout", "models", "run", "mae_f1", "mae_f2"])
    df2 = pd.DataFrame(None, columns = ["dropout", "mae_mean_f1", "mae_std_f1", "mae_mean_f2", "mae_std_f2"])

    plt.rcParams.update({'font.size': 20})
    fig,ax = plt.subplots(figsize=(7.1,7.1))
    seaborn.set(style= "ticks")

    dp = [0, 1,2,3,4,5, 6, 7, 8, 9]
    sds_f1 = [] 
    means_f1 = []
    sds_f2 = [] 
    means_f2 = []
    for i in dp:  
        if i< 1:
            y_preds_f1 = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\nodropout\\sims_frac100\\tuned\setting1\\freeze1\y_preds.npy")).squeeze(2)
            y_preds_f2 = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\nodropout\\sims_frac100\\tuned\setting1\\freeze2\y_preds.npy")).squeeze(2)
        else:
            y_preds_f1 = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\dropout\\0{i}\sims_frac100\\tuned\setting1\\freeze1\y_preds.npy")).squeeze(2)
            y_preds_f2 = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\dropout\\0{i}\sims_frac100\\tuned\setting1\\freeze2\y_preds.npy")).squeeze(2)
            #y_tests = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\dropout\\0{i}\sims_frac100\\tuned\setting1\y_tests.npy")).squeeze(2)
        errors_f1 = []
        errors_f2 = []
        for model in range(5):
            errors_f1.append(metrics.mean_absolute_error(Y_test, y_preds_f1[model,:]))
            errors_f2.append(metrics.mean_absolute_error(Y_test, y_preds_f2[model,:]))
            df = df.append({"mae_f1":metrics.mean_absolute_error(Y_test, y_preds_f1[model,:]),
                            "mae_f2":metrics.mean_absolute_error(Y_test, y_preds_f2[model,:]),
                            "run":model,
                            "models":5,
                            "dropout":i},ignore_index=True)
        means_f1.append(np.mean(errors_f1))
        sds_f1.append(np.std(errors_f1)*2)
        means_f2.append(np.mean(errors_f2))
        sds_f2.append(np.std(errors_f2)*2)
        
        df2 = df2.append({"mae_mean_f1":np.mean(errors_f1),
                          "mae_std_f1":np.std(errors_f1)*2,
                          "mae_mean_f2":np.mean(errors_f2),
                          "mae_std_f2":np.std(errors_f2)*2,
                            "dropout":i},ignore_index=True)

    
    plt.errorbar(dp, means_f1, yerr=sds_f1, capsize=4, capthick = 4, linewidth=3, color="darkblue", label="Re-train H2")
    plt.errorbar(dp, means_f2, yerr=sds_f2, capsize=4, capthick = 4, linewidth=3, color="green", label="Re-train H1+H2")

    ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    ax.set_xlabel("Dropout probability", size=20, family='Palatino Linotype')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    ax.set_xticklabels(labels = ["", "0.0",  "0.2", "0.4",  "0.6",  "0.8",  "1.0"], size=20, family='Palatino Linotype')
    ax.set_xlim((-0.2,9.5))
    ax.set_ylim((0.4,1.9))
    plt.legend(loc="upper left", prop = {'size':20, 'family':'Palatino Linotype'})
    
    return df2
            

#%%
df2 = plot6()

#%% Szie of the source domain: NUMBERS
maes = []
for frac in [30, 50, 70, 100]:
    y_preds = np.load(os.path.join(data_dir, f"python\outputs\models\mlp10\\nodropout\\sims_frac{frac}\y_preds.npy")).squeeze(2)
    y_tests = np.load(os.path.join(data_dir, f"python\outputs\models\mlp10\\nodropout\\sims_frac{frac}\y_tests.npy")).squeeze(2)
    
    for i in range(5):
        maes.append(metrics.mean_absolute_error(y_tests[i,:], y_preds[i,:]))
        
    print(f"For simsfrac {frac}: ", np.round(np.mean(maes), 4),np.round(np.std(maes), 4))
#%%
def running_losses(val_losses, 
                   length = 10000, lowerlim = 0.0, upperlim = None,
                   colors1 = ["lightgreen",  "thistle"],
                   colors2 = ["green",  "blueviolet"],
                   labels = ["fixed", "uniform"],
                   legend = True,
                   CI = True):

    #if model=="mlp":
    #    colors = ["blue","lightblue"]
    #elif model=="cnn":
    #    colors = ["darkgreen", "palegreen"]
    #elif model=="lstm":
    #    colors = ["blueviolet", "thistle"]
    #else:

    fig, ax = plt.subplots(figsize=(7,7))
    
    for i in range(len(val_losses)):
        #ci_train = np.quantile(train_losses[i], (0.05,0.95), axis=0)
        ci_val = np.quantile(val_losses[i], (0.05,0.95), axis=0)
        #train_loss = np.mean(train_losses[i], axis=0)
        val_loss = np.mean(val_losses[i], axis=0)
       # ax.fill_between(np.arange(len(train_losses[i])), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        if CI:
            ax.fill_between(np.arange(length), ci_val[0],ci_val[1], color=colors1[i], alpha=0.3)
        
        #ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=1.2)
        ax.plot(val_loss, color=colors2[i], label = labels[i], linewidth=1.2)

    ax.set_xlabel("Epochs", size=20, family='Palatino Linotype')
    ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(20) 
                    tick.label.set_fontfamily('Palatino Linotype')
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)
                    tick.label.set_fontfamily('Palatino Linotype')
    #bm = fulltab.loc[(fulltab.id == "MLP4nP1D0R")].reset_index()
    #bestmlp0 = bm.iloc[bm['mae_val'].idxmin()].to_dict()["mae_val"]
    #plt.hlines(bestmlp0, -1, length,colors="gray", linestyles="dashed", label="Best MLP", linewidth=2)
    plt.ylim(bottom = lowerlim, top = upperlim)
    if legend:
        plt.legend(loc="upper right", prop = {'size':20, 'family':'Palatino Linotype'})
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

#%%
rl4 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp4\relu\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl0 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp0\relu\running_losses.npy"), allow_pickle=True).item()
rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\relu\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses(
               [rl5["mae_val"][:,:epochs], rl0["mae_val"][:,:epochs],  rl4["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "mediumseagreen", "grey"],
               colors2 = ["darkblue",  "seagreen", "dimgrey"],
               labels = [ "A1 - shallow","A2 - deep", "A3 - AP"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS



#%%
rl7 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp7\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl10 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp10\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
rl12 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp12\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses(
               [ rl10["mae_val"][:,:epochs],  rl12["mae_val"][:,:epochs], rl7["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "mediumseagreen","grey"],
               colors2 = [ "darkblue",  "seagreen", "dimgrey"],
               labels = ["A1 - shallow","A2 - deep",  "A3 - AP"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
#%%
rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl13 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp13\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
rl14 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp14\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses( 
               [rl13["mae_val"][:,:epochs],  rl14["mae_val"][:,:epochs], rl5["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = [ "mediumblue",  "mediumseagreen", "grey"],
               colors2 = [ "darkblue",  "seagreen", "dimgrey"],
               labels = [ "A1 - shallow","A2 - deep",  "A3 - AP"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%%
rl8 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp8\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()


epochs = 10000
running_losses([rl8["mae_train"][:,:epochs]], 
               [rl8["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "mediumseagreen", "grey"],
               colors2 = ["darkblue",  "seagreen", "dimgrey"],
               labels = ["MLP (A4 - AP)"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%%
rl0 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp0\relu\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl0p = np.load(os.path.join(data_dir,r"python\outputs\models\mlp10\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

epochs = 2500
running_losses([rl0["mae_train"][:,:epochs],  rl0p["mae_train"][:,:epochs]], 
               [rl0["mae_val"][:,:epochs],  rl0p["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "lightgreen"],
               colors2 = ["darkblue",  "green"],
               labels = ["Base", "Pre-training"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
#%%
rl4 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp4\relu\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl4p = np.load(os.path.join(data_dir,r"python\outputs\models\mlp12\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses([rl4["mae_train"][:,:epochs],  rl4p["mae_train"][:,:epochs]], 
               [rl4["mae_val"][:,:epochs],  rl4p["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "lightgreen"],
               colors2 = ["darkblue",  "green"],
               labels = ["Base", "Pre-training"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
#%%
rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\relu\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl5p = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses([rl5["mae_train"][:,:epochs],  rl5p["mae_train"][:,:epochs]], 
               [rl5["mae_val"][:,:epochs],  rl5p["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "lightgreen"],
               colors2 = ["darkblue",  "green"],
               labels = ["Base", "Pre-training"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
#%%
rl5p = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()
#rl6 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp6\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl8p = np.load(os.path.join(data_dir,r"python\outputs\models\mlp8\nodropout\sims_frac100\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses([rl5p["mae_train"][:,:epochs],  rl8p["mae_train"][:,:epochs]], 
               [rl5p["mae_val"][:,:epochs],  rl8p["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["mediumblue",  "lightgreen"],
               colors2 = ["darkblue",  "green"],
               labels = ["A3 - Bily Kriz", "A4 - Simulations"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%% RETRAIN LAST LAYER: LEARNINGCURVES
rl10 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp10\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl12 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp12\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
#rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses([rl10["mae_train"][:,:epochs], rl12["mae_train"][:,:epochs]], #, rl5["mae_train"][:,:epochs]], 
               [rl10["mae_val"][:,:epochs], rl12["mae_val"][:,:epochs]], #, rl5["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = [ "plum", "lightgreen"],
               colors2 = ["purple", "green"],
               labels = [ "MLP (A1 - Shallow)", "MLP (A2 - Deep)"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()
rl8 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp8\nodropout\sims_frac100\tuned\setting1\running_losses.npy"), allow_pickle=True).item()

epochs = 5000
running_losses([rl5["mae_train"][:,:epochs], rl8["mae_train"][:,:epochs]], 
               [rl5["mae_val"][:,:epochs], rl8["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = [ "khaki", "lightblue"],
               colors2 = ["yellow", "blue"],
               labels = [ "MLP (A3 - AP)", "MLP (A4 - AP)"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%% FULL-BACKPROPAGATION: LEARNINGCURVES
rl10 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp10\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
rl12 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp12\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
#rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()

epochs = 2000
running_losses([rl10["mae_train"][:,:epochs], rl12["mae_train"][:,:epochs]], #, rl5["mae_train"][:,:epochs]], 
               [rl10["mae_val"][:,:epochs], rl12["mae_val"][:,:epochs]], #, rl5["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = [ "plum", "lightgreen"],
               colors2 = ["purple", "green"],
               labels = [ "MLP (A1 - Shallow)", "MLP (A2 - Deep)"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

rl5 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp5\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()
rl8 = np.load(os.path.join(data_dir,r"python\outputs\models\mlp8\nodropout\sims_frac100\tuned\setting0\running_losses.npy"), allow_pickle=True).item()

epochs = 2000
running_losses([rl5["mae_train"][:,:epochs], rl8["mae_train"][:,:epochs]], 
               [rl5["mae_val"][:,:epochs], rl8["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = [ "khaki", "lightblue"],
               colors2 = ["yellow", "blue"],
               labels = ["MLP (A3 - AP)", "MLP (A4 - AP)"],
               lowerlim=None)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%% PLOT PARAMETER SAMPLES
data_dir = 'OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt'
params = pd.read_csv(os.path.join(data_dir, r"data\parameter_default_values.csv"), sep =";", names=["name", "value"])

def plot4(parameter, label):
    
    data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
    si_n = pd.read_csv(os.path.join(data_dir, r"data\simulations\normal_params\sims_in.csv"), sep =";")
    si_u = pd.read_csv(os.path.join(data_dir, r"data\simulations\uniform_params\sims_in.csv"), sep =";")
    
    fig, ax = plt.subplots(figsize=(7,7))
    plt.hist(si_u[parameter], bins=50, histtype="stepfilled", density=False, color="blue",alpha=0.7, linewidth=1.2, label="Uniform")
    plt.hist(si_n[parameter], bins=50, histtype="stepfilled", density=False, color="orange", alpha=0.7, linewidth=1.2, label="Normal")
    plt.xlabel(label, fontsize=26, family='Palatino Linotype')
    plt.ylabel("Counts", fontsize=26, family='Palatino Linotype')
    plt.xticks(size=26, family='Palatino Linotype')
    plt.yticks(size=26, family='Palatino Linotype')
    plt.ylim(bottom = 0, top= 2900)
    plt.legend(prop={"size":26, 'family':'Palatino Linotype'})
#%%
plot4("beta","Beta")
plot4("X0", "X0")
plot4("alpha","Alpha")
plot4("gamma", "Gamma")
plot4("chi", "Chi")

#%% PLOT CLIMATE SIMULATIONS
X_sims, Y_sims = preprocessing.get_simulations(os.path.join(data_dir, r"data\simulations\uniform_params"),
                                     to_numpy = False,
                                     DOY=True,
                                     standardized=False)
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001, 2002, 2003, 2004, 2005, 2006, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos", "DOY"],
                                simulations = None,
                                standardized = False,
                                to_numpy=False)

plt.figure(figsize=(7,7))
plt.scatter(X_sims["DOY"], Y_sims, alpha=0.7, color="grey", s=10, label="Simulated")
plt.scatter(X["DOY"], Y, alpha=0.7, s = 12, color="green", label="Observed")
plt.xticks(size=20, family='Palatino Linotype')
plt.yticks(size=20, family='Palatino Linotype')
plt.xlabel("Day of year", size=20, family='Palatino Linotype')
plt.ylabel("Gross Primary Production [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
lgnd = plt.legend(loc="upper right", prop={'size':20, 'family':'Palatino Linotype'})
#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [40]
lgnd.legendHandles[1]._sizes = [40]
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

#%%
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%%
def sparse_results(dd_yref, dd_ytrans):

        sparse = [1,2,3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,100]
        dp = [int(np.floor(X.shape[0]/100*spa)) for spa in sparse]
        errors = []
        steps = []
        fb_errors = []
        fw_errors = []
        ols_errors = []
        direct_errors = []
        for spa in sparse:
            y_preds = np.load(os.path.join(dd_yref, f"{spa}\y_preds.npy")).squeeze(2)
            y_preds_fb = np.load(os.path.join(dd_ytrans, f"setting0\sparse\\{spa}\y_preds.npy")).squeeze(2)
            y_preds_fw = np.load(os.path.join(dd_ytrans, f"setting1\sparse\\{spa}\y_preds.npy")).squeeze(2)
            predictions_test, ols_errs = finetuning.featureExtractorC("mlp", 4, None, 100, sparse = spa)
            predictions_test, direct_errs = finetuning.featureExtractorA("mlp", 4, None, 100, sparse = spa, dummies = False)
            for i in range(5):
                errors.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds)[:,i]))
                fb_errors.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_fb)[:,i]))
                fw_errors.append(metrics.mean_absolute_error(Y_test, np.transpose(y_preds_fw)[:,i]))
                steps.append(spa)
                ols_errors.append(ols_errs[3][i])
                direct_errors.append(direct_errs[3][i])
        
        
        df = pd.DataFrame(list(zip(errors, steps)),
                          columns=["maes_test", "sparse"]) 
        df["datapoints"] = np.repeat(dp,5)
        df["maes_ols"] = ols_errors
        df["maes_direct"] = direct_errors
        df["maes_fb"] = fb_errors
        df["diff_fb"] = df["maes_test"] - df["maes_fb"]
        df["maes_fw"] = fw_errors
        df["diff_fw"] = df["maes_test"] - df["maes_fw"]
        
        return df, dp
#%%
df_shallow, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp0\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp10\\nodropout\sims_frac100\\tuned"))      

df_deep, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp4\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp12\\nodropout\sims_frac100\\tuned"))   

df_AP, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp5\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp7\\nodropout\sims_frac100\\tuned"))  

df_shallow_D, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp0\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp13\\nodropout\sims_frac100\\tuned"))      

df_deep_D, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp4\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp14\\nodropout\sims_frac100\\tuned"))   

df_AP_D, dp = sparse_results(dd_yref=os.path.join(data_dir, "python\outputs\models\mlp5\\relu\sparse\\"), 
                    dd_ytrans=os.path.join(data_dir, "python\outputs\models\mlp5\\nodropout\dummies\sims_frac100\\tuned")) 

#%%

def prep(df, finetuning = "fw", diff=True):
        
    means = df.groupby("datapoints")[f"maes_{finetuning}"].mean()
    sds = df.groupby("datapoints")[f"maes_{finetuning}"].std()
    q1 = df.groupby("datapoints")[f"maes_{finetuning}"].quantile(0.05)
    q2 = df.groupby("datapoints")[f"maes_{finetuning}"].quantile(0.95)
    
    print("Absolute means: ", np.array(means))
    print("Absolute standard deviations: ", np.array(2*sds))

    means_diff = df.groupby("datapoints")[f"diff_{finetuning}"].mean()
    sds_diff = df.groupby("datapoints")[f"diff_{finetuning}"].std()
    q1_diff = df.groupby("datapoints")[f"diff_{finetuning}"].quantile(0.05)
    q2_diff = df.groupby("datapoints")[f"diff_{finetuning}"].quantile(0.95)

    print("Difference in means: ", np.array(means_diff))
    print("Difference in standard deviations: ", np.array(2*sds_diff))
    
    means_test = df.groupby("datapoints")[f"maes_test"].mean()
    sds_test = df.groupby("datapoints")[f"maes_test"].std()
    
    print("Base network means: ", np.array(means_test))
    print("Base network standard deviations: ", np.array(2*sds_test))
    
    if diff:
        return means_diff, q1_diff, q2_diff
    else:
        return means, q1, q2
    
#%%
means, q1, q2 = prep(df_shallow, "fw", diff=True)
print(dp)
means, q1, q2 = prep(df_shallow_D, "fw", diff=True)
print(dp)
#%% PARTIAL BACK-PROB: Difference
        
def plot_sparse(df, dp, finetuning,
                colors = ["purple"], fill=["plum"],labels = ["E$_{reference}$ - E$_{finetuned}$"],
                diff=True):

    fig, ax = plt.subplots(figsize=(7,7))
    
    for d in range(len(df)):
        means, q1, q2 = prep(df[d], finetuning = finetuning, diff=diff)

        ax.fill_between(dp, q1,q2, color=fill[d], alpha=0.6)
        ax.plot(means, color=colors[d], label=labels[d] )

    plt.legend(loc="upper right", prop = {'size':20, 'family':'Palatino Linotype'})
    plt.xlabel("Available training data [days]", size=20, family='Palatino Linotype')
    if diff:
        plt.ylabel("Difference in mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    else:
        plt.ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    plt.xticks(size=20, family='Palatino Linotype')
    plt.yticks(size=20, family='Palatino Linotype')
    if diff:
        plt.ylim((-0.4, 0.3))
        plt.hlines(y = 0, xmin = 50, xmax = 1826, color="black", linestyle = "dashed")
    else:
        plt.ylim(top= 2.5)
#%%
plot_sparse([df_shallow, df_deep, df_AP], dp, "fw",
            colors = ["darkblue", "green", "gray"], fill=["lightblue", "lightgreen", "lightgray"], labels = ["A1 - shallow", "A2 - deep", "A3 - AP"],
            diff = False)
#%%
plot_sparse([df_shallow_D, df_deep_D, df_AP_D], dp, "fw", 
            colors = ["darkblue", "green", "gray"], fill=["lightblue", "lightgreen", "lightgray"], labels = ["A1 - shallow", "A2 - deep", "A3 - AP"],
            diff=False)

#%%
plot_sparse([df_shallow, df_deep, df_AP], dp, "fb",
            colors = ["darkblue", "green", "gray"], fill=["lightblue", "lightgreen", "lightgray"], labels = ["A1 - shallow", "A2 - deep", "A3 - AP"])
#%%
plot_sparse([df_shallow_D, df_deep_D, df_AP_D], dp, "fb", 
            colors = ["darkblue", "green", "gray"], fill=["lightblue", "lightgreen", "lightgray"], labels = ["A1 - shallow", "A2 - deep", "A3 - AP"])

#%%
sparse = [1,2,3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] #, 55, 60, 65, 70, 75, 80, 85, 90, 95,100]
rls = []
for spa in sparse:
     rls.append(np.load(os.path.join(data_dir,f"python\outputs\models\mlp0\\relu\\sparse\\{spa}\\running_losses.npy"), allow_pickle=True).item())
     rls.append(np.load(os.path.join(data_dir,f"python\outputs\models\mlp10\\nodropout\\sims_frac100\\tuned\\setting1\\sparse\\{spa}\\running_losses.npy"), allow_pickle=True).item())
   

epochs = 5000
running_losses(
               [rls[0]["mae_val"][:,:epochs], rls[1]["mae_val"][:,:epochs], rls[2]["mae_val"][:,:epochs], rls[3]["mae_val"][:,:epochs], rls[4]["mae_val"][:,:epochs], 
                rls[5]["mae_val"][:,:epochs], rls[6]["mae_val"][:,:epochs], rls[7]["mae_val"][:,:epochs], rls[8]["mae_val"][:,:epochs], rls[9]["mae_val"][:,:epochs],
                rls[10]["mae_val"][:,:epochs], rls[11]["mae_val"][:,:epochs], rls[12]["mae_val"][:,:epochs], rls[13]["mae_val"][:,:epochs], rls[14]["mae_val"][:,:epochs]],
               length = epochs,
               colors1 = ["grey", "lightblue",  "grey", "lightblue", "grey",  "lightblue", "grey","lightblue", "grey",  "lightblue", "grey",  "lightblue", "grey","lightblue", "grey"],
               colors2 = ["dimgrey", "blue",  "dimgrey", "blue", "dimgrey",  "blue", "dimgrey","blue", "dimgrey", "blue", "dimgrey",  "blue", "dimgrey","blue", "dimgrey"],
               labels = [ "A3 - AP", "","","","","","","","","","","","","",""],
               lowerlim=None,
               legend=False,
               CI=True)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS



#%%
dp_prob = [1,2,3,4,5, 6, 7, 8, 9]
rls = []
for dp in dp_prob:
    rls.append(np.load(os.path.join(data_dir,f"python\outputs\models\mlp5\dropout\\0{dp}\sims_frac100\\tuned\\setting1\\freeze1\\running_losses.npy"), allow_pickle=True).item())
    rls.append(np.load(os.path.join(data_dir,f"python\outputs\models\mlp5\dropout\\0{dp}\sims_frac100\\tuned\\setting1\\freeze2\\running_losses.npy"), allow_pickle=True).item())

epochs = 5000
running_losses(
               [rls[0]["mae_val"][:,:epochs], rls[1]["mae_val"][:,:epochs], rls[2]["mae_val"][:,:epochs], rls[3]["mae_val"][:,:epochs], rls[4]["mae_val"][:,:epochs], 
                rls[5]["mae_val"][:,:epochs], rls[6]["mae_val"][:,:epochs], rls[7]["mae_val"][:,:epochs], rls[8]["mae_val"][:,:epochs],
                rls[9]["mae_val"][:,:epochs], rls[10]["mae_val"][:,:epochs], rls[11]["mae_val"][:,:epochs], rls[12]["mae_val"][:,:epochs], rls[13]["mae_val"][:,:epochs], 
                rls[14]["mae_val"][:,:epochs], rls[15]["mae_val"][:,:epochs], rls[16]["mae_val"][:,:epochs], rls[17]["mae_val"][:,:epochs]],
               length = epochs,
               colors2 = ["darkblue", "green", "darkblue", "green", "darkblue", "green", "darkblue", "green", "darkblue", "green", "darkblue", "green", "darkblue", "green", "darkblue", "green" ,"darkblue", "green"],
               colors1 = ["lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen","lightblue", "lightgreen"],
               labels = [ "Re-train H2", "Re-train H1+H2","","","","","","","","", "","", "","", "","" ,"",""],
               lowerlim=None,
               legend=True)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS

#%%
freeze = [1,2]
rls = []
for fr in freeze:
    rls.append(np.load(os.path.join(data_dir,f"python\outputs\models\mlp5\\nodropout\\sims_frac100\\tuned\\setting1\\freeze{fr}\\running_losses.npy"), allow_pickle=True).item())
    
epochs = 5000
running_losses(
               [rls[0]["mae_val"][:,:epochs], rls[1]["mae_val"][:,:epochs]],
               length = epochs,
               colors2 = ["darkblue", "green"],
               colors1 = ["lightblue", "lightgreen"],
               labels = [ "Re-train H2", "Re-train H1+H2","","","","","","",""],
               lowerlim=None,
               legend=True)#%% PLOT4: PLOT PARAMETER SAMPLED FOR PRELES SIMULATIONS
