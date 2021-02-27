# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:32:31 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import numpy as np
import pandas as pd
import os.path
import setup.models as models
from ast import literal_eval
import torch.nn as nn
import torch
import setup.preprocessing as preprocessing
import setup.dev_mlp as dev_mlp
import setup.utils as utils
import collect_results
import finetuning
from sklearn import metrics

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"


X_train, Y_train = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%% Number of Network Parameters
#mods = [5,7,10,12,13,14]
mods = [0,4,5]
dummies = False
for mod in mods:
        
    hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", mod, None, data_dir, dummies)
    
    X_test = torch.tensor(X_test).type(dtype=torch.float)
    y_test = torch.tensor(Y_test).type(dtype=torch.float)
    X_train = torch.tensor(X).type(dtype=torch.float)
    y_train = torch.tensor(Y).type(dtype=torch.float)
        
    if ((mod > 7) | (mod == 0) | (mod == 4)):
        model = models.MLP(model_design["dimensions"], model_design["activation"])
    else:
        model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
    model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\relu\model0.pth")))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    
    rmse_train = []
    rmse_val = []
    mae_train = []
    mae_val= []

    for i in range(5):
        
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{mod}\\relu\model{i}.pth")))
    
        pred_train = model(X_train)
        pred_test = model(X_test)#
        with torch.no_grad():
            rmse_train.append(utils.rmse(y_train, pred_train))
            rmse_val.append(utils.rmse(y_test, pred_test))
            mae_train.append(metrics.mean_absolute_error(y_train, pred_train))
            mae_val.append(metrics.mean_absolute_error(y_test, pred_test))
    
    

    print("MLP ", mod, " rmse train: ", np.round(np.mean(rmse_train), 4), np.round(np.std(rmse_train)*2, 2))
    print("MLP ", mod, " rmse val: ", np.round(np.mean(rmse_val), 4), np.round(np.std(rmse_val)*2, 2))
    print("MLP ", mod, " mae train: ", np.round(np.mean(mae_train), 4), np.round(np.std(mae_train)*2, 2))
    print("MLP ", mod, " mae val: ", np.round(np.mean(mae_val), 4), np.round(np.std(mae_val)*2, 2))
    print("MLP ", mod, " number of parameters: ", pytorch_total_params)
    
    
#%% Varying size of the source domain 
mods = [5,7,10,12,13,14]

mae_vals_means = []
mae_vals_sds = []

for model in mods:
    for i in [30, 50, 70, 100]:
        #hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", model, None, data_dir)
        y_preds = np.load(os.path.join(data_dir, f"python\outputs\models\mlp{model}\\nodropout\\sims_frac{i}\y_preds.npy")).squeeze(2)
        y_tests = np.load(os.path.join(data_dir, f"python\outputs\models\mlp{model}\\nodropout\\sims_frac{i}\y_tests.npy")).squeeze(2)
        errors = []
        for j in range(5):
                errors.append(metrics.mean_absolute_error(y_tests[j,:], y_preds[j,:]))
        mae_vals_means.append(np.mean(errors))
        mae_vals_sds.append(np.std(errors)*2)
        
        print("MLP", model)
        print("Percentage of source domain used: ", i)
        print("Mean: ", np.mean(errors), ". 2*Sd:", np.std(errors)*2)
        
#%% MLP 5: Freeze one or two layers.
maes_train = []
maes_test = []
maes_train_m = []
maes_test_m = []
maes_train_s = []
maes_test_s = []
froz = []
froz_m = []
frac_m = []
frac = []
sourcedomain = ["", "", "",  ""]
labels = ["Re-train H2", "Re-train H1+H2"]

for f in range(2):
    
    frozen = f+1
    z= 0
    for i in [100]:
    
        #hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", model, None, data_dir)
        #y_preds = np.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\nodropout\\dummies\\sims_frac{i}\\tuned\setting1\\freeze{frozen}\y_preds.npy")).squeeze(2)
        test_errors = []
        train_errors = []
        
        hparams, model_design, X_train, Y_train, X_test, Y_test = utils.settings(5)
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        Y_test = torch.tensor(Y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        Y_train = torch.tensor(Y_train).type(dtype=torch.float)
                
        for j in range(5):
            
                model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])

                model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp5\\nodropout\sims_frac{i}\\tuned\\setting1\\freeze{frozen}\model{j}.pth")))
                
                pred_test = model(X_test)#
                pred_train = model(X_train)
                
                with torch.no_grad():
                    maes_test.append(metrics.mean_absolute_error(Y_test, pred_test))
                    maes_train.append(metrics.mean_absolute_error(Y_train, pred_train))
                    test_errors.append(metrics.mean_absolute_error(Y_test, pred_test))
                    train_errors.append(metrics.mean_absolute_error(Y_train, pred_train))
                    
                frac.append(sourcedomain[z])
                froz.append(labels[f])
        
        print("Percentage of source domain used: ", i)
        print("Layers frozen: ", f)
        print("Mean test: ", np.mean(test_errors), ". 2*Sd test:", np.std(test_errors)*2) 
        print("Mean train: ", np.mean(train_errors), ". 2*Sd train:", np.std(train_errors)*2) 
        
        maes_train_m.append(np.mean(train_errors))
        maes_test_m.append(np.mean(test_errors))
        maes_train_s.append(np.std(train_errors)*2)
        maes_test_s.append(np.std(test_errors)*2)
        froz_m.append(labels[f])
        frac_m.append(sourcedomain[z])
        
        z = z+1

df = pd.DataFrame(list(zip(maes_test, froz)),
                  columns=["maes_test", "frozen"]) 
df["maes_train"] = maes_train
df["frac"] = frac

df2 = pd.DataFrame(list(zip(froz_m, frac_m)),
                  columns=[ "frozen", "frac"]) 
df2["maes_train_mean"] = maes_train_m
df2["maes_test_mean"] = maes_test_m
df2["maes_train_std"] = maes_train_s
df2["maes_test_std"] = maes_test_s

#%%

for child in model.children():
    for name, parameter in child.named_parameters():
        print(name)
        print(parameter.numel())
        
    

#%%
plt.figure(num=None, figsize=(7,7), facecolor='w', edgecolor='k')
mypal = {"Re-train H2":"darkblue", "Re-train H1+H2":"green"}
bplot = seaborn.boxplot(y="maes_test",
                x = "frac",
                hue = "frozen",
                #hue_order = ["$\mathcal{D}_{T}$", "$\mathcal{D}_{S,7}$", "$\mathcal{D}_{S,12}$"],
                palette = mypal,
                data = df,
                width=0.6,
                showmeans=True,
                orient = "v",
                meanprops={"marker":"o",
                           "markerfacecolor":"black", 
                           "markeredgecolor":"black",
                           "markersize":"12"}
                )

cols= ["darkblue", "green"]
for i in range(len(cols)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(cols[i])
        mybox.set_edgecolor("black")
                            
plt.ylim(bottom=0.4, top = 1.8)
plt.ylabel("Mean Absolute Error [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
plt.xlabel("A3 - AP: partial re-training", size = 20, family='Palatino Linotype')
    #bplot.set_xticklabels(labs)
plt.xticks(size=20, family='Palatino Linotype')
plt.yticks(size=20, family='Palatino Linotype')
plt.legend(loc="upper right", prop = {'size':20, 'family':'Palatino Linotype'})


#%% Part I: Reduced Amount of data
mae_vals = []
mae_trains = []

for i in [30, 40, 50, 60, 70, 75, 80, 85, 90, 95]:
    
    preds = np.load(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\y_preds.npy"))
    tests = np.load(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\y_tests.npy"))
    res = pd.read_csv(os.path.join(data_dir, f"outputs\models\mlp0\adaptive_pooling\architecture3\nodropout\sigmoid\data{i}perc\selected_results.csv"))
    
    mae_vals.append(res["mae_val"].item())
    mae_trains.append(res["mae_train"].item())

import matplotlib.pyplot as plt
plt.plot(mae_vals)

#%% Part II: Weight Analysis of MLP 0.

res = pd.read_csv(os.path.join(data_dir, r"outputs\models\mlp0\noPool\relu\selected_results.csv"))
dimensions = [7]
for hdim in literal_eval(res["hiddensize"].item()):
    dimensions.append(hdim)
dimensions.append(1)

weights1 = []
weights2 = []
for i in range(5):
    model = models.MLP(dimensions, nn.ReLU)
    model.load_state_dict(torch.load(os.path.join(data_dir, f"outputs\models\mlp0\\noPool\\relu\model{i}.pth")))

    hidden0 = model[0].weight.detach().numpy()
    hidden1 = model[2].weight.detach().numpy()

    weight_sums1 = []
    weight_sums2 = []

    for i in range(7):
        weight_sums1.append(np.sum(hidden0[:,i]))
        weight_sums2.append(np.sum(hidden0[:,i] + np.transpose(hidden1)[:,0]) )

    weights1.append(weight_sums1)
    weights2.append(weight_sums2)

#%%
weights2m = np.mean(np.array(weights2),0)
weights2sd = np.std(np.array(weights2),0)

weights2error_upper = weights2m + 2*weights2sd
weights2error_lower = weights2m - 2*weights2sd
variables = ("PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos")

plt.errorbar(variables, y = weights2m, yerr = weights2sd, fmt='', marker = 'o')


#%% Part III: train models on Boreal forest data.
# A) new training
X, Y = preprocessing.get_borealsites(year = "train")
X_test, Y_test = preprocessing.get_borealsites(year = "test")

hparams = {"epochs":500,
           "batchsize":int(res["batchsize"]),
           "history":int(res["history"]),
           "learningrate":res["learningrate"].item()}
model_design = {"dimensions":dimensions,
                "activation":nn.ReLU,
                "featuresize":None}

running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, 
                                                                       {"X_test":X_test, "Y_test":Y_test}, 
                                                                       0.0, data_dir, False)

visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], legend=True)

#%% B) predict with fitted model (architecture 2)
prediction_errors = collect_results.borealsites_predictions()

#%% C) PCA and GLM for two years of Borealsites and two years of Profound
X_bor, Y_bor = preprocessing.get_borealsites(year = "both")
X_prof, Y_prof = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003,2004, 2005,2006],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

from sklearn.decomposition import PCA

pca_bor = PCA(n_components = 7)
pca_bor.fit(X_bor)
print(pca_bor.explained_variance_ratio_)

pca_prof = PCA(n_components = 7)
pca_prof.fit(X_prof)
print(pca_prof.explained_variance_ratio_)

# colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos"]

import statsmodels.api as sm

X_prof = sm.add_constant(X_prof) # Add intercept.
glm_prof = sm.GLM(Y_prof, X_prof) 
results = glm_prof.fit()
print(results.pvalues)         

X_bor = sm.add_constant(X_bor) # Add intercept.
glm_bor = sm.GLM(Y_bor, X_bor) 
results = glm_bor.fit()
print(results.pvalues)  

#%% Correlation of source and target domain.
from scipy.stats.stats import pearsonr   
X_prof, Y_prof = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003,2004,2005,2006,2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None,
                                standardized = False)
X_sims, Y_sims = preprocessing.get_simulations(data_dir = os.path.join(data_dir, r"data\simulations\uniform_params"), 
                                               drop_parameters=True,
                                               standardized = False)
idx = np.random.randint(X_sims.shape[0], size=X_prof.shape[0])
X_sims = X_sims[idx]

for i in range(7):
    print(pearsonr(X_sims[:,i],X_prof[:,i]))
