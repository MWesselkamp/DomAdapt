# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:42:45 2020

@author: marie
"""
import setup.models as models
import pandas as pd
import os.path
from ast import literal_eval
import torch.nn as nn
import torch
import setup.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import statsmodels.api as sm
#%%
data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\mlp\grid_search_results_mlp1.csv"))
    
setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()

dimensions = literal_eval(setup["hiddensize"])
dimensions.append(1) # adds the output dimension!

hparams = {"batchsize": int(setup["batchsize"]), 
           "epochs":1000, 
           "history": int(setup["history"]), 
           "hiddensize":literal_eval(setup["hiddensize"]),
           "learningrate":setup["learningrate"]}

model_design = {"dimensions":dimensions,
                "activation":nn.ReLU}

featuresize = 7 
#%%
model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
model.load_state_dict(torch.load(os.path.join(data_dir, r"python\outputs\models\mlp7\nodropout\model0.pth")))

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

X = torch.tensor(X).type(dtype=torch.float)
X_test = torch.tensor(X_test).type(dtype=torch.float)

#%% Predict with network directly to finnish data
preds = model(X_test).detach().numpy()

metrics.mean_absolute_error(Y_test, preds)

plt.plot(preds)

#%% Finetune network on finish data, Full Backprob.
import finetuning

running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 1000, "mlp", 7, "nodropout", featuresize, True)
model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
model.load_state_dict(torch.load(os.path.join(data_dir, r"python\outputs\models\mlp7\nodropout\tuned\model0.pth")))

preds = model(X_test).detach().numpy()

metrics.mean_absolute_error(Y_test, preds)

plt.plot(preds)

#%% finetune network on finish data: Freeze all but last two layers.

running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 1000, "mlp", 7, "nodropout", featuresize, True, 
                                                                   ["hidden2.weight", "hidden3.weight"])
model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
model.load_state_dict(torch.load(os.path.join(data_dir, r"python\outputs\models\mlp7\nodropout\tuned\setting1\model0.pth")))

preds = model(X_test).detach().numpy()

metrics.mean_absolute_error(Y_test, preds)

plt.plot(preds)

#%% Finetune With Different Classifiers

#%% 0) Replace last layer by Layer with selected number of hidden nodes
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier.add_module("hidden4", nn.Linear(16, 256))
model.classifier.add_module("activation4", nn.ReLU())
model.classifier.add_module("fc_out", nn.Linear(256, 1))

#%%
running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 1000, "mlp", 7, "nodropout", featuresize, False, 
                                                                   ["hidden4.weight"])



#%% 1) Ordinary Least Squares
model.classifier = nn.Sequential(*list(model.classifier.children())[:-2]) # Remove Final layer and activation.

out_train = model(X).detach().numpy()
out_train = sm.add_constant(out_train) # Add intercept.
out_test = model(X_test).detach().numpy()
out_test = sm.add_constant(out_test) # Add intercept.


#%%
ols = sm.OLS(Y, out_train) 
results = ols.fit()
print(results.summary())

predictions = np.expand_dims(results.predict(), axis=1)
metrics.mean_absolute_error(Y, predictions)
plt.plot(predictions)

#%%
preds_test = np.expand_dims(results.predict(out_test), axis=1)
metrics.mean_absolute_error(Y_test, preds_test)
plt.plot(preds_test)
plt.plot(Y_test)

#%% 2) Non-negative Least Squares
from scipy.optimize import nnls

theta = np.expand_dims(nnls(out_train, Y[:,0])[0], axis=1)

predictions = np.dot(out_train, theta)

plt.plot(predictions)
plt.plot(Y)

metrics.mean_absolute_error(Y, predictions)

#%% 3) Generalized Least Squares

gamma_model = sm.GLM(Y, out_train, family=sm.families.InverseGaussian(sm.families.links.log())) 
g_results = gamma_model.fit()
print(g_results.summary())

predictions = g_results.predict()

plt.plot(predictions)
metrics.mean_absolute_error(Y, predictions)
