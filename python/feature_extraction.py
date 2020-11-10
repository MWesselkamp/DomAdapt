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
import setup.utils as utils
import finetuning
import visualizations

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
def featureExtractorA(X, Y, splits=5):
    
    X = torch.tensor(X).type(dtype=torch.float)
    
    predictions = []
    mae = []
    rmse = []

    for i in range(splits):
        
        model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp7\\nodropout\model{i}.pth")))
        
        preds = model(X).detach().numpy()

        mae.append(metrics.mean_absolute_error(Y, preds))
        rmse.append(utils.rmse(Y, preds))
        predictions.append(preds)
    
    errors = {"rmse_val":rmse, "mae_val":mae}

    return predictions, errors

#%%
predictions, errors = featureExtractorA(X_test, Y_test)

np.mean(errors["mae_val"])
np.mean(errors["rmse_val"])

predictions[0].shape

#%% Finetune network on finish data, Full Backprob.
running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 3000, "mlp", 7, "nodropout", featuresize, 
                                                                   True, None, {"X_test":X_test, "Y_test":Y_test})

#%%
errors = np.mean(np.array(performance), axis=0)

visualizations.plot_prediction(Y_test, y_preds, "Finetuning")

#%% finetune network on finish data: Freeze all but last two layers.
running_losses,performance, y_tests, y_preds = finetuning.finetune(X, Y, 50, "mlp", 7, "nodropout", featuresize,
                                                                   False, ["hidden2.weight", "hidden3.weight"],{"X_test":X_test, "Y_test":Y_test})


errors = np.mean(np.array(performance), axis=0)

#%% 1) Ordinary Least Squares
def featureExtractorB(X, Y, X_test, Y_test, classifier = "ols", splits = 5):
    
    predictions_train = []
    predictions_test = []
    
    for i in range(splits):
    
        model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp7\\nodropout\model{i}.pth")))
      
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2]) # Remove Final layer and activation.

        out_train = model(X).detach().numpy()
        out_train = sm.add_constant(out_train) # Add intercept.
        out_test = model(X_test).detach().numpy()
        out_test = sm.add_constant(out_test) # Add intercept.

        if classifier == "ols":
            extractor = sm.OLS(Y, out_train) 
        elif classifier == "glm":
            print("Fitting glm with Inverse Gaussian family and log-Link.")
            extractor = sm.GLM(Y, out_train, family=sm.families.InverseGaussian(sm.families.links.log())) 
        else:
            print("Don't know classifier.")
            
        results = extractor.fit()
        
        predictions_train.append(np.expand_dims(results.predict(), axis=1))
        predictions_test.append(np.expand_dims(results.predict(out_test), axis=1))
        
    mae_train = [metrics.mean_absolute_error(Y, sublist) for sublist in predictions_train]
    mae_val = [metrics.mean_absolute_error(Y_test, sublist) for sublist in predictions_test]
    rmse_train = [utils.rmse(Y, sublist) for sublist in predictions_train]
    rmse_val = [utils.rmse(Y_test, sublist) for sublist in predictions_test]
    
    errors = [rmse_train, rmse_val, mae_train, mae_val]
    
    return predictions_test, errors

#%% 1)
predictions_ols, errors = featureExtractorB(X, Y, X_test, Y_test, "ols")

errors = np.mean(np.array(errors), axis=1)

visualizations.plot_prediction(Y_test, predictions_ols, "OLS")

#%% 2)
predictions_glm, errors = featureExtractorB(X, Y, X_test, Y_test, "glm")

errors = np.mean(np.array(errors), axis=1)

visualizations.plot_prediction(Y_test, predictions_glm, "GLM with Inverse Gaussian Family")

#%% 3) Non-negative Least Squares
from scipy.optimize import nnls

theta = np.expand_dims(nnls(out_train, Y[:,0])[0], axis=1)

predictions = np.dot(out_train, theta)

plt.plot(predictions)
plt.plot(Y)

metrics.mean_absolute_error(Y, predictions)


#%% 0) Replace last layer by Layer with selected number of hidden nodes

model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp7\\nodropout\model{i}.pth")))
      
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier.add_module("hidden4", nn.Linear(16, 256))
model.classifier.add_module("activation4", nn.ReLU())
model.classifier.add_module("fc_out", nn.Linear(256, 1))