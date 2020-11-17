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
import setup.models as models
import pandas as pd
import os.path
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn import metrics

import numpy as np

import finetuning
import visualizations

import setup.utils as utils
import matplotlib.pyplot as plt

#%%
def summarise_results(types, simsfrac):
    
    domadapt_errors = []
    domadapt_predictions = []

    for typ in types:
        for frac in simsfrac:
    
            predictions, errors, Y_test = finetuning.featureExtractorA("mlp", typ, None, frac)
            domadapt_errors.append(["mlp", typ,frac, "A", None, None, None, np.mean(errors["rmse_val"]),None, np.mean(errors["mae_val"])])
            predictions.append(predictions)
            
            # 1) Ordinary Least Squares as Classifier
            predictions_ols, errors = finetuning.featureExtractorC("mlp",typ, None, frac, "ols")
            errors = np.mean(np.array(errors), axis=1)
            domadapt_errors.append(["mlp", typ,frac, "C", "OLS", None, errors[0], errors[1],errors[2], errors[3]])
            predictions.append(predictions_ols)
            #visualizations.plot_prediction(Y_test, predictions_ols, "OLS")
            
            # 2) Non-negative Least Squares as Classifier
            predictions_nnls, errors = finetuning.featureExtractorC("mlp", typ, None,  frac, "nnls")
            errors = np.mean(np.array(errors), axis=1)
            domadapt_errors.append(["mlp", typ,frac, "C", "NNLS",None, errors[0], errors[1],errors[2], errors[3]])
            predictions.append(predictions_nnls)
            #visualizations.plot_prediction(Y_test, predictions_nnls, "Non-negative least squares")
            
            #3) MLP with architecture 2 as Classifier
            running_losses, errors = finetuning.featureExtractorD("mlp", typ, 3000, frac)
            errors = np.mean(np.array(errors),0)
            domadapt_errors.append(["mlp", typ,frac, "D", "MLP2",None, errors[0], errors[1],errors[2], errors[3]])
            
            ## Feature Extractor B due to computation time only used on cluster! ##
            ## LOADING RESULTS ##
            
            #4) Full Backprob with pretrained weights
            rets_fb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\sims_frac{frac}\\tuned\setting0\selected_results.csv")
            domadapt_errors.append(["mlp", typ, frac, "B", "full_backprop",rets_fb["epochs"],rets_fb["rmse_train"], rets_fb["rmse_val"],rets_fb["mae_train"], rets_fb["mae_val"]])
            
            #5) Backprop only last layer.
            rets_hb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\sims_frac{frac}\\tuned\setting0\selected_results.csv")
            domadapt_errors.append(["mlp", typ, frac, "B", "freezeW2",rets_hb["epochs"],rets_hb["rmse_train"], rets_hb["rmse_val"],rets_hb["mae_train"], rets_hb["mae_val"]])

            
    domadapt_results = pd.DataFrame(domadapt_errors,
                        columns = ["model", "typ", "simsfrac", "featureextractor", "spec", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val"])
    
    domadapt_results.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\featureextraction.csv")
    
    return(domadapt_results)
    
#%%
domadapt_errors = summarise_results(types = [7,8], simsfrac = [30, 50])

