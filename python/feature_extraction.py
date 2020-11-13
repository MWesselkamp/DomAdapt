# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:42:45 2020

@author: marie
"""
import setup.models as models
import pandas as pd
import os.path
import torch.nn as nn
import torch

import numpy as np

import finetuning
import visualizations


#%%
predictions, errors, Y_test = finetuning.featureExtractorA("mlp", 7, None)

np.mean(errors["mae_val"])
np.mean(errors["rmse_val"])


#%% finetune network on finish data: Freeze no layer.
performance, y_preds, Y_test = finetuning.featureExtractorB("mlp", 7, 1500)

visualizations.plot_prediction(Y_test, y_preds, "Finetuning")

#%% finetune network on finish data: Freeze all but last two layers.
# Actually computed on Cluster due to computation time.
#performance, y_preds, Y_test = finetuning.featureExtractorB("mlp", 7, 10, ["hidden2.weight", "hidden3.weight"])
#visualizations.plot_prediction(Y_test, y_preds, "Perceptron")
ypreds = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\tuned\setting1\y_preds.npy")
ytests = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\tuned\setting1\y_tests.npy")
rls = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\tuned\setting1\running_losses.npy", allow_pickle=True).item()

visualizations.plot_prediction(Y_test, ypreds, "Perceptron")

rets = pd.read_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\nodropout\tuned\setting1\selected_results.csv")

visualizations.plot_running_losses(rls["mae_train"], rls["mae_val"], "mlp7")

#%% 1) Ordinary Least Squares as Classifier
predictions_ols, errors = finetuning.featureExtractorC("mlp", None, "ols")

errors = np.mean(np.array(errors), axis=1)

visualizations.plot_prediction(Y_test, predictions_ols, "OLS")

#%% 2) Generalized Linear Model as Classifier
predictions_glm, errors = finetuning.featureExtractorC("mlp", None,  "glm")

errors = np.mean(np.array(errors), axis=1)

visualizations.plot_prediction(Y_test, predictions_glm, "GLM with Inverse Gaussian Family")

#%% 3) Non-negative Least Squares as Classifier
predictions_nnls, errors = finetuning.featureExtractorC("mlp", None,  "nnls")

errors = np.mean(np.array(errors), axis=1)

visualizations.plot_prediction(Y_test, predictions_nnls, "Non-negative least squares")


#%% 0) Replace last layer by Layer with selected number of hidden nodes

model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp7\\nodropout\model{i}.pth")))
      
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier.add_module("hidden4", nn.Linear(16, 256))
model.classifier.add_module("activation4", nn.ReLU())
model.classifier.add_module("fc_out", nn.Linear(256, 1))