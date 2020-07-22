# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:12:20 2020

@author: marie
"""
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import experiment
import preprocessing
import models

X, Y= preprocessing.get_profound_data(dataset="trainval", data_dir = r'data\profound', to_numpy = True, simulation=False)
X = preprocessing.normalize_features(X)

hparams = {"batchsize": 150, "epochs":200, "history":1, "optimizer":"adam", "criterion":"mse"}

model = models.LinNet(D_in = 7, H = 200, D_out = 1)

train_loss, val_loss = experiment.train_model(model, hparams, X, Y)

import matplotlib.pyplot as plt

plt.plot(train_loss["mae"])
plt.plot(val_loss["mae"])
