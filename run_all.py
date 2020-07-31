# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:39:25 2020

@author: marie
"""
#%% Set working directory
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing
import random_forest
from experiment import train_model_CV
from experiment import plot_nn_loss
from experiment import plot_nn_predictions

import random 

X, Y = preprocessing.get_splits(sites = ["hyytiala"], dataset = "profound")

random_forest.random_forest_CV(X, Y, splits=5, shuffled = False)
random_forest.random_forest_CV(X, Y, splits=5, shuffled = True)

fitted_rf = random_forest.random_forest_fit(X, Y, data = "profound")

#%% Network training
hparams = {"hiddensize":256, 
           "batchsize": 500, 
           "epochs":1000, 
           "trainingruns":1,
           "history":2, 
           "optimizer":"adam", 
           "criterion":"mse", 
           "learningrate":3e-2,
           "shuffled_CV":False}

train_loss, val_loss, predictions = train_model_CV(hparams, X, Y, splits=6)

plot_nn_loss(train_loss["rmse"], val_loss["rmse"], data="profound", hparams = hparams)
plot_nn_predictions(predictions, data = "Hyytiala profound")


#%% Grid search of hparams
searchsize = 20

hiddensize = [16, 64, 128, 256]
batchsize = [2, 8, 64, 128, 256]
learningrate = [7e-3, 1e-2, 3e-2, 8e-2]
hp_list = [hiddensize, batchsize, learningrate]

for i in range(searchsize):
    search = [random.choice(sublist) for sublist in hp_list]
