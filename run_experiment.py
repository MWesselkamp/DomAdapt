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
import utils
   
import matplotlib.pyplot as plt
import numpy as np

#%%
def plot_nn_loss(train_loss, val_loss, data, hparams):
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    fig.suptitle(f"Fully connected Network: {data} data \n Batches = {hparams['batches']}, Epochs = {hparams['epochs']}, History = {hparams['history']} \n Hiddensize = {hparams['hiddensize']}, Learning_rate = {hparams['learningrate']}")

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        ax.fill_between(np.arange(hparams["batches"]), ci_train[0],ci_train[1], color="lightgreen", alpha=0.3)
        ax.fill_between(np.arange(hparams["batches"]), ci_val[0],ci_val[1], color="lightblue", alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    ax.plot(train_loss, color="green", label="Training loss", linewidth=0.8)
    ax.plot(val_loss, color="blue", label = "Validation loss", linewidth=0.8)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set(xlabel="Batches", ylabel="Root Mean Squared Error")
    fig.legend(loc="upper left")
    #plt.savefig(f"plots\data_quality_evaluation\{data}_loss")
    #plt.close()

#%% Training
def run_experiment(hparams, X, Y, data, minibatches = False):

    hiddensize = hparams["hiddensize"]
    epochs = hparams["epochs"]
    batches= hparams["batches"]
    
    X = utils.minmax_scaler(X)
    
    train_loss = np.zeros((epochs, batches))
    val_loss = np.zeros((epochs, batches))

    for epoch in range(epochs):
        
        model = models.LinNet(D_in = X.shape[1], H = hiddensize, D_out = 1)
        
        training, validation = experiment.train_model(model, hparams, X, Y, minibatches = minibatches)
        train_loss[epoch,:] = np.array(training["rmse"])
        val_loss[epoch,:] = np.array(validation["rmse"])

    #plot_nn_loss(train_loss, val_loss, data = data, hparams = hparams)
    
    return(train_loss, val_loss)#train_loss, val_loss)
#%% Load data
    
X_profound, Y_profound = preprocessing.get_profound_data(dataset="trainval", data_dir = r'data\profound', to_numpy = True, simulation=False)
X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', to_numpy = True, preles=False)

# Merge profound and preles data into one large data set.
X_both = np.concatenate((X_profound, X_borealsites), axis=0)
Y_both = np.concatenate((Y_profound, Y_borealsites), axis=0)

#%% vary learning rate and hidden sizes
#learningrates = [2e-3, 5e-3, 9e-3, 1e-2]
#plots = [1, 2, 3, 4]
#hiddensizes = [100,256]

#for lr, plot in zip(learningrates, plots):
    
#    for hs in hiddensizes:
    
hparams = {"hiddensize":100, 
                   "batchsize": 1000, 
                   "batches":300, 
                   "epochs":1,
                   "history":0, 
                   "optimizer":"adam", 
                   "criterion":"mse", 
                   "learningrate":2e-2}

train_loss, val_loss = run_experiment(hparams, X = X_both, Y=Y_both, data="Borealsites", minibatches=False)

plot_nn_loss(train_loss, val_loss , data=f"Borealsites", hparams=hparams)
