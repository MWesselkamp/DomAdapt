# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:35 2020

@author: marie

This file can be used to create and display event files for tensorboard.

To run file with tensorboard:
    
1. Open Anaconda Prompt
2. Activate tensorenv
3. run: python [directory]/[file]: python OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\[file]
4. after sucessful execution, run: tensorboard --logdir OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\tensorboard_eventfiles

If used with tensorboard, comment out the following to change working directory:
"""
#%% Set working directory
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

import models
import preprocessing
import utils

from sklearn.model_selection import train_test_split
from sklearn import metrics

#%% Load data
X, Y= preprocessing.get_profound_data(dataset="trainval", data_dir = r'data\profound', to_numpy = True, simulation=False)
X = preprocessing.normalize_features(X)

#%% Set up Training
# Layer dimensions and model
D_in, D_out, N, H = 6, 1, 150, 200
model = models.LinNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
epochs = 200
history = 1

#%% Split data into test and training samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# log trainings loss
training_loss_rmse = np.zeros((epochs,1))
validation_loss_rmse = np.zeros((epochs,1))
training_loss_mae = np.zeros((epochs,1))
validation_loss_mae = np.zeros((epochs,1))

X_test = torch.tensor(X_test).type(dtype=torch.float)
#y_test = torch.tensor(y_test).type(dtype=torch.float)
yrange = np.ptp(y_test, axis=0)


#%%
tensorboard = False

## TensorBoard setup
if (tensorboard):
    
    import tensorflow as tf
    from torch.utils.tensorboard import SummaryWriter
    # Everything that should be displayed is encapsulated as a tf.summary object
    # Define the SummaryWriter, the key object for writing information to tensorboard
    print("Setting up tensorboard")

    # Sets up a timestamped log directory. By doing so, tensorboard treats each log as an individual run.
    logdir = r'C:\Users\marie\OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\tensorboard_eventfiles\linnet' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)

#%% Training

for i in range(epochs):
        
    # Training
    # Subset data set to small batch in each epoch.
    # If hisotry > 0, batches are extended by the specified lags.
    
        subset = [i for i in random.sample(range(X_train.shape[0]), N) if i > history]
        subset_h = [item for sublist in [list(range(i-history,i)) for i in subset] for item in sublist]
        
        X_batch = np.concatenate((X_train[subset], X_train[subset_h]), axis=0)
        y_batch = np.concatenate((y_train[subset], y_train[subset_h]), axis=0)
        
        x = torch.tensor(X_batch).type(dtype=torch.float)
        y = torch.tensor(y_batch).type(dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(x)
        
        loss = criterion(output, y)
        #training_loss[i,:] = loss.item()
        training_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64")))
        training_loss_mae[i:] = metrics.mean_absolute_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64"))
        
        loss.backward()
        optimizer.step()
    
    # Evaluate current model
        preds = model(X_test)
        validation_loss_rmse[i:] = np.sqrt(metrics.mean_squared_error(y_test, preds.detach().numpy().astype("float64")))
        validation_loss_mae[i:] = metrics.mean_absolute_error(y_test, preds.detach().numpy().astype("float64"))
    # Writing to tensorboard
       # writer.add_scalars("Train", {"train":training_loss[i,:], "val":validation_loss[i,:]}, i)
       # writer.flush()


#%% Save the model
#PATH = './Pydata/simple_lin_net.pth'
#torch.save(model.state_dict(), PATH)

#%% Testing the model against data

#%% Plot trainings- and validation loss: RSME
%matplotlib qt5

plt.plot(training_loss_rmse[:,0], label="training loss")
plt.plot(validation_loss_rmse[:,0], label="validation loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error")
plt.suptitle(f"3-layer fully connected network / history {history} days\n Full sample size = {X.shape[0]} \n Batch size = {N*(history+1)}; Hidden size = {H} \n")

#%% Plot trainings- and validation loss: MAE
plt.plot(training_loss_mae[:,0], label="training loss")
plt.plot(validation_loss_mae[:,0], label="validation loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.suptitle(f"3-layer fully connected network / history {history} days\n Full sample size = {X.shape[0]} \n Batch size = {N*(history+1)}; Hidden size = {H} \n")

#%% Plot model predictions to boreal sites data.

X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', to_numpy = True, preles=False)
X = torch.tensor(preprocessing.normalize_features(X_borealsites)).type(dtype=torch.float)
predictions = model(X)
result = predictions.data.numpy()

plt.plot(result, label = "predictions")
plt.plot(Y_borealsites, label = "observations")
plt.plot(Y_borealsites-result, label="error")
plt.suptitle(f"Network predictions / history {history} days")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Growth Primary Production")