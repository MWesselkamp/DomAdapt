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

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
#%% Load data

X_profound, Y_profound = preprocessing.get_profound_data(data_dir = r'data\profound', ignore_env = True, preles=False)
X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', ignore_env = True, preles=False)

# Merge profound and preles data into one large data set.
X = np.concatenate((X_profound, X_borealsites), axis=0)
Y = np.concatenate((Y_profound, Y_borealsites), axis=0)

X = preprocessing.normalize_features(X)


#%% Set up Training
# Layer dimensions and model
D_in, D_out, N, H = 7, 1, 200, 100
model = models.LinNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
epochs = 3000
history = 3

#%% Split data into test and training samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

#%% Training
# log trainings loss
training_loss = np.zeros((epochs,1))
validation_loss = np.zeros((epochs,1))

X_test = torch.tensor(X_test).type(dtype=torch.float)
#y_test = torch.tensor(y_test).type(dtype=torch.float)
yrange = np.ptp(y_test, axis=0)


#%%
## TensorBoard setup

# Everything that should be displayed is encapsulated as a tf.summary object
# Define the SummaryWriter, the key object for writing information to tensorboard
print("Setting up tensorboard")

# Sets up a timestamped log directory. By doing so, tensorboard treats each log as an individual run.
logdir = r'C:\Users\marie\OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\tensorboard_eventfiles\linnet' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

#%% Training

for i in range(epochs):
        
    # Training
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
        training_loss[i:] = utils.percentage_error(y.detach().numpy().astype("float64"), output.detach().numpy().astype("float64") , y_range = yrange)
        
        loss.backward()
        optimizer.step()
    
    # Validation
        preds = model(X_test)
        validation_loss[i:] = utils.percentage_error(y_test, preds.detach().numpy().astype("float64") , y_range = yrange)

    # Writing to tensorboard
       # writer.add_scalars("Train", {"train":training_loss[i,:], "val":validation_loss[i,:]}, i)
       # writer.flush()


#%% Save the model
#PATH = './Pydata/simple_lin_net.pth'
#torch.save(model.state_dict(), PATH)

#%% Testing the model against data

#%% Plot trainings- and validation loss.
%matplotlib qt5

plt.plot(training_loss[:,0], label="training loss")
plt.plot(validation_loss[:,0], label="validation loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Percentage Error")
plt.suptitle(f"3-layer fully connected network / history {history} days\n Full sample size = {X.shape[0]} \n Batch size = {N*(history+1)}; Hidden size = 100 \n")
#%% Plot predictions


