# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:35 2020

@author: marie
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

import models
import preprocessing
import utils

from sklearn.model_selection import train_test_split

#%% Load data
X, Y = preprocessing.get_profound_data(data_dir = 'data\profound', ignore_env = True, preles=True)

#x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
#y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

#%% Normalize features
X = preprocessing.normalize_features(X)
Y = Y[:,0].reshape(-1,1) # only GPP

#%% Set up Training
# Layer dimensions and model
D_in, D_out, N, H = 7, 1, 50, 50
model = models.LinNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-1)
epochs = 3000
history = 0

#%% Split data into test and training samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

#%% Training
# log trainings loss
training_loss = np.zeros((epochs,1))
validation_loss = np.zeros((epochs,1))

X_test = torch.tensor(X_test).type(dtype=torch.float)
#y_test = torch.tensor(y_test).type(dtype=torch.float)

yrange = np.ptp(y_test, axis=0)

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

        


#%% Save the model
PATH = './Pydata/simple_lin_net.pth'
torch.save(model.state_dict(), PATH)

#%% Testing the model against data



#%% Plot trainings- and validation loss.
%matplotlib qt5

plt.plot(training_loss[:,0])
plt.plot(validation_loss[:,0])

#%% Plot predictions


