# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:35 2020

@author: marie
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import models
import preprocessing

#%% Load data
X, Y = preprocessing.get_data()

#x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
#y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

#%% Normalize features
X = preprocessing.normalize_features(X)

#%% Set up Training
# Layer dimensions and model
D_in, D_out, N, H = 12, 2, 730, 25
model = models.ConvNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

#%% Restructure data to smaller batches
window_size = 20
X, Y = preprocessing.to_batches(X,Y, size = window_size)

X_train, Y_train, X_test, Y_test = preprocessing.split_data(X,Y, size=0.5)

epochs = 1
simulations = len(X_train) # number of PREles simulations
batches = X_train[0].shape[2] # batches of smaller time series, determined by window_size

# log trainings loss
training_loss = np.zeros((batches, simulations, epochs))

#%% Training

for epoch in range(epochs):
    
    running_loss = 0.0
    
    for j in range(simulations):
        
        inputs, labels = X_train[j], Y_train[j]
        
        #inputs = np.dstack([inputs[(i-window_size):i] for i in range(window_size, inputs.shape[0]-1)])
        #labels = np.dstack([labels[i+1] for i in range(window_size, len_dat-1)])
        
        for i in range(batches):
            
            x = torch.tensor(np.transpose(inputs[:,:,i])).type(dtype=torch.float).unsqueeze(0)
            y = torch.tensor(np.transpose(labels[:,:,i])).type(dtype=torch.float)
        
            optimizer.zero_grad()
        
            outputs = model(x)
            loss = criterion(outputs, y)
        
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            training_loss[i, j, epoch] = loss.item()
            
            print(running_loss)
        

#%% Testing the model against data

# log validation loss
validation_loss = np.zeros((batches, simulations))

with torch.no_grad():
    
        for j in range(simulations):
        
            inputs, labels = X_test[j], Y_test[j]
        
            for i in range(batches):
                    
                x = torch.tensor(np.transpose(inputs[:,:,i])).type(dtype=torch.float).unsqueeze(0)
                y = torch.tensor(np.transpose(labels[:,:,i])).type(dtype=torch.float)
        
                outputs = model(x)
                
                loss = criterion(outputs, y)
                validation_loss[i, j] = loss.item()
                
 # Accuracy:
# Mean Absolute Error
# Mean Squared Error
# r²

#%% Plot trainings- and validation loss.

plt.plot(training_loss[:,0,0])
plt.plot(validation_loss[:,0])        
        