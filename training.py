# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:35 2020

@author: marie
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import models
import preprocessing

#%% Load data
X, Y = preprocessing.get_data()

#x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
#y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

#%% Normalize features
X = preprocessing.normalize_features(X)


#%% Set up Training

D_in, D_out, N, H = 12, 2, 730, 25

model = models.ConvNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

window_size=20

#%% Restructure data to smaller batches
X, Y = preprocessing.to_batches(X,Y, size= window_size)

#%% Training
counter = 0

for epoch in range(2):
    
    running_loss = 0.0
    
    for i in range(len(X)):
        
        inputs, labels = X[i], Y[i]
        len_dat = inputs.shape[0]
        
        #inputs = np.dstack([inputs[(i-window_size):i] for i in range(window_size, inputs.shape[0]-1)])
        #labels = np.dstack([labels[i+1] for i in range(window_size, len_dat-1)])
        
        for i in range(0,inputs.shape[2]):
            
            counter += 1
            
            x = torch.tensor(np.transpose(inputs[:,:,i])).type(dtype=torch.float).unsqueeze(0)
            y = torch.tensor(np.transpose(labels[:,:,i])).type(dtype=torch.float)
        
            optimizer.zero_grad()
        
            outputs = model(x)
            loss = criterion(outputs, y)
        
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            print(loss.item())
        

#%% Testing the model against data
       
        
 # Accuracy:

# Mean Absolute Error

# Mean Squared Error

# r²
   
        
        