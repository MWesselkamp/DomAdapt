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

import models
import preprocessing

#%% Load data
X, Y = preprocessing.get_profound_data(data_dir = 'data\profound', ignore_env = True)

#x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
#y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

#%% Normalize features
X = preprocessing.normalize_features(X)

#%% Set up Training
# Layer dimensions and model
D_in, D_out, N, H = 7, 1, 300, 100
model = models.LinNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

#%% Split data into training and test



# log trainings loss
training_loss = np.zeros((batches, simulations, epochs))

#%% Training


#%% Save the model
PATH = './Pydata/simple_lin_net.pth'
torch.save(model.state_dict(), PATH)

#%% Testing the model against data



#%% Plot trainings- and validation loss.
%matplotlib qt5

#%% Plot predictions


