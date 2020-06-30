# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:19:35 2020

@author: marie
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import utils
import models


sims,filenames = utils.get_data()
x = torch.tensor(np.transpose(sims['sim1'][0])).type(dtype=torch.float)
y = torch.tensor(np.transpose(sims['sim1'][1])).type(dtype=torch.float)

x = x.unsqueeze(0)

#%% Training

D_in, D_out, N, H = 12, 2, 730, 25

model = models.ConvNet(D_in, H, D_out)

# loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

for epoch in range(2):
    
    running_loss = 0.0
    
    for filename in filenames:
        
        inputs, labels = sims[filename]
        
        x = torch.tensor(np.transpose(inputs)).type(dtype=torch.float).unsqueeze(0)
        y = torch.tensor(np.transpose(labels)).type(dtype=torch.float)
        
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        print(running_loss)