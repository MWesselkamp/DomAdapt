# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:48:40 2020

@author: marie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


#%%

def MLP(dimensions, activation = nn.ReLU):
    
    network = nn.Sequential()
    
    for i in range(len(dimensions)-1):
        
        network.add_module(f'hidden{i}', nn.Linear(dimensions[i], dimensions[i+1]))
    
        if i < len(dimensions)-2:
            
            network.add_module(f'activation{i}', activation())
    
    return network
    
    

#%% 1d-Conv-Net

class ConvNet(nn.Module):
    def __init__(self, D_in, H, D_out): 
        super(ConvNet, self).__init__()
        
        # Define layers as class attributes
        #       kernel size = filter size
        #       output channels = number of filter
        #       input channels 
        
        self.conv1 = nn.Conv1d(in_channels = D_in, out_channels = 24, kernel_size = 4)
        self.conv2 = nn.Conv1d(in_channels = 24, out_channels = 48, kernel_size = 4)
 
        # fc for fully connected layer.
        # Flatten the tensor when coming from convolutional to linear layer
        self.fc1 = nn.Linear(in_features = 96, out_features = H)
        self.fc2 = nn.Linear(in_features = H, out_features = D_out)
    
    def forward(self, x):
        out = self.conv1(x) # layer operation
        out = F.max_pool1d(F.relu(out), kernel_size=2) # transformation operation
        out = self.conv2(out)
        out = F.max_pool1d(F.relu(out), kernel_size=2)
        
        # flatten tensor before passing to linear layer,
        out = out.view(size=(-1,96))
        # Add two dense layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

#%% Fully Connected Linear Net

class LinNet(nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(LinNet, self).__init__()
        
        self.fc1 = nn.Linear(in_features = D_in, out_features = H)
        self.fc2 = nn.Linear(in_features = H, out_features = H)
        self.fc3 = nn.Linear(in_features = H, out_features = D_out)
        
    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))
        out = torch.sigmoid(self.fc2(out))
        out = self.fc3(out)
        
        return out
    
#%% 
