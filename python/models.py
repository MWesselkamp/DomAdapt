# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:48:40 2020

@author: marie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
#%%

def MLP(dimensions, activation):
    
    network = nn.Sequential()
    
    for i in range(len(dimensions)-1):
        
        network.add_module(f'hidden{i}', nn.Linear(dimensions[i], dimensions[i+1]))
    
        if i < len(dimensions)-2:
            
            network.add_module(f'activation{i}', activation())
    
    return network

#%%
class Flatten(nn.Module):
    
    def forward(self, x):
        # Reshape to Batchsíze
        return(x.view(x.shape[0], -1))
    
#%%
def ConvN(dimensions, dim_channels, kernel_size, length, activation = nn.ReLU):
    
    linear_in = utils.num_infeatures(dim_channels, kernel_size, length)
    
    network = nn.Sequential()
    
    network.add_module("conv1", nn.Conv1d(in_channels = dimensions[0], out_channels = dim_channels[0], kernel_size = kernel_size))
    network.add_module("activation1", activation())
    network.add_module("conv2", nn.Conv1d(in_channels = dim_channels[0], out_channels = dim_channels[1], kernel_size = kernel_size))
    network.add_module("activation2", activation())
    network.add_module("flatten", Flatten())
    network.add_module("fc1", nn.Linear(linear_in, dimensions[1]))
    network.add_module("activation3", activation())
    network.add_module("fc2",nn.Linear(dimensions[1], dimensions[2]))
    
    return network


#%% LSTM
class LSTM(nn.Module):
    
    def __init__(self, D_in, n_hidden, D_out, seq_len, activation, n_layers = 1):
        super().__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(D_in, self.n_hidden, batch_first = False)

        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, D_out)
        self.activation = activation
        
    def init_hidden(self, batchsize):
        
        return (torch.zeros(self.n_layers,batchsize, self.n_hidden),
                            torch.zeros(self.n_layers,batchsize, self.n_hidden))

    def forward(self, x):
        
        hidden_cell = self.init_hidden(x.shape[1])
        
        out, hidden_cell = self.lstm(x, hidden_cell)
        out = self.activation(out)
        out = self.fc1(out[-1,:,:])
        out = self.activation(out)
        out = self.fc2(out)
        
        return out 


#%% 1d-Conv-Net

class ConvNet(nn.Module):
    def __init__(self, D_in, H, D_out): 
        super(ConvNet, self).__init__()
        
        # Define layers as class attributes
        #       kernel size = filter size
        #       output channels = number of filter
        #       input channels 
        
        self.conv1 = nn.Conv1d(in_channels = D_in, out_channels = 12, kernel_size = 2)
        self.conv2 = nn.Conv1d(in_channels = 12, out_channels = 18, kernel_size = 2)
        
        # fc for fully connected layer.
        # Flatten the tensor when coming from convolutional to linear layer
        self.fc1 = nn.Linear(in_features = 324, out_features = H)
        self.fc2 = nn.Linear(in_features = H, out_features = D_out)
    
    def forward(self, x):
        out = self.conv1(x) # layer operation
        out = F.relu(out) # transformation operation
        out = self.conv2(out)
        out = F.relu(out)
        
        # flatten tensor before passing to linear layer,
        out = out.view(out.shape[0], -1)
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
