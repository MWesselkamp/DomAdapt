# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:48:40 2020

@author: marie
"""
import torch
import torch.nn as nn
import numpy as np

import setup.utils as utils
#%%

def MLP(dimensions, activation):
    
    network = nn.Sequential()
    
    for i in range(len(dimensions)-1):
        
        network.add_module(f'hidden{i}', nn.Linear(dimensions[i], dimensions[i+1]))
    
        if i < len(dimensions)-2:
            
            network.add_module(f'activation{i}', activation())
    
    return network


#%%
class MLPmod(nn.Module):
    
    def __init__(self, featuresize, dimensions, activation, dropout_prob = 0.0, dropout = False):
        
        super(MLPmod, self).__init__()
        self.featuresize = featuresize
        self.hidden_features = dimensions[1]
        self.activation = activation()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        
        self.encoder = nn.Linear(1, self.hidden_features)
        self.layers = nn.ModuleList()
        for i in range(dimensions[0]):
            self.layers.append(nn.Linear(1, self.hidden_features))
        self.avgpool = nn.AdaptiveMaxPool1d(self.featuresize)
        self.classifier = self.mlp(dimensions, activation)
        
    def forward(self, x):
        
        latent = torch.empty(size=(x.shape[0], self.hidden_features, x.shape[1]))
        for i in range(x.shape[1]):
            layer = self.layers[i]
            encoded_feature = layer(x[:,i].unsqueeze(1))
            if self.dropout:
                if i > 7:
                    drop = np.random.binomial(1, self.dropout_prob)
                    if drop == 1:
                        encoded_feature[encoded_feature!=0] = 0
                    
            latent[:,:,i] = encoded_feature
            
        out = self.activation(latent)
        out = self.avgpool(out).view(x.shape[0],-1)
        #out = self.activation(out)
        out = self.classifier(out)
        
        return(out)

    def mlp(self, dimensions, activation):
    
        network = nn.Sequential()
        network.add_module(f"hidden0", nn.Linear(self.featuresize*self.hidden_features, dimensions[1]))
        network.add_module(f'activation0', activation())
        for i in range(len(dimensions)-2):
            network.add_module(f'hidden{i+1}', nn.Linear(dimensions[i+1], dimensions[i+2]))
            if i < len(dimensions)-3:
                network.add_module(f'activation{i+1}', activation())
    
        return(network)
    
    def encode(self, x):
        
        latent = []
        
        for i in range(x.shape[1]):
            encoded_feature = self.encoder(x.unsqueeze(1)[:,:,i]).unsqueeze(2)
            if i > 7:
                drop = np.random.binomial(1, self.dropout_prob)
                if drop == 1:
                    encoded_feature[encoded_feature!=0] = 0
                    
            latent.append(encoded_feature)
                    
        
        latent = torch.stack(latent, dim=2).squeeze(3)

        return(latent)
            
        
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
    network.add_module("max_pool1", nn.MaxPool1d(kernel_size=2, stride=1))
    
    if len(dim_channels) == 2:
        network.add_module("conv2", nn.Conv1d(in_channels = dim_channels[0], out_channels = dim_channels[1], kernel_size = kernel_size))
        network.add_module("activation2", activation())
        network.add_module("max_pool3", nn.MaxPool1d(kernel_size=2, stride=1))

    network.add_module("flatten", Flatten())
    network.add_module("fc1", nn.Linear(linear_in, dimensions[1]))
    network.add_module("activation4", activation())
    network.add_module("fc2", nn.Linear(dimensions[1], dimensions[1]))
    network.add_module("activation5", activation())
    network.add_module("f3",nn.Linear(dimensions[1], dimensions[2]))
    
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
