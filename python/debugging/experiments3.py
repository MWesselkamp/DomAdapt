# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:30:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import numpy as np
import utils
import preprocessing
import matplotlib.pyplot as plt

#%%%
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = None,
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR","DOY_sin", "DOY_cos"],
                                to_numpy = True)

#%% Train

X = utils.minmax_scaler(X)
X = torch.tensor(X).type(dtype=torch.float)
Y = torch.tensor(Y).type(dtype=torch.float)

#model = models.MLP([X.shape[1],12,1], nn.ReLU)
#model = models.LSTM(X.shape[1], 12, 1, 10, F.relu)

#x, target = utils.create_batches(X, Y, 128, 0)
#x_test, target_test = utils.create_batches(X, Y, 128, 0)


#x = torch.tensor(x).type(dtype=torch.float)
#target = torch.tensor(target).type(dtype=torch.float)
#x_test = torch.tensor(x_test).type(dtype=torch.float)
#target_test = torch.tensor(target_test).type(dtype=torch.float)

#%%
x_test, target_test = utils.create_inout_sequences(X, Y, 64, 10, model="cnn")

rmse_train = []
rmse_test = []

model = models.ConvN([X.shape[1],32,1], [X.shape[1],14], 2 ,10, nn.ReLU)

optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

x_train, y_train = utils.create_inout_sequences(X, Y, "full", 10, model="cnn")

for epoch in range(3000):
    
    x, target = utils.create_inout_sequences(X, Y, 64, 10, model="cnn")

    
    model.train()
    output = model(x)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        
        pred_train = model(x_train)
        preds = model(x_test)
        #plt.plot(utils.minmax_rescaler(preds.numpy(), Y_mean, Y_std), label= epoch)
        mse_train = utils.rmse(pred_train, y_train)
        mse_test = utils.rmse(preds, target_test)
        #print('Epoch {}, loss {}, train_a {}, test_a {}'.format(epoch,loss.item(), 
          #np.sqrt(np.mean(mse_train.numpy())),
          #np.sqrt(np.mean(mse_test.numpy()))))
    
    rmse_train.append(mse_train)
    rmse_test.append(mse_test)


plt.plot(rmse_train, color="blue", label="train")
plt.plot(rmse_test, color="green", label="test")
 #%%
plt.plot(preds)
plt.plot(target)
#%%
XX = utils.reshaping(X, 10, model="cnn")
pred_train.shape
