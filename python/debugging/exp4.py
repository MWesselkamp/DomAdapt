# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:29:12 2020

@author: marie
"""
#%%
import os.path
import dev_cnn
import dev_mlp
import dev_lstm
import torch
import torch.optim as optim
import torch.nn as nn
import preprocessing
import visualizations
import models
from sklearn import metrics
import utils
import numpy as np
#%%

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2003, 2004],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X = utils.minmax_scaler(X)
#%%



fc = nn.Linear(1,32)

latent = []
for feature in range(3):
    latent.append(fc(X.unsqueeze(1)[:,:,feature]).unsqueeze(2))
        
latent = torch.stack(latent, dim=2).squeeze(3)
latent.shape
latent = torch.mean(latent, dim=2)
latent.shape

avgpool = nn.AdaptiveAvgPool1d(1)

out = avgpool(tt).squeeze(2)
out.shape

fc2 = nn.Linear(32, 64)
out = fc2(out)
out.shape

#%%
dimensions = [32,64,64, 1]

X = torch.tensor(X).type(dtype=torch.float)


Y = torch.tensor(Y).type(dtype=torch.float)

#%%
model = models.MLPmod(7, dimensions, nn.ReLU)
out = model(X)
out.shape
#%%
model = models.MLPmod(7, dimensions, nn.ReLU)
mae_trains = []
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

#x_test, target_test = utils.create_batches(X, Y, 64, 1)

for epoch in range(1500):
    
    x, y = utils.create_batches(X, Y, 256, 1)
    
    x = torch.tensor(x).type(dtype=torch.float)
    y = torch.tensor(y).type(dtype=torch.float)

    model.train()
    output = model(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        
        pred_train = model(X)
        #preds = model(x_test)
        #plt.plot(utils.minmax_rescaler(preds.numpy(), Y_mean, Y_std), label= epoch)
        mae_train = metrics.mean_absolute_error(Y, pred_train)
        #mse_test = utils.rmse(preds, target_test)
        #print('Epoch {}, loss {}, train_a {}, test_a {}'.format(epoch,loss.item(), 
          #np.sqrt(np.mean(mse_train.numpy())),
          #np.sqrt(np.mean(mse_test.numpy()))))
    
    mae_trains.append(mae_train)
    #rmse_test.append(mse_test)

#%%
import matplotlib.pyplot as plt

plt.plot(mae_trains[5:])
mae_trains[-1]
