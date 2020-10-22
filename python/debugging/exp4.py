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
import utils
import numpy as np
#%%

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['le_bray'],
                                years = [2001,2003],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

#%%

fc = nn.Linear(1,32)

tt = torch.empty(730,32,1)

for i in range(3):
    tt = torch.cat((tt, fc(X[:,:,i]).unsqueeze(2)),dim=2)
tt.shape

avgpool = nn.AdaptiveAvgPool1d(1)

out = avgpool(tt).squeeze(2)
out.shape

fc2 = nn.Linear(32, 64)
out = fc2(out)
out.shape

#%%
dimensions = [32,64,64, 1]

X = utils.minmax_scaler(X)
X = torch.tensor(X).type(dtype=torch.float)
Y = torch.tensor(Y).type(dtype=torch.float)

#%%
model = models.MLPmod(32, dimensions, nn.ReLU)
out = model(X)
out.shape
#%%
rmse_trains = []
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

#x_test, target_test = utils.create_batches(X, Y, 64, 1)

for epoch in range(100):
    
    x, y = utils.create_batches(X, Y, 64, 1)
    
    x = torch.tensor(x).type(dtype=torch.float)
    y = torch.tensor(y).type(dtype=torch.float)

    model.train()
    output = model(x)
    loss = criterion(output, y)
    print(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        
        pred_train = model(X)
        #preds = model(x_test)
        #plt.plot(utils.minmax_rescaler(preds.numpy(), Y_mean, Y_std), label= epoch)
        rmse_train = utils.rmse(pred_train, Y)
        #mse_test = utils.rmse(preds, target_test)
        #print('Epoch {}, loss {}, train_a {}, test_a {}'.format(epoch,loss.item(), 
          #np.sqrt(np.mean(mse_train.numpy())),
          #np.sqrt(np.mean(mse_test.numpy()))))
    
    rmse_trains.append(rmse_train)
    #rmse_test.append(mse_test)

#%%
    
m = nn.AdaptiveAvgPool1d(5)
input = torch.randn(1, 64, 8)
output = m(input)
output.shape
