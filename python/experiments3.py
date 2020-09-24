# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:30:58 2020

@author: marie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import numpy as np
import utils
import preprocessing
#%%%
datadir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
X, Y = preprocessing.get_splits(sites = ['collelongo','le_bray'],
                                years = [2001,2002,2003,2004,2005,2006, 2007],
                                datadir = os.path.join(datadir, "data"), 
                                dataset = "profound",
                                simulations = "preles",
                                colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR","DOY_sin", "DOY_cos"],
                                to_numpy = True)

Y_mean, Y_std = np.mean(Y), np.std(Y)

X = torch.tensor(X).type(dtype=torch.float)
Y = torch.tensor(Y).type(dtype=torch.float)
#%% Train
#model = models.MLP([X.shape[1],12,1], nn.ReLU)
#model = models.LSTM(X.shape[1], 12, 1, 10, F.relu)
model = models.ConvN([X.shape[1],12,1], [X.shape[1],14], 3,10, nn.ReLU)

x, target = utils.create_inout_sequences(X, Y, 128, 10, model="cnn")
x_test, target_test = utils.create_inout_sequences(X, Y, 128, 10, model="cnn")

#x, target = utils.create_batches(X, Y, 128, 0)
#x_test, target_test = utils.create_batches(X, Y, 128, 0)

x, target = utils.minmax_scaler(x), utils.minmax_scaler(target)
x_test, target_test = utils.minmax_scaler(x_test), utils.minmax_scaler(target_test)

x = torch.tensor(x).type(dtype=torch.float)
target = torch.tensor(target).type(dtype=torch.float)
x_test = torch.tensor(x_test).type(dtype=torch.float)
target_test = torch.tensor(target_test).type(dtype=torch.float)

optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

rmse_train = []
rmse_test = []
for epoch in range(700):
    
    model.train()
    output = model(x)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        
        preds = model(x_test)
        #plt.plot(utils.minmax_rescaler(preds.numpy(), Y_mean, Y_std), label= epoch)
        mse_train = np.square(target-model(x))
        mse_test = np.square(target_test-preds)
    print('Epoch {}, loss {}, train_a {}, test_a {}'.format(epoch, 
          loss.item(), 
          np.sqrt(np.mean(mse_train.numpy())),
          np.sqrt(np.mean(mse_test.numpy()))))
    
    rmse_train.append(np.sqrt(np.mean(mse_train.numpy())))
    rmse_test.append(np.sqrt(np.mean(mse_test.numpy())))
    

import matplotlib.pyplot as plt
plt.plot(rmse_train)
plt.plot(rmse_test)
 #%%
plt.plot(preds)
plt.plot(target)
