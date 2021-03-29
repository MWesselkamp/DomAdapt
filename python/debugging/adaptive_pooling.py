# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:00:08 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import torch.nn as nn
import torch
import os.path
import setup.preprocessing as preprocessing
import setup.utils as utils
import numpy as np

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"

X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)

X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
#%%
m = nn.AdaptiveMaxPool1d(5)
inp = torch.randn(1, 10, 8)
inp.shape
out = m(inp)
out.shape

#%%
x, y = utils.create_batches(X, Y , 64, 1)
x.shape
x = torch.tensor(x).type(dtype=torch.float)
fc = nn.Linear(1, 64)

ls = []
for i in range(x.shape[1]):
    latent = fc(x[:,i].unsqueeze(1))
    ls.append(latent)
    
torch.stack(ls).shape
#x.unsqueeze(1).shape

#%%
latent = []
        
for i in range(x.shape[1]):
    print("input dimensions: ", x.shape)
    encoded_feature = fc(x.unsqueeze(1)[:,:,i]).unsqueeze(2)
    print("encoded dimensions: ", encoded_feature.shape)
    if i > 7:
        drop = np.random.binomial(1, 0.5)
        if drop == 1:
            encoded_feature[encoded_feature!=0] = 0
                    
    latent.append(encoded_feature)
                    
        
latent = torch.stack(latent, dim=2).squeeze(3)
print("latent dimensions: ", latent.shape)

m(latent).view(x.shape[0],-1).shape

fc2 = nn.Linear(5*64, 32)

g = fc2(m(latent).view(x.shape[0],-1))
g.shape


#%%
m = nn.MaxPool1d(3, stride=2)
inp = torch.randn(20, 16, 50)
inp.shape
outp = m(inp)
outp.shape


#%%
import setup.models as models
import finetuning
import setup.dev_mlp as dev_mlp
import visualizations

x = torch.tensor(X).type(dtype=torch.float)
y = torch.tensor(Y).type(dtype=torch.float)

hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", 5, 4000, data_dir)
model_design["dimensions"] = [12,32,1]

X, Y = preprocessing.get_simulations(data_dir = os.path.join(data_dir, f"data/simulations/uniform_params"), drop_parameters=False)

X, Y = X[:500], Y[:500]     
X_f = np.random.rand(366, 12)
X_f[:,:7] = X_test
X_test = X_f
running_losses, performance, y_tests, y_preds = dev_mlp.train_model_CV(hparams, model_design, 
                                                                       X, Y, {"X_test":X_test, "Y_test":Y_test}, 
                                                                       0.4, data_dir, False)

visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], True, True)
