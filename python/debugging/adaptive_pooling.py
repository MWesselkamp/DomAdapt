# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:00:08 2020

@author: marie
"""
import torch.nn as nn
import torch

#%%
m = nn.AdaptiveAvgPool1d(2)

#inp = torch.randn(1, 8, 4)
inp = torch.Tensor(((1,0,2,0),(2,0,3,0),(3,0,4,0))).unsqueeze(0)
inp
out = m(inp)
out
