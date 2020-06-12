# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import os
import os.path
os.getcwd()
os.chdir('OneDrive/Dokumente/Sc_Master/Masterthesis/Project/DomAdapt')

import pandas as pd

data_dir = ('data\preles')

data = {}

filesnum = int(len([name for name in os.listdir('data\preles')])/2)
filenames = [f'sim{i}' for i in range(1,filesnum+1)]

#%%
for filename in filenames:
    data[filename] = {}
    path_in = os.path.join(data_dir, f"{filename}_in")
    path_out = os.path.join(data_dir, f"{filename}_out")
    data[f"{filename}"]['y'] = pd.read_csv(path_out)
    # Doesn't work! Fix data in R.
    data[f"{filename}"]['X'] = pd.read_csv(path_in)

