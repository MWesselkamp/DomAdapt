# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import os
os.getcwd()
os.chdir('OneDrive/Dokumente/Sc_Master/Masterthesis/Project/DomAdapt')
import numpy as np
import pandas as pd

import rpy2
print(rpy2.__version__) #2.9.4
import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri


pi = robjects.r['pi']
pi[0]

in_arr = np.array(robjects.r['''load("Rdata/input_arr.Rdata")'''])

numpy2ri.ri2numpy(r['d'])