# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:21:32 2021

@author: marie
"""

import setup.preprocessing as preprocessing
import setup.models as models
import finetuning
import os.path
import numpy as np
import pandas as pd
import torch

data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"
#%%
X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                years = [2001, 2002, 2003, 2004, 2005, 2006, 2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None,
                                to_numpy=False)

df_new = pd.DataFrame({"PAR": X_test["PAR"].mean(),
                       "TAir": np.arange(X_test["TAir"].min(), X_test["TAir"].max(), step=0.01),
                       "VPD": X_test["VPD"].mean(),
                       "Precip": X_test["Precip"].mean(),
                       "fAPAR": X_test["fAPAR"].mean(),
                       "DOY_sin": X_test["DOY_sin"].mean(),
                       "DOY_cos": X_test["DOY_cos"].mean()})
df_new.to_csv(os.path.join(data_dir, "data\post_analysis\df1.csv"), sep=",")

df = df_new.to_numpy()

#%%
preds_reference = []
for i in range(5):
    hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", 10, None, data_dir)
    model = models.MLP(model_design["dimensions"], model_design["activation"])
    model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp0\\relu\model{i}.pth")))
    x = torch.tensor(df).type(dtype=torch.float)
    preds_reference.append(model(x).detach().numpy())
    
preds_finetuned= []
for i in range(5):
    hparams, model_design, X, Y, X_test, Y_test = finetuning.settings("mlp", 10, None, data_dir)
    model = models.MLP(model_design["dimensions"], model_design["activation"])
    model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp10\\nodropout\sims_frac100\\tuned\setting1\model{i}.pth")))
    x = torch.tensor(df).type(dtype=torch.float)
    preds_finetuned.append(model(x).detach().numpy())

#%%
m = np.mean(np.transpose(np.array(preds_reference).squeeze(2)), axis=1)
q1 = np.quantile(np.transpose(np.array(preds_reference).squeeze(2)), 0.05, axis=1)
q2 = np.quantile(np.transpose(np.array(preds_reference).squeeze(2)), 0.95, axis=1)

fig, ax = plt.subplots(figsize=(7,7))
ax.fill_between(np.arange(len(m)), q1,q2, color="lightgrey", alpha=0.6)
ax.plot(m, color="black", label= "E$_{reference}$ - E$_{finetuned}$")