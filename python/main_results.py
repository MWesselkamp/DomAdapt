# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:42:45 2020

@author: marie

The feature extractors take the arguments:
    
    model (str): pretrained network class to load (use mlp)
    typ (int): what type of pretrained network to load. 7 has seen normal parameters, 8 uniform parameters.
    epochs (int): if feature extraction involves retraining, enter number of epochs. Else None.
    simsfrac (int): on how much of the simuluated data has the network been pretrained? (E.g. 30 for 30%. Available: 30,50,80,100)
    
"""

import pandas as pd
import collect_results
import matplotlib.pyplot as plt
import seaborn

#%%
subtab1, running_losses, predictions = collect_results.feature_extraction_results(types = [7,8], simsfrac = [30, 50])
subtab2 = collect_results.selected_networks_results(types = [7,8], simsfrac = [30,50, 70, 100])

fulltab = pd.concat([subtab1, subtab2])
fulltab.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\results_full.xlsx")
fulltab.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv")

#%% PLOT 1

xi = [[fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A")]["mae_val"]], 
                   [fulltab.loc[fulltab.task =="selected"]["mae_val"]]]
yi = [[fulltab.loc[(fulltab.task =="finetuning")& (fulltab.finetuned_type != "A")]["rmse_val"]], 
                   [fulltab.loc[fulltab.task =="selected"]["rmse_val"]]]
m = ['x', 'o']
for i in range(len(xi)):
    plt.scatter(xi[i], yi[i], marker=m[i])
plt.ylim(0, 1.5)    
plt.xlim(0, 1.5)

#%% PLOT 1.2
cols = seaborn.color_palette(palette="pastel")
seaborn.boxplot(x = "finetuned_type",
            y = "mae_val",
            palette = "pastel",
            data = fulltab.loc[(fulltab.task =="finetuning")  & (fulltab.finetuned_type != "A")],
            width=0.5,
            linewidth = 0.8)

#%%
seaborn.boxplot(y = "mae_val", 
                x = "simsfrac",
                hue = "typ",
                palette = "pastel",
                data = fulltab.loc[(fulltab.task =="finetuning") & (fulltab.finetuned_type != "A") & (fulltab.finetuned_type != "C-NNLS")] ,
                width=0.5,
                linewidth = 0.8)
#%% Plot 4: plt.errorbar! linestyle='None', marker='^'