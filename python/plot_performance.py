# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import setup.preprocessing as preprocessing
import visualizations


#%% plot the losses of the baseline model MLP0 

visualizations.losses("mlp", 0, r"noPool\relu")
visualizations.losses("mlp", 0, r"noPool\sigmoid")
visualizations.losses("mlp", 0, r"AdaptPool\architecture3\nodropout")
visualizations.losses("mlp", 0, r"AdaptPool\architecture3\dropout\dropout10")
visualizations.losses("mlp", 0, r"AdaptPool\architecture3\dropout\dropout20")

l = visualizations.losses("mlp", 4, r"")
l = visualizations.losses("mlp", 4, r"adaptPool\nodropout")
l = visualizations.losses("mlp", 4, r"adaptPool\dropout")

l = visualizations.losses("mlp", 2, r"bilykriz\adaptPool\dropout")
l = visualizations.losses("mlp", 2, r"bilykriz\adaptPool\nodropout")
l = visualizations.losses("mlp", 2, r"bilykriz\noPool")

l = visualizations.losses("cnn", 2, r"nodropout")
l = visualizations.losses("lstm", 2, r"nodropout")

visualizations.losses("mlp", 5, r"paramsFix\nodropout")
visualizations.losses("mlp", 6, r"paramsFix\nodropout")

visualizations.losses("mlp", 5, r"paramsFix\dropout")
visualizations.losses("mlp", 6, r"paramsFix\dropout")

l = visualizations.losses("mlp", 7, r"nodropout")
l = visualizations.losses("mlp", 7, r"dropout")
l = visualizations.losses("mlp", 8, r"nodropout")
l = visualizations.losses("mlp", 8, r"dropout")
#%%
visualizations.predictions("mlp", 0, r"noPool")
visualizations.predictions("mlp", 0, r"AdaptPool\nodropout")
visualizations.predictions("mlp", 0, r"AdaptPool\dropout")

visualizations.predictions("mlp", 2, r"hyytiala")
visualizations.predictions("mlp", 2, r"bilykriz\noPool")

visualizations.predictions("lstm", 2, r"bilykriz")
visualizations.predictions("cnn", 2, r"bilykriz")
visualizations.predictions("lstm", 2, r"hyytiala")
visualizations.predictions("cnn", 2, r"hyttiala")

visualizations.predictions("rf", 2)

visualizations.predictions("mlp", 5, r"paramsFix\nodropout")
visualizations.predictions("mlp", 6, r"paramsFix\nodropout")
#%%
visualizations.performance_boxplots(typ=2)

#%%
visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], "mlp")

#%%
visualizations.plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], "mlp")
