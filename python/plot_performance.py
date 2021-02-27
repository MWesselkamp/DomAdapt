# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020

@author: marie
"""
import sys
sys.path.append('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python')

import setup.preprocessing as preprocessing
import visualizations

#%%
l = visualizations.losses("cnn", 2, r"")
l = visualizations.losses("lstm", 0, r"")

# REFERNCE NETWORK:SHALLOW ARCHITECTURE 
l = visualizations.losses("mlp", 0, r"relu")
l = visualizations.losses("mlp", 0, r"relu\sparse\1")
l = visualizations.losses("mlp", 0, r"relu\sparse\5")
l = visualizations.losses("mlp", 0, r"relu\sparse\50")

# REFERNCE NETWORK:DEEP ARCHITECTURE 
l = visualizations.losses("mlp", 4, r"relu")
l = visualizations.losses("mlp", 4, r"relu\sparse\1")
l = visualizations.losses("mlp", 4, r"relu\sparse\5")
l = visualizations.losses("mlp", 4, r"relu\sparse\50")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting0\sparse\5")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting1\sparse\5")
l = visualizations.losses("mlp", 4, r"relu\sparse\10")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting0\sparse\10")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting1\sparse\10")
l = visualizations.losses("mlp", 4, r"relu\sparse\20")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting0\sparse\20")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting1\sparse\20")
l = visualizations.losses("mlp", 4, r"relu\sparse\30")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting0\sparse\30")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting1\sparse\30")
l = visualizations.losses("mlp", 4, r"relu\sparse\90")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting0\sparse\90")
l = visualizations.losses("mlp", 4, r"nodropout\sims_frac100\tuned\setting1\sparse\90")

l = visualizations.losses("mlp", 2, r"bilykriz\adaptPool\dropout")
l = visualizations.losses("mlp", 2, r"bilykriz\adaptPool\nodropout")
l = visualizations.losses("mlp", 2, r"bilykriz\noPool")

l = visualizations.losses("cnn", 2, r"nodropout")
l = visualizations.losses("lstm", 2, r"nodropout")

# REFERNCE NETWORK AND PRETRAINING: ARCHITECTURE WITH ADAPTIVE POOLING. 
l = visualizations.losses("mlp", 5, r"relu")
l = visualizations.losses("mlp", 5, r"relu\sparse\1")
l = visualizations.losses("mlp", 5, r"relu\sparse\2")
l = visualizations.losses("mlp", 5, r"relu\sparse\3")
l = visualizations.losses("mlp", 5, r"relu\sparse\5")
l = visualizations.losses("mlp", 5, r"relu\sparse\50")
l = visualizations.losses("mlp", 5, r"relu\sparse\95")

# PRETRAINING WITH PARAMETERS
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac50\tuned\setting0")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac50\tuned\setting1")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac100\tuned\setting1")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac30")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac70\tuned\setting1")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac100")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 5, r"nodropout\dummies\sims_frac100\tuned\setting1\freeze1")
l = visualizations.losses("mlp", 5, r"nodropout\sims_frac100\tuned\setting1\freeze2")
l = visualizations.losses("mlp", 5, r"dropout\01\sims_frac100")
l = visualizations.losses("mlp", 5, r"dropout\02\sims_frac100")
l = visualizations.losses("mlp", 5, r"dropout\03\sims_frac100")
l = visualizations.losses("mlp", 5, r"dropout\04\sims_frac100")
l = visualizations.losses("mlp", 5, r"dropout\05\sims_frac100")
l = visualizations.losses("mlp", 5, r"dropout\01\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 5, r"dropout\02\sims_frac100\tuned\setting1")
l = visualizations.losses("mlp", 5, r"dropout\03\sims_frac100\tuned\setting1")
l = visualizations.losses("mlp", 5, r"dropout\04\sims_frac100\tuned\setting1\freeze2")
l = visualizations.losses("mlp", 5, r"dropout\05\sims_frac100\tuned\setting1\freeze1")


l = visualizations.losses("mlp", 6, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 6, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 6, r"nodropout\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 6, r"nodropout\sims_frac50\tuned\setting0")
l = visualizations.losses("mlp", 6, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 6, r"nodropout\sims_frac100\tuned\setting1")

# ARCHITECTURE WITH ADAPTIVE POOLING. PRETRAINING WITHOUT PARAMETERS
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 7, r"nodropout\sims_frac100\tuned\setting1")

l = visualizations.losses("mlp", 8, r"relu")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac50")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac30\\tuned\setting1")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 8, r"nodropout\sims_frac100\tuned\setting1")

# PRETRAINING:SHALLOW ARCHITECTURE WITH FIXED PARAMETERS
l = visualizations.losses("mlp", 9, r"nodropout\sims_frac50")
l = visualizations.losses("mlp", 9, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 9, r"nodropout\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 9, r"nodropout\sims_frac70\tuned\setting0")

# PRETRAINING:SHALLOW ARCHITECTURE WITHOUT PARAMETERS
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac100\tuned\setting1")

# PRETRAINING:DEEP ARCHITECTURE WITH FIXED PARAMETERS
l = visualizations.losses("mlp", 11, r"nodropout\sims_frac50")
l = visualizations.losses("mlp", 11, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 11, r"nodropout\sims_frac30\tuned\setting0")
l = visualizations.losses("mlp", 11, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 11, r"nodropout\sims_frac100\tuned\setting1")

# PRETRAINING:DEEP ARCHITECTURE WITHOUT PARAMETERS
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac100\tuned\setting0")
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac100\tuned\setting1")

# PRETRAINING:SHALLOW ARCHITECTURE WITH PARAMETERS (DUMMIES)
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac50")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac70")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac50\tuned\setting1")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac70\tuned\setting1")
l = visualizations.losses("mlp", 13, r"nodropout\sims_frac100\tuned\setting1")

# PRETRAINING:DEEP ARCHITECTURE WITH PARAMETERS (DUMMIES)
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac30")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac50")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac70")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac100")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac30\tuned\setting1")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac50\tuned\setting1")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac70\tuned\setting1")
l = visualizations.losses("mlp", 14, r"nodropout\sims_frac100\tuned\setting1")


# PRETRAINING:DEEP ARCHITECTURE WITHOUT PARAMETERS
l = visualizations.losses("mlp", 0, r"relu\sparse\15")
l = visualizations.losses("mlp", 10, r"nodropout\sims_frac100\tuned\setting0\sparse\15")
l = visualizations.losses("mlp", 4, r"relu\sparse\5")
l = visualizations.losses("mlp", 12, r"nodropout\sims_frac100\tuned\setting1\sparse\5")

#%%
visualizations.predictions("mlp", 0, r"noPool\sigmoid")
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
