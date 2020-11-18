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

import numpy as np

import finetuning

import collect_results

#%%
def feature_extraction_results(types, simsfrac):
    
    domadapt_errors = []
    domadapt_predictions = {}
    running_losses = {}

    for typ in types:
        for frac in simsfrac:
    
            predictions, errors, Y_test = finetuning.featureExtractorA("mlp", typ, None, frac)
            errors = np.mean(np.array(errors), 1)
            domadapt_errors.append([f"MLP{typ}D0{frac}FA", "mlp", typ, 5, frac, "A", 0, None, errors[0], errors[1],errors[2], errors[3], "finetuning"])
            domadapt_predictions["A-None"] = predictions
            
            # 1) Ordinary Least Squares as Classifier
            predictions_ols, errors = finetuning.featureExtractorC("mlp",typ, None, frac, "ols")
            errors = np.mean(np.array(errors), axis=1)
            domadapt_errors.append([f"MLP{typ}D0{frac}FC1", "mlp", typ,5,frac, "C-OLS", 0, None, errors[0], errors[1],errors[2], errors[3], "finetuning"])
            domadapt_predictions["C-OLS"] = predictions_ols
            #visualizations.plot_prediction(Y_test, predictions_ols, "OLS")
            
            # 2) Non-negative Least Squares as Classifier
            predictions_nnls, errors = finetuning.featureExtractorC("mlp", typ, None,  frac, "nnls")
            errors = np.mean(np.array(errors), axis=1)
            domadapt_errors.append([f"MLP{typ}D0{frac}FC2", "mlp", typ, 5, frac, "C-NNLS", 0, None, errors[0], errors[1],errors[2], errors[3], "finetuning"])
            domadapt_predictions["C-NNLS"] = predictions_nnls
            #visualizations.plot_prediction(Y_test, predictions_nnls, "Non-negative least squares")
            
            #3) MLP with architecture 2 as Classifier
            rl, errors, predictions_mlp2 = finetuning.featureExtractorD("mlp", typ, 500, frac)
            errors = np.mean(np.array(errors),0)
            domadapt_errors.append([f"MLP{typ}D0{frac}FD", "mlp", typ,5, frac, "D-MLP2", 0, 500, errors[0], errors[1],errors[2], errors[3], "finetuning"])
            running_losses["D-MLP2"] = rl
            domadapt_predictions["D-MLP2"] = predictions_mlp2
            
            ## Feature Extractor B due to computation time only used on cluster! ##
            ## LOADING RESULTS ##
            
            #4) Full Backprob with pretrained weights
            rets_fb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\\nodropout\sims_frac{frac}\\tuned\setting0\selected_results.csv")
            rl_fb = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\\nodropout\sims_frac{frac}\\tuned\setting0\\running_losses.npy", allow_pickle=True)
            domadapt_errors.append([f"MLP{typ}D0{frac}FB1", "mlp", typ, 5, frac, "B-fb", 0, rets_fb["epochs"][0],rets_fb["rmse_train"][0], rets_fb["rmse_val"][0],rets_fb["mae_train"][0], rets_fb["mae_val"][0], "finetuning"])
            running_losses["B-full_backprop"] = rl_fb
            
            #5) Backprop only last layer.
            rets_hb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\\nodropout\sims_frac{frac}\\tuned\setting1\selected_results.csv")
            rl_fw2 = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp7\\nodropout\sims_frac{frac}\\tuned\setting1\\running_losses.npy", allow_pickle=True)
            domadapt_errors.append([f"MLP{typ}D0{frac}FB2", "mlp", typ, 5, frac, "B-fW2", 0, rets_hb["epochs"][0],rets_hb["rmse_train"][0], rets_hb["rmse_val"][0],rets_hb["mae_train"][0], rets_hb["mae_val"][0], "finetuning"])
            running_losses["B-freezeW2"] = rl_fw2
            
    domadapt_results = pd.DataFrame(domadapt_errors,
                        columns = ["id", "model", "typ", "architecture", "simsfrac", "finetuned_type","dropout", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val", "task"])
    
    domadapt_results.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\featureextraction.xlsx")
    domadapt_results.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\featureextraction.csv")
    
    return(domadapt_results, running_losses, domadapt_predictions)
    
#%%
subtab1, running_losses, predictions = feature_extraction_results(types = [7,8], simsfrac = [30, 50])
subtab2 = collect_results.selected_networks_results(types = [7,8], simsfrac = [30,50, 70, 100])

fulltab = pd.concat([subtab1, subtab2])
fulltab.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\results_full.xlsx")
fulltab.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\results_full.csv")
