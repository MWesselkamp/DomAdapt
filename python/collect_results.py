# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:18:31 2020

@author: marie
"""
import pandas as pd
import os.path
import numpy as np

import finetuning
import visualizations
#%%
def selected_networks_results(types, simsfrac):

    df_sel = pd.DataFrame(columns = ["id", "model", "typ", "architecture", "simsfrac","finetuned_type","dropout", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val", "task"])

    l = visualizations.losses("mlp", 0, r"noPool\relu")
    df_sel = df_sel.append({"id":"MLP0nP2D0R",
               "model":"mlp",
               "typ":0,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"noPool\sigmoid", plot=False)
    df_sel = df_sel.append({"id":"MLP0nP2D0S",
               "model":"mlp",
               "typ":0,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\nodropout\relu", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D0R",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\nodropout\sigmoid", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D0S",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\dropout\dropout10\relu", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D1R",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":1,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\dropout\dropout10\sigmoid", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D1S",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":1,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\dropout\dropout20\relu", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D2R",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":2,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)

    l = visualizations.losses("mlp", 0, r"adaptive_pooling\architecture3\dropout\dropout20\sigmoid", plot=False)
    df_sel = df_sel.append({"id":"MLP0aP3D2S",
               "model":"mlp",
               "typ":0,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":2,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)
    
    l = visualizations.losses("cnn", 0, r"")
    df_sel = df_sel.append({"id":"CNN0nP2D0R",
               "model":"cnn",
               "typ":0,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)
    
    l = visualizations.losses("lstm", 0, r"")
    df_sel = df_sel.append({"id":"LSTM0nP2D0R",
               "model":"lstm",
               "typ":0,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"selected"}, ignore_index=True)
    
    for typ in types:
        epochs = [50000,50000,60000,80000]
        for i in range(len(simsfrac)):
            l = visualizations.losses("mlp", typ, f"nodropout\sims_frac{simsfrac[i]}")
            df_sel = df_sel.append({"id":f"MLP{typ}D0{simsfrac[i]}P0",
                                    "model":"mlp",
                                    "typ":typ,
                                    "architecture":3,
                                    "simsfrac":None,
                                    "finetuned_type":None,
                                    "dropout":2,
                                    "epochs":epochs[i],
                                    "rmse_train":l["rmse_train"][0],
                                    "rmse_val":l["rmse_val"][0],
                                    "mae_train":l["mae_val"][0],
                                    "mae_val":l["mae_val"][0],
                                    "task":"pretraining"}, ignore_index=True)
    
    
    df_sel.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\selectednetworks.xlsx")
    df_sel.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\selectednetworks.csv")
    
    return(df_sel)


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
def get_predictions(model, typ, searchpath,
                      data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"):
    
    data_dir = os.path.join(data_dir, f"outputs\models\{model}{typ}")
    data_dir = os.path.join(data_dir, searchpath)

    results = pd.read_csv(os.path.join(data_dir, "selected_results.csv"))
    print(results)
    y_preds = np.load(os.path.join(data_dir,"y_preds.npy"), allow_pickle=True).item()

    #visualizations.plot_nn_predictions(y_tests, y_preds)
    #return(y_tests,y_preds)
    return(y_preds)
