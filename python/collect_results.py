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
import setup.preprocessing as preprocessing
import setup.utils as utils
from sklearn import metrics

import torch
import torch.nn as nn
import setup.models as models
from ast import literal_eval
#%%
def selected_networks_results(types, simsfrac):

    df_sel = pd.DataFrame(columns = ["id", "model", "typ", "architecture", "simsfrac","finetuned_type","dropout", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val", "task"])

    l = visualizations.losses("mlp", 0, r"noPool\relu", plot=False)
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
    
    l = visualizations.losses("cnn", 0, r"", plot=False)
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
    
    l = visualizations.losses("lstm", 0, r"", plot=False)
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
    
    l = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\rf0\errors.npy")
    df_sel = df_sel.append({"id":"RF0",
               "model":"rf",
               "typ":0,
               "architecture":None,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":None,
               "epochs":None,
               "rmse_train":l[0],
               "rmse_val":l[1],
               "mae_train":l[2],
               "mae_val":l[3],
               "task":"selected"}, ignore_index=True)
    
    l = visualizations.losses("mlp", 2, r"adaptive_pooling\dropout", plot=False)
    df_sel = df_sel.append({"id":"MLP2aP3D1R",
               "model":"mlp",
               "typ":2,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":1,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"architecture_search"}, ignore_index=True)
    
    l = visualizations.losses("mlp", 2, r"adaptive_pooling\nodropout", plot=False)
    df_sel = df_sel.append({"id":"MLP2aP3D0R",
               "model":"mlp",
               "typ":2,
               "architecture":3,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"architecture_search"}, ignore_index=True)
    
    l = visualizations.losses("mlp", 2, r"noPool", plot=False)
    df_sel = df_sel.append({"id":"MLP2nP2D0R",
               "model":"mlp",
               "typ":2,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"architecture_search"}, ignore_index=True)
    
    l = visualizations.losses("cnn", 2, r"nodropout", plot=False)
    df_sel = df_sel.append({"id":"CNN2nP2D0R",
               "model":"cnn",
               "typ":2,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"architecture_search"}, ignore_index=True)
    
    l = visualizations.losses("lstm", 2, r"nodropout", plot=False)
    df_sel = df_sel.append({"id":"LSTM2nP2D0R",
               "model":"lstm",
               "typ":2,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":10000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"architecture_search"}, ignore_index=True)
    
    l = np.load(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\rf2\errors.npy")
    l = np.mean(l,1)
    df_sel = df_sel.append({"id":"RF2",
               "model":"rf",
               "typ":2,
               "architecture":2,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":None,
               "epochs":None,
               "rmse_train":l[0],
               "rmse_val":l[1],
               "mae_train":l[2],
               "mae_val":l[3],
               "task":"architecture_search"}, ignore_index=True)
    
    for dropout in ["nodropout", "dropout"]:
        if dropout == "nodropout":
            do = 0
        else:
            do = 1
        epochs = [10000,20000,30000,40000]
        for i in range(len(simsfrac)):
            l = visualizations.losses("mlp", 6, f"{dropout}\sims_frac{simsfrac[i]}", plot=False)
            df_sel = df_sel.append({"id":f"MLP6D{do}{simsfrac[i]}P0",
                                    "model":"mlp",
                                    "typ":6,
                                    "architecture":3,
                                    "simsfrac":simsfrac[i],
                                    "finetuned_type":None,
                                    "dropout":do,
                                    "epochs":epochs[i],
                                    "rmse_train":l["rmse_train"][0],
                                    "rmse_val":l["rmse_val"][0],
                                    "mae_train":l["mae_train"][0],
                                    "mae_val":l["mae_val"][0],
                                    "task":"pretraining"}, ignore_index=True)
    
    for typ in types:
        epochs = [50000,50000,60000,80000]
        for i in range(len(simsfrac)):
            l = visualizations.losses("mlp", typ, f"nodropout\sims_frac{simsfrac[i]}", plot=False)
            df_sel = df_sel.append({"id":f"MLP{typ}D0{simsfrac[i]}P0",
                                    "model":"mlp",
                                    "typ":typ,
                                    "architecture":3,
                                    "simsfrac":simsfrac[i],
                                    "finetuned_type":None,
                                    "dropout":2,
                                    "epochs":epochs[i],
                                    "rmse_train":l["rmse_train"][0],
                                    "rmse_val":l["rmse_val"][0],
                                    "mae_train":l["mae_train"][0],
                                    "mae_val":l["mae_val"][0],
                                    "task":"pretraining"}, ignore_index=True)
    
    
    pre = preles_errors("hyytiala")
    df_sel = df_sel.append({"id":"preles2008hy",
                                    "model":"preles",
                                    "typ":0,
                                    "architecture":None,
                                    "simsfrac":None,
                                    "finetuned_type":None,
                                    "dropout":None,
                                    "epochs":None,
                                    "rmse_train":pre[0],
                                    "rmse_val":pre[1],
                                    "mae_train":pre[2],
                                    "mae_val":pre[3],
                                    "task":"processmodel"}, ignore_index=True)
    
    pre = preles_errors("bily_kriz")
    df_sel = df_sel.append({"id":"preles2008bk",
                                    "model":"preles",
                                    "typ":2,
                                    "architecture":None,
                                    "simsfrac":None,
                                    "finetuned_type":None,
                                    "dropout":None,
                                    "epochs":None,
                                    "rmse_train":pre[0],
                                    "rmse_val":pre[1],
                                    "mae_train":pre[2],
                                    "mae_val":pre[3],
                                    "task":"processmodel"}, ignore_index=True)
    
    preds_er = borealsites_predictions()["mlp_prediction_errors"]
    df_sel = df_sel.append({"id":"mlp0nP2D0Rbs",
                                    "model":"mlp",
                                    "typ":0,
                                    "architecture":2,
                                    "simsfrac":None,
                                    "finetuned_type":None,
                                    "dropout":None,
                                    "epochs":None,
                                    "rmse_train":None,
                                    "rmse_val":preds_er[0],
                                    "mae_train":None,
                                    "mae_val":preds_er[1],
                                    "task":"borealsitesprediction"}, ignore_index=True)
    
    preds_er = borealsites_predictions()["preles_prediction_errors"]
    df_sel = df_sel.append({"id":"prelesbs",
                                    "model":"preles",
                                    "typ":None,
                                    "architecture":None,
                                    "simsfrac":None,
                                    "finetuned_type":None,
                                    "dropout":None,
                                    "epochs":None,
                                    "rmse_train":None,
                                    "rmse_val":preds_er[0],
                                    "mae_train":None,
                                    "mae_val":preds_er[1],
                                    "task":"borealsitesprediction"}, ignore_index=True)
    
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
            rl, errors, predictions_mlp2 = finetuning.featureExtractorD("mlp", typ, 1000, frac)
            errors = np.mean(np.array(errors),0)
            domadapt_errors.append([f"MLP{typ}D0{frac}FD", "mlp", typ,5, frac, "D-MLP2", 0, 1000, errors[0], errors[1],errors[2], errors[3], "finetuning"])
            running_losses["D-MLP2"] = rl
            domadapt_predictions["D-MLP2"] = predictions_mlp2
            
            ## Feature Extractor B due to computation time only used on cluster! ##
            ## LOADING RESULTS ##
            
            #4) Full Backprob with pretrained weights
            rets_fb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting0\selected_results.csv")
            rl_fb = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting0\\running_losses.npy", allow_pickle=True)
            domadapt_errors.append([f"MLP{typ}D0{frac}FB1", "mlp", typ, 5, frac, "B-fb", 0, rets_fb["epochs"][0],rets_fb["rmse_train"][0], rets_fb["rmse_val"][0],rets_fb["mae_train"][0], rets_fb["mae_val"][0], "finetuning"])
            running_losses["B-full_backprop"] = rl_fb
            
            #5) Backprop only last layer.
            rets_hb = pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting1\selected_results.csv")
            rl_fw2 = np.load(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\outputs\models\mlp{typ}\\nodropout\sims_frac{frac}\\tuned\setting1\\running_losses.npy", allow_pickle=True)
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

#%%
def preles_errors(site, data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    X_test, Y_test = preprocessing.get_splits(sites = [site],
                                years = [2008],
                                datadir = os.path.join(data_dir, "data"), 
                                dataset = "profound",
                                simulations = None)
    
    prelesGPP_def =  pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\data\profound\output{site}2008def", sep=";")
    prelesGPP_calib =  pd.read_csv(f"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\data\profound\output{site}2008calib", sep=";")

    rmse_train = utils.rmse(Y_test, prelesGPP_def)[0]
    rmse_val = utils.rmse(Y_test, prelesGPP_calib)[0]
    mae_train = metrics.mean_absolute_error(Y_test, prelesGPP_def)
    mae_val = metrics.mean_absolute_error(Y_test, prelesGPP_calib)
    
    errors = [rmse_train, rmse_val, mae_train, mae_val]
    
    return(errors)
#%% 
def borealsites_predictions(data_dir="OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    res = pd.read_csv(os.path.join(data_dir, r"python\outputs\models\mlp0\noPool\relu\selected_results.csv"))
    dimensions = [7]
    
    for hdim in literal_eval(res["hiddensize"].item()):
        dimensions.append(hdim)
    dimensions.append(1)
        
    X, Y = preprocessing.get_borealsites(year = "both")
    X = torch.tensor(X).type(dtype=torch.float)
    
    val_errors_mlp = {"rmse":[], "mae":[]}
    for i in range(5):
        
        model = models.MLP(dimensions, nn.ReLU)
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp0\\noPool\\relu\model{i}.pth")))
           
        preds = model(X)
            
        val_errors_mlp["rmse"].append(utils.rmse(Y , preds.detach().numpy()))
        val_errors_mlp["mae"].append(metrics.mean_absolute_error(Y, preds.detach().numpy()))
            
    val_errors_mlp = [np.mean(val_errors_mlp["rmse"]), np.mean(val_errors_mlp["mae"])]
    preles_preds = preprocessing.get_borealsites(year = "both", preles=True)
    
    val_errors_preles = [utils.rmse(Y , preles_preds), metrics.mean_absolute_error(Y, preles_preds)]
            
    prediction_errors = {"mlp_prediction_errors":val_errors_mlp,
                             "preles_prediction_errors":val_errors_preles}

    return(prediction_errors)
    
#%%
def sparse_networks_results(sparses):

    df_sel = pd.DataFrame(columns = ["id", "model", "typ", "architecture", "sparse", "simsfrac","finetuned_type","dropout", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val", "task"])
    
    for sparse in sparses:
        l = visualizations.losses("mlp", 0, f"sparse{sparse}",sparse=True, plot=False)
        df_sel = df_sel.append({"id":f"MLP0{sparse}1D0",
               "model":"mlp",
               "typ":0,
               "architecture":6,
               "sparse":sparse,
               "simsfrac":None,
               "finetuned_type":None,
               "dropout":0,
               "epochs":1000,
               "rmse_train":l["rmse_train"][0],
               "rmse_val":l["rmse_val"][0],
               "mae_train":l["mae_val"][0],
               "mae_val":l["mae_val"][0],
               "task":"sparse_selected"}, ignore_index=True)
    
    settings=["B-fb","B-fW2"]
    epochs = [5000, 40000]
    for i in range(len(settings)):
        for typ in [6,7,8]:
            for sparse in [1,2,3]:
                l = visualizations.losses("mlp", typ, f"sparse1\setting{i}", sparse=True, plot=False)
                df_sel = df_sel.append({"id":f"MLP0S{sparse}D0",
                                        "model":"mlp",
                                        "typ":typ,
                                        "architecture":6,
                                        "sparse":sparse,
                                        "simsfrac":30,
                                        "finetuned_type":settings[i],
                                        "dropout":0,
                                        "epochs":epochs[i],
                                        "rmse_train":l["rmse_train"][0],
                                        "rmse_val":l["rmse_val"][0],
                                        "mae_train":l["mae_val"][0],
                                        "mae_val":l["mae_val"][0],
                                        "task":"sparse_finetuning"}, ignore_index=True)
    
    years_list = [[2006], [2005, 2006], [2004,2005,2006], [2003,2004,2005,2006], [2001, 2003,2004,2005,2006]]
    for typ in [6,7,8]:
        for i in range(len(years_list)):
            predictions_test, errors = finetuning.featureExtractorC("mlp", 7, None, 30, classifier = "ols", 
                      years = years_list[i])
            errors = np.mean(np.array(errors), axis=1)
            df_sel = df_sel.append({"id":f"MLP0S{i+1}D0",
                                        "model":"mlp",
                                        "typ":typ,
                                        "architecture":6,
                                        "sparse":i+1,
                                        "simsfrac":30,
                                        "finetuned_type":"C-OLS",
                                        "dropout":0,
                                        "epochs":None,
                                        "rmse_train":errors[0],
                                        "rmse_val":errors[1],
                                        "mae_train":errors[2],
                                        "mae_val":errors[3],
                                        "task":"sparse_finetuning"}, ignore_index=True)
                
    
    df_sel.to_excel(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\sparsenetworks.xlsx")
    df_sel.to_csv(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\results\tables\sparsenetworks.csv")
    
    return(df_sel)

    
#%% Table 4.

def analyse_basemodel_results(percentages, data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):

    df_perc = pd.DataFrame(columns = ["id", "model", "typ", "CVfold","finetuned_type", "perc", "epochs", "rmse_train", "rmse_val", "mae_train", "mae_val", "task"])


    for perc in percentages:
    
        rets = pd.read_csv(os.path.join(data_dir, f"python\outputs\models\mlp0\\adaptive_pooling\\architecture3\\nodropout\sigmoid\data{perc}perc\selected_results.csv"))
        df_perc = df_perc.append({"id": None,
                              "model":"mlp",
                              "typ":0,
                              "CVfold":None,
                              "finetuned_type":None,
                              "perc":perc,
                              "epochs":10000,
                              "rmse_train":rets["rmse_train"].item(),
                              "rmse_val":rets["rmse_val"].item(),
                              "mae_train":rets["mae_train"].item(),
                              "mae_val":rets["mae_val"].item()}, ignore_index=True)
    
    df_perc.to_excel(os.path.join(data_dir, r"results\analyse_basemodel.xlsx"))
    df_perc.to_csv(os.path.join(data_dir, r"results\tables\analyse_basemodel.csv"))
    
    return(df_perc)

