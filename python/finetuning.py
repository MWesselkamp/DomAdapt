# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:48:58 2020
@author: marie
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold
import os.path
import pandas as pd
from ast import literal_eval

import setup.models as models
import setup.utils as utils
import setup.preprocessing as preprocessing

from scipy.optimize import nnls
import statsmodels.api as sm

#%%
def settings(model, epochs, data_dir, sims = True,
             years = [2001,2002,2003, 2004, 2005, 2006, 2007]):

    X, Y = preprocessing.get_splits(sites = ['hyytiala'],
                                    years = years,
                                    datadir = os.path.join(data_dir, "data"), 
                                    dataset = "profound",
                                    simulations = None)
    X_test, Y_test = preprocessing.get_splits(sites = ['hyytiala'],
                                              years = [2008],
                                              datadir = os.path.join(data_dir, "data"), 
                                              dataset = "profound",
                                              simulations = None)
    
    if sims:
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\simulations\grid_search_results_{model}2_adaptPool.csv"))
    else:
        gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\observations\mlp\grid_search_results_{model}2.csv"))
        
    setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()

    if sims:
        dimensions = literal_eval(setup["hiddensize"])
        dimensions.append(1) # adds the output dimension!
    else:
        dimensions = [X.shape[1]]
        for dim in literal_eval(setup["hiddensize"]):
            dimensions.append(dim)
        dimensions.append(Y.shape[1])
    
    if sims:
        featuresize = setup["featuresize"]
    else:
        featuresize = None

    hparams = {"batchsize": int(setup["batchsize"]), 
               "epochs":epochs, 
               "history": int(setup["history"]), 
               "hiddensize":literal_eval(setup["hiddensize"]),
               "learningrate":setup["learningrate"]}

    model_design = {"dimensions":dimensions,
                    "activation":nn.ReLU,
                    "featuresize":featuresize}

      
    return hparams, model_design, X, Y, X_test, Y_test

#%%
def training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, featuresize, data_dir,
                   save, splits = 5):
    
    """
    
    
    """
    
    epochs = hparams["epochs"]
    
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    rmse_train = np.zeros((splits, epochs))
    rmse_val = np.zeros((splits, epochs))
    mae_train = np.zeros((splits, epochs))
    mae_val = np.zeros((splits, epochs))
    
    # z-score data
    #X_mean, X_std = np.mean(X), np.std(X)
    #X = utils.minmax_scaler(X)
    
    if not eval_set is None:
        print("Test set used for model evaluation")
        Xt_test = eval_set["X_test"]
        #Xt_test= utils.minmax_scaler(Xt_test, scaling = [X_mean, X_std])
        yt_test = eval_set["Y_test"]
        yt_test = torch.tensor(yt_test).type(dtype=torch.float)
        Xt_test = torch.tensor(Xt_test).type(dtype=torch.float)
        yt_tests = []
        
    i = 0
    
    performance = []
    y_tests = []
    y_preds = []
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        X_test = torch.tensor(X_test).type(dtype=torch.float)
        y_test = torch.tensor(y_test).type(dtype=torch.float)
        X_train = torch.tensor(X_train).type(dtype=torch.float)
        y_train = torch.tensor(y_train).type(dtype=torch.float)
        
        if isinstance(model_design, dict):
            print("Loading pretrained Model.")
            model = models.MLPmod(featuresize, model_design["dimensions"], model_design["activation"])
            model.load_state_dict(torch.load(os.path.join(data_dir, f"model{i}.pth")))
        else:
            model = model_design
        model.eval()
        
        if not feature_extraction is None:
            print("Freezing all weights.")
            for child in model.children():
                for name, parameter in child.named_parameters():
                    if not name in feature_extraction:
                        parameter.requires_grad = False
                    #else:
                    #    parameter.requires_grad = False
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = hparams["learningrate"])
        
        for epoch in range(epochs):
            
            # Training
            model.train()

            x, y = utils.create_batches(X_train, y_train, hparams["batchsize"], hparams["history"])
            
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
            
            output = model(x)
            
            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(X_train)
                if eval_set is None:
                    pred_test = model(X_test)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(y_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(y_test, pred_test)  
                else:
                    pred_test = model(Xt_test)
                    rmse_train[i, epoch] = utils.rmse(y_train, pred_train)
                    rmse_val[i, epoch] = utils.rmse(yt_test, pred_test)
                    mae_train[i, epoch] = metrics.mean_absolute_error(y_train, pred_train)
                    mae_val[i, epoch] = metrics.mean_absolute_error(yt_test, pred_test)
                    
         
        # Predict with fitted model
        with torch.no_grad():
            preds_train = model(X_train)
            if eval_set is None:
                preds_test = model(X_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(y_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(y_test, preds_test.numpy())])
            else:
                preds_test = model(Xt_test)
                performance.append([utils.rmse(y_train, preds_train),
                                    utils.rmse(yt_test, preds_test),
                                    metrics.mean_absolute_error(y_train, preds_train.numpy()),
                                    metrics.mean_absolute_error(yt_test, preds_test.numpy())])
    
        if save:
            if not feature_extraction is None:
                torch.save(model.state_dict(), os.path.join(data_dir, f"tuned\setting1\model{i}.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(data_dir, f"tuned\setting0\model{i}.pth"))
        
        y_tests.append(y_test.numpy())
        y_preds.append(preds_test.numpy())
        
    
        i += 1
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}

    if eval_set is None:
        return(running_losses, performance, y_tests, y_preds)
    else:
        return(running_losses, performance, yt_tests, y_preds)
        
#%%
def finetune(X, Y, epochs, model, pretrained_type, searchpath, featuresize, save=False, 
             feature_extraction = None, eval_set = None,
             data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    gridsearch_results = pd.read_csv(os.path.join(data_dir, f"python\outputs\grid_search\simulations\grid_search_results_{model}2_adaptPool.csv"))
    
    setup = gridsearch_results.iloc[gridsearch_results['mae_val'].idxmin()].to_dict()

    dimensions = literal_eval(setup["hiddensize"])
    dimensions.append(1) # adds the output dimension!

    hparams = {"batchsize": int(setup["batchsize"]), 
               "epochs":epochs, 
               "history": int(setup["history"]), 
               "hiddensize":literal_eval(setup["hiddensize"]),
               "learningrate":setup["learningrate"]}

    model_design = {"dimensions":dimensions,
                    "activation":nn.ReLU}
    
    data_dir = os.path.join(data_dir, f"python\outputs\models\{model}{pretrained_type}\{searchpath}")

    running_losses,performance, y_tests, y_preds = training_CV(hparams, model_design, X, Y,  feature_extraction, eval_set, featuresize,
                                                               data_dir, 
                                                               save)
        
    if not feature_extraction is None:
        data_dir = os.path.join(data_dir, "tuned\setting1")
    else:
        data_dir = os.path.join(data_dir, "tuned\setting0")
    
    performance = np.mean(np.array(performance), axis=0)
    rets = [epochs, pretrained_type, 
            performance[0], performance[1], performance[2], performance[3]]
    results = pd.DataFrame([rets], 
                           columns=["epochs", "pretrained_type", 
                                    "rmse_train", "rmse_val", "mae_train", "mae_val"])
    results.to_csv(os.path.join(data_dir, r"selected_results.csv"), index = False)
        
    # Save: Running losses, ytests and ypreds.
    np.save(os.path.join(data_dir, "running_losses.npy"), running_losses)
    np.save(os.path.join(data_dir, "y_tests.npy"), y_tests)
    np.save(os.path.join(data_dir, "y_preds.npy"), y_preds)
    
    return(running_losses,performance, y_tests, y_preds)

#%%
def featureExtractorA(model, typ, epochs, simsfrac,
                      years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                      splits=5, data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    if ((typ == 9)| (typ == 10)):
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years, sims=False)
        model_design["featuresize"] = None
    else:
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years)
    
    X = torch.tensor(X).type(dtype=torch.float)
    X_test = torch.tensor(X_test).type(dtype=torch.float)
    
    predictions = []
    mae_train = []
    rmse_train = []
    mae_val = []
    rmse_val = []

    for i in range(splits):
        
        if ((typ == 9)| (typ == 10)):
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{typ}\\nodropout\sims_frac{simsfrac}\model{i}.pth")))
        
        preds_test = model(X_test).detach().numpy()
        preds_train = model(X).detach().numpy()

        mae_val.append(metrics.mean_absolute_error(Y_test, preds_test))
        rmse_val.append(utils.rmse(Y_test, preds_test))
        mae_train.append(metrics.mean_absolute_error(Y, preds_train))
        rmse_train.append(utils.rmse(Y, preds_train))
        predictions.append(preds_test)
    
    errors = [rmse_train, rmse_val, mae_train, mae_val]

    return predictions, errors, Y_test

#%% Finetune network on finish data, Full Backprob.

def featureExtractorB(model, typ, epochs, simsfrac, feature_extraction= None,
                      years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                      data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    if ((typ == 9)| (typ == 10)):
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years, sims=False)
        model_design["featuresize"] = None
    else:
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years)
    
    running_losses,performance, y_tests, y_preds = finetune(X, Y, epochs, model, typ, f"nodropout\sims_frac{simsfrac}", model_design["featuresize"], 
                                                                       False, feature_extraction, {"X_test":X_test, "Y_test":Y_test})
    
    return performance, y_preds, Y_test

#%% 1) Ordinary Least Squares and friends
    
def featureExtractorC(model, typ, epochs, simsfrac, classifier = "ols", 
                      years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                      splits = 5, data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    if ((typ == 9)| (typ == 10)):
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years, sims=False)
        model_design["featuresize"] = None
    else:
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years)
    
    X = torch.tensor(X).type(dtype=torch.float)
    X_test = torch.tensor(X_test).type(dtype=torch.float)
    
    predictions_train = []
    predictions_test = []
    
    for i in range(splits):
    
        if ((typ == 9)| (typ == 10)):
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
            
        model.load_state_dict(torch.load(os.path.join(data_dir, f"python\outputs\models\mlp{typ}\\nodropout\sims_frac{simsfrac}\model{i}.pth")))
        
        if ((typ == 9)| (typ == 10)):
            model = model[:-1]
        else:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # Remove Final layer and activation.

        out_train = model(X).detach().numpy()
        out_train = sm.add_constant(out_train) # Add intercept.
        out_test = model(X_test).detach().numpy()
        out_test = sm.add_constant(out_test) # Add intercept.

        if classifier == "ols":
            extractor = sm.OLS(Y, out_train) 
            results = extractor.fit()
            predictions_train.append(np.expand_dims(results.predict(), axis=1))
            predictions_test.append(np.expand_dims(results.predict(out_test), axis=1))
            
        elif classifier == "glm":
            print("Fitting glm with Inverse Gaussian family and log-Link.")
            extractor = sm.GLM(Y, out_train, family=sm.families.InverseGaussian(sm.families.links.log())) 
            results = extractor.fit()
            predictions_train.append(np.expand_dims(results.predict(), axis=1))
            predictions_test.append(np.expand_dims(results.predict(out_test), axis=1))
            
        elif classifier == "nnls":
            theta = np.expand_dims(nnls(out_train, Y[:,0])[0], axis=1)
            predictions_train.append(np.dot(out_train, theta))
            predictions_test.append(np.dot(out_test, theta))
            
        else:
            print("Don't know classifier.")
            
    mae_train = [metrics.mean_absolute_error(Y, sublist) for sublist in predictions_train]
    mae_val = [metrics.mean_absolute_error(Y_test, sublist) for sublist in predictions_test]
    rmse_train = [utils.rmse(Y, sublist) for sublist in predictions_train]
    rmse_val = [utils.rmse(Y_test, sublist) for sublist in predictions_test]
    
    errors = [rmse_train, rmse_val, mae_train, mae_val]
    
    return predictions_test, errors
#%%
def train_model(hparams_add, model_design_add, X, Y, X_test, Y_test, i, 
                data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    epochs = hparams_add["epochs"]
    
    rmse_train = np.zeros((epochs))
    rmse_val = np.zeros((epochs))
    mae_train = np.zeros((epochs))
    mae_val = np.zeros((epochs))
    
    # Standardize X and X_test together!!
    #mu = np.concatenate((X, X_test), 0).mean()
    #sigma = np.concatenate((X, X_test), 0).std()
    #X = utils.minmax_scaler(X, [mu, sigma])
    #X_test = utils.minmax_scaler(X_test, [mu, sigma])

    X_test = torch.tensor(X_test).type(dtype=torch.float)
    y_test = torch.tensor(Y_test).type(dtype=torch.float)
    X_train = torch.tensor(X).type(dtype=torch.float)
    y_train = torch.tensor(Y).type(dtype=torch.float)
    
    model_design_add["dimensions"].insert(0,X.shape[1])
    
    model = models.MLP(model_design_add["dimensions"], model_design_add["activation"])
    optimizer = optim.Adam(model.parameters(), lr = hparams_add["learningrate"])
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
            
            # Training
            model.train()

            x, y = utils.create_batches(X_train, y_train, hparams_add["batchsize"], hparams_add["history"])
            
            x = torch.tensor(x).type(dtype=torch.float)
            y = torch.tensor(y).type(dtype=torch.float)
                
            output = model(x)
            
            # Compute training loss
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Evaluate current model at test set
            model.eval()
            
            with torch.no_grad():
                pred_train = model(X_train)
                pred_test = model(X_test)
                rmse_train[epoch] = utils.rmse(y_train, pred_train)
                rmse_val[epoch] = utils.rmse(y_test, pred_test)
                mae_train[epoch] = metrics.mean_absolute_error(y_train, pred_train)
                mae_val[epoch] = metrics.mean_absolute_error(y_test, pred_test)
    
    torch.save(model.state_dict(), os.path.join(data_dir, f"python\outputs\models\mlp7\\nodropout\sims_frac30\\tuned\setting2\model{i}.pth"))
    
    running_losses = {"rmse_train":rmse_train, "mae_train":mae_train, "rmse_val":rmse_val, "mae_val":mae_val}
    
    return running_losses, pred_test

#%%
def featureExtractorD(model, typ, epochs, simsfrac, splits = 5,
                      years = [2001,2002,2003, 2004, 2005, 2006, 2007],
                      data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    
    if ((typ == 9)| (typ == 10)):
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years, sims=False)
        model_design["featuresize"] = None
    else:
        hparams, model_design, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years)
        
    hparams_add, model_design_add, X, Y, X_test, Y_test = settings(model, epochs, data_dir, years = years, sims=False)
    
    X = torch.tensor(X).type(dtype=torch.float)
    X_test = torch.tensor(X_test).type(dtype=torch.float)

    errors = []
    preds_tests = []
    
    for i in range(splits):
        
        # Load pretrained model
        if ((typ == 9)| (typ == 10)):
            model = models.MLP(model_design["dimensions"], model_design["activation"])
        else:
            model = models.MLPmod(model_design["featuresize"], model_design["dimensions"], model_design["activation"])
            
        model.load_state_dict(torch.load(os.path.join(data_dir, 
                                              f"python\outputs\models\mlp{typ}\\nodropout\sims_frac{simsfrac}\model{i}.pth")))
        # modify classifier
        if ((typ == 9)| (typ == 10)):
            model = model[:-1]
        else:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # Remove Final layer and activation.
            
        # extract features
        out_train = model(X).detach().numpy()
        out_test = model(X_test).detach().numpy()
        # specify dimensions of model to train (architecture 2)
        model_design_add["dimensions"].insert(0,out_train.shape[1])
        
        # Train mlp with extracted features as input, predicting Y.
        running_losses, pred_test = train_model(hparams_add, model_design_add, out_train, Y, out_test, Y_test, i)
        
        # Evaluate model (reload it.)
        model = models.MLP(model_design_add["dimensions"], model_design_add["activation"])
        model.load_state_dict(torch.load(os.path.join(data_dir, 
                                              f"python\outputs\models\mlp7\\nodropout\sims_frac30\\tuned\setting2\model{i}.pth")))
        
        preds_test = model(torch.tensor(out_test).type(dtype=torch.float)).detach().numpy()
        preds_train = model(torch.tensor(out_train).type(dtype=torch.float)).detach().numpy()
        
        errors.append([utils.rmse(Y, preds_train), 
                       utils.rmse(Y_test, preds_test), 
                       metrics.mean_absolute_error(Y, preds_train),
                       metrics.mean_absolute_error(Y_test, preds_test)])
        preds_tests.append(preds_test)
        
    return(running_losses, errors, preds_test)
