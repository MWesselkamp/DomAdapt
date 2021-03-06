# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:46:14 2020

@author: marie
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from pylab import *

import setup.dev_rf as dev_rf

cols = sns.color_palette(palette="Paired")
#%%
def plot_running_losses(train_loss, val_loss, suptitle, model):

    #if model=="mlp":
    #    colors = ["blue","lightblue"]
    #elif model=="cnn":
    #    colors = ["darkgreen", "palegreen"]
    #elif model=="lstm":
    #    colors = ["blueviolet", "thistle"]
    #else:
    colors=["blue", "lightblue"]
    
    fig, ax = plt.subplots(figsize=(10,6))
    fig.suptitle(suptitle)

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        ax.fill_between(np.arange(len(train_loss)), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        ax.fill_between(np.arange(len(train_loss)), ci_val[0],ci_val[1], color="lightgreen", alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=0.8)
    ax.plot(val_loss, color="green", label = "Validation loss", linewidth=0.8)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set(xlabel="Epochs", ylabel="MAE")
    plt.ylim(bottom = 0)
    plt.rcParams.update({'font.size': 14})
    fig.legend(loc="upper left")
    

#%% SELECTED MODELS: PERFORMANCE

def losses(model, typ, suptitle,
                      data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"):

    datadir = os.path.join(data_dir, f"outputs\models\{model}{typ}")

    results = pd.read_csv(os.path.join(datadir, "selected_results.csv"))
    print(results)
    running_losses = np.load(os.path.join(datadir,"running_losses.npy"), allow_pickle=True).item()


    plot_running_losses(running_losses["mae_train"], running_losses["mae_val"], suptitle, model)
    #visualizations.plot_nn_predictions(y_tests, y_preds)
    #return(y_tests,y_preds)
    return(results)
    
#%%
def plot_nn_predictions(y_tests, y_preds):
    
    """
    Plot model predictions.
    """
    
    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle(f"Network Predictions")

    for i in range(len(y_tests)):
        ax.plot(y_tests[i], color="grey", label="Ground Truth", linewidth=0.9, alpha=0.6)
        ax.plot(y_preds[i], color="darkblue", label="Network Prediction", linewidth=0.9, alpha=0.6)
        #ax.plot(y_tests[i] - y_preds[i], color="lightgreen", label="absolute error", linewidth=0.9, alpha=0.6)
    
    #handles, labels = ax[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    
    
#%%
def plot_prediction_error(predictions, history, datadir, model):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    
    fig, ax = plt.subplots(len(predictions), figsize=(10,10))
    fig.suptitle(f"Network Prediction: Root Mean Squared Error (RMSE)")

    for i in range(len(predictions)):
        ax[i].plot(np.sqrt(np.square(predictions[i][0] - predictions[i][1])), color="green", label="rmse", linewidth=0.9, alpha=0.6)
        ax[i].set(xlabel="Day of Year", ylabel="RMSE")
    
    #handles, labels = ax[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    
#%%
def hparams_optimization_errors(results, model = "all", error = "rmse", train_val = False):
    """
    Scatterplot (_valtrain_erros).
    
    This function plots the training and validation errors of all models contained in the results.
    It should be applied after onto the results of hyperparametrization of all models (RF, CNN, MLP and LSTM) at once, in order to compare their performance.
    
    """
    if model == "RandomForest":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\plots\data_quality_evaluation\fits_rf"
    elif model == "mlp":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\plots\data_quality_evaluation\fits_nn\mlp"
    elif model == "convnet":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\plots\data_quality_evaluation\fits_nn\convnet"
    else:
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python\plots\data_quality_evaluation"

    fig, ax = plt.subplots()
    
    custom_xlim = (0, 3)
    custom_ylim = (0, 3)

    # Setting the values for all axes.
    plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

    if train_val:
        x = f"{error}_val"
        y = f"{error}_train"
        #fig.suptitle(f"Hyperparameter Optimization \n Training vs. Validation Error")
        data_dir = os.path.join(data_dir, f"_valtrain_errors_")
    else:
        x = "rmse_val"
        y = "mae_val"
        #fig.suptitle(f"Hyperparameter Optimization\n Validation Errors")
        data_dir = os.path.join(data_dir, f"_val_errors_")
        
    if isinstance(model, list):
        colors = [cols[1], cols[3], cols[5], cols[7]]
        #colors = ["blue", "darkgreen", "blueviolet", "gold"]
        #markers = ["o", "o", "*", "*"]
        models = ["MLP", "CNN", "LSTM", "RF"]
        for i in range(len(results)):
            ax.scatter(results[i][x], results[i][y], color=colors[i], 
                       #edgecolors = "black", 
                       alpha = 0.9, label=models[i])
    else:
        ax.scatter(results[x], results[y])
    
    plt.legend()
    if error == "rmse":
        error="RMSE"
    else:
        error="MAE"
        
    plt.xlabel(f"Validation {error}")
    plt.ylabel(f"Training {error}")
    #plt.savefig(data_dir)
    #plt.close()
    
#%%
def performance_boxplots(errors, running_losses_mlp, running_losses_conv, error = "rmse_val"):
    
    """
    This function returns a boxplot of the cross-validation training or validation errors (see argument error) after model selection.
    
    It compares the performance of all final models.
    
    """
    data_to_plot = [errors[error], np.mean(running_losses_mlp[error], axis=1), np.mean(running_losses_conv[error], axis=1) ]

    fig = plt.figure()
    #fig.suptitle(f"")
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, showmeans=True)

    ax.set_xticklabels(['RF', 'MLP', 'CNN'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("RMSE")

    for box in bp['boxes']:
        # change outline color
        box.set( color='black', linewidth=1)
        # change fill color
        #box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='orange', linewidth=2)
        x, y = median.get_xydata()[1] # top of median line
        # overlay median value
        text(x, y, '%.1f' % y,
        horizontalalignment='left') # draw above, centered

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.7)
                  
#%%
def performance(X, Y, models = ["rf", "mlp", "cnn"],
                data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt"):
    
    errors = np.load(os.path.join(data_dir,f"python\outputs\models\{models[0]}1\\errors.npy"), allow_pickle=True).item()
    running_losses_mlp = np.load(os.path.join(data_dir,f"python\outputs\models\{models[1]}1\\running_losses.npy"), allow_pickle=True).item()
    running_losses_cnn = np.load(os.path.join(data_dir,f"python\outputs\models\{models[2]}1\\running_losses.npy"), allow_pickle=True).item()

    performance_boxplots(errors, running_losses_mlp, running_losses_cnn)
    
#%%
def main(y_preds_rf, y_tests_rf, y_tests_nn, y_preds_nn, rfp, nnp, datadir):
    
    txt=f"RF: maximum grown trees ({rfp['n_trees']}), depth of trees ({rfp['depth']}).\n MLP: Hiddensize ({int(nnp['hiddensize'])}), Batchsize ({int(nnp['batchsize'])}), History ({int(nnp['history'])}), Learningrate ({nnp['learningrate']}) "
    fig, axs = plt.subplots(1, 2, figsize=(9,7))
    fig.suptitle("Model Selection Results (6-fold CV)")
    fig.text(.5, .01, txt, ha='center')
    for i in range(6):
        axs[0].plot(y_tests_rf[i][:,0], 'o', color="grey", alpha=0.7, markersize=2.5, label="Targets")
        axs[0].plot(y_preds_rf[i], color="darkred", alpha=0.5, linewidth = 0.7, label="Predictions")
        axs[0].set_ylabel(r"GPP [g C m$^{-2}$ day$^{-1}$]")
        axs[0].set_xlabel("DOY")
        axs[0].set_title(f"Random Forest \n (rmse = {np.round(rfp['rmse_val'], 4)})")
    for i in range(6):
        axs[1].plot(y_tests_nn[i][:,0], 'o', color="grey", alpha=0.7, markersize=2.5)
        axs[1].plot(y_preds_nn[i], color="darkblue", alpha=0.5, linewidth = 0.7)
        axs[1].set_xlabel("DOY")
        axs[1].set_title(f"Mulitlayer Perceptron \n (rmse = {np.round(nnp['rmse_val'], 4)})")
    plt.savefig(os.path.join(datadir, r"plots\data_quality_evaluation\_main_"))
    plt.close()
    