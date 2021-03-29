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
from sklearn import metrics

import setup.dev_rf as dev_rf
from collections import OrderedDict

cmaps = OrderedDict()
cols = sns.color_palette(palette="Paired")
#%%
def plot_running_losses(train_loss, val_loss, legend, plot_train_loss,
                        colors=["blue", "lightblue"],
                        colors_test_loss = ["green","lightgreen"]):

    #if model=="mlp":
    #    colors = ["blue","lightblue"]
    #elif model=="cnn":
    #    colors = ["darkgreen", "palegreen"]
    #elif model=="lstm":
    #    colors = ["blueviolet", "thistle"]
    #else:

    
    fig, ax = plt.subplots(figsize=(7,7))

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        if plot_train_loss:
            ax.fill_between(np.arange(len(train_loss)), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        ax.fill_between(np.arange(len(train_loss)), ci_val[0],ci_val[1], color=colors_test_loss[1], alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    if plot_train_loss:
        ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=1.2)
        ax.plot(val_loss, color="green", label = "Test loss", linewidth=1.2)
    else:
        ax.plot(val_loss, color=colors_test_loss[0], label = "Test loss\nfull re-training", linewidth=1.2)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20)
    ax.set_xlabel("Epochs", size=20)
    #plt.ylim(bottom = 0.0)
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    plt.rcParams.update({'font.size': 20})
    if legend:
        fig.legend(loc="upper right")
    

#%% SELECTED MODELS: PERFORMANCE

def losses(model, typ, searchpath, error = "mae", plot = True, legend=True, sparse =False, plot_train_loss=True,
                      data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"):
    
    if sparse:
        data_dir = os.path.join(data_dir, f"outputs\sparse\models\{model}{typ}")
        data_dir = os.path.join(data_dir, searchpath)
    else:
        data_dir = os.path.join(data_dir, f"outputs\models\{model}{typ}")
        data_dir = os.path.join(data_dir, searchpath)

    results = pd.read_csv(os.path.join(data_dir, "selected_results.csv"))
    print(results)
    running_losses = np.load(os.path.join(data_dir,"running_losses.npy"), allow_pickle=True).item()

    if plot:
        if error =="mae":
            plot_running_losses(running_losses["mae_train"], running_losses["mae_val"],  legend, plot_train_loss)
        else:
            plot_running_losses(running_losses["rmse_train"], running_losses["rmse_val"],  legend, plot_train_loss)
    #visualizations.plot_nn_predictions(y_tests, y_preds)
    #return(y_tests,y_preds)
    return(results)
    

#%%
def plot_prediction(y_tests, predictions, mae):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    
    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.plot(y_tests, color="lightgrey", label="Ground truth", marker = "o", linewidth=0.8, alpha=0.9, markerfacecolor='lightgrey', markersize=4)
    try:
        ci_preds = np.quantile(np.array(predictions), (0.05,0.95), axis=0)
        m_preds = np.mean(np.array(predictions), axis=0)
    
        ax.fill_between(np.arange(len(ci_preds[0])), ci_preds[0],ci_preds[1], color="lightgreen", alpha=0.9)
        ax.plot(m_preds, color="green", label="Predictions", marker = "", alpha=0.5)
    except:
        print("Plotting Preles Predictions.")
        ax.plot(predictions, color="green", label="Predictions", marker = "", alpha=0.5)
        mae = metrics.mean_absolute_error(y_tests, predictions)
        
    errors = np.subtract(np.transpose(np.array(predictions)), y_tests)
    ci_preds = np.transpose(np.quantile(errors, (0.05,0.95), axis=1))
    m_errors = np.mean(errors, axis=1)
    ax.fill_between(np.arange(errors.shape[0]), ci_preds[:,0],ci_preds[:,1], color="lightsalmon", alpha=0.9)
    ax.plot(m_errors, color="red", label="Error", marker = "", alpha=0.5)
        
    ax.set_xlabel("Day of Year", size=20, family='Palatino Linotype')
    ax.set_ylabel("Gross primary produdction [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    
    ax.legend(loc="upper left", prop={'size':20, 'family':'Palatino Linotype'})
    ax.set_ylim((-4,14.2))
    
    plt.text(250, 13.0, f"MAE = {mae: .4f}", family='Palatino Linotype', size=20)
#%%
def scatter_prediction(y_tests, predictions, corr):
    
    fig, ax = plt.subplots(figsize=(7,7))
    r2 = []
    for i in range(5):
        rss = np.sum(np.square(y_tests - predictions[i]))
        tss = np.sum(np.square(y_tests - np.mean(y_tests)))
        r2.append((1-(rss/tss)))
        #plt.scatter(np.arange(366), res)
        plt.scatter(predictions[i], y_tests, color="gray", alpha=0.6)
        plt.plot(np.arange(12), np.arange(12), color="red")
    
    ax.set_xlabel("Predicted gross primary produdction [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    ax.set_ylabel("Gross primary produdction [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')

    plt.text(0, 11.0, f"Pearson's r = {corr: .4f}", family='Palatino Linotype', size=20)
    plt.text(0, 10.0, f"R$^2$ = {np.mean(r2): .4f}", family='Palatino Linotype', size=20)
    
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

    fig, ax = plt.subplots(figsize=(7,7))
    
    if train_val:
        custom_xlim = (0, 3.0)
        custom_ylim = (0, 3.0)

        #Setting the values for all axes.
        plt.setp(ax, xlim=custom_xlim, ylim = custom_ylim)
    else:
        
        custom_xlim = (1.2, 3.5)
        custom_ylim = (1.2, 3.5)

        #Setting the values for all axes.
        plt.setp(ax, xlim=custom_xlim, ylim = custom_ylim)

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
    
    cols = ["gray","green", "darkred", "steelblue", "orange", "purple"]
    markers = ['o', 'o', 'o', 'o', 'o', '*']
    if isinstance(model, list):
        colors = [cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]]
        models = model
        for i in range(len(results)):
            ax.scatter(results[i][x], results[i][y], color=colors[i],marker =markers[i],
                       alpha = 0.8, label=models[i], s=70)
    else:
        ax.scatter(results[x], results[y])
    
    if train_val:
        plt.legend(loc="upper left", ncol=1, prop= {'size':22, 'family':'Palatino Linotype'})
        if error == "rmse":
            error="RMSE"
        else:
            error="MAE"
        
        plt.xlabel(f"Mean absolute test error [g C m$^{-2}$ day$^{-1}$]", size=22, family='Palatino Linotype')
        plt.ylabel(f"Mean absolute training error [g C m$^{-2}$ day$^{-1}$]", size=22, family='Palatino Linotype')
        plt.xticks(size=22, family='Palatino Linotype')
        plt.yticks(size=22, family='Palatino Linotype')
    else:
        plt.legend(loc="upper left", ncol=2 , prop= {'size':22, 'family':'Palatino Linotype'})
        if error == "rmse":
            error="RMSE"
        else:
            error="MAE"
        
        plt.xlabel(f"Test RMSE", size=22, family='Palatino Linotype')
        plt.ylabel(f"Test MAE", size =22, family='Palatino Linotype')
        plt.xticks(size=22, family='Palatino Linotype')
        plt.yticks(size=22, family='Palatino Linotype')
    #plt.savefig(data_dir)
    #plt.close()
    
#%%
def performance_boxplots(typ, error = "mae_val",
                         data_dir = "OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\python"):
    
    """
    This function returns a boxplot of the cross-validation training or validation errors (see argument error) after model selection.
    
    It compares the performance of all final models.
    
    """
    
    running_losses_mlp = np.load(os.path.join(data_dir,f"outputs\models\mlp{typ}\hyytiala\\running_losses.npy"), allow_pickle=True).tolist()
    running_losses_cnn = np.load(os.path.join(data_dir,f"outputs\models\cnn{typ}\hyttiala\\running_losses.npy"), allow_pickle=True).tolist()
    running_losses_lstm = np.load(os.path.join(data_dir, f"outputs\models\lstm{typ}\hyytiala\\running_losses.npy"), allow_pickle=True).tolist()
    errors = np.load(os.path.join(data_dir,f"outputs\models\\rf{typ}\errors.npy"), allow_pickle=True).tolist()
    
    data_to_plot = [[sublist[-1] for sublist in running_losses_mlp["mae_val"]], 
                    [sublist[-1] for sublist in running_losses_cnn["mae_val"]], 
                    [sublist[-1] for sublist in running_losses_lstm["mae_val"]],
                    errors[3]]

    fig = plt.figure()
    #fig.suptitle(f"")
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, showmeans=True)

    ax.set_xticklabels(['MLP', 'CNN', 'LSTM', 'RF'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAE")

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
    