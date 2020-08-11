# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:46:14 2020

@author: marie
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

#%%
#%%
def plot_nn_loss(train_loss, val_loss, hparams, figure = "", model = "convnet"):
    
    data_dir = os.path.join(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn", model)
    
    fig, ax = plt.subplots(figsize=(10,6))
    fig.suptitle(f"Fully connected Network \n Epochs = {hparams['epochs']}, Shuffled_CV = {hparams['shuffled_CV']}, History = {hparams['history']} \n Hiddensize = {hparams['hiddensize']}, Batchsize = {hparams['batchsize']}, Learning_rate = {hparams['learningrate']}")

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        ax.fill_between(np.arange(hparams["epochs"]), ci_train[0],ci_train[1], color="lightgreen", alpha=0.3)
        ax.fill_between(np.arange(hparams["epochs"]), ci_val[0],ci_val[1], color="lightblue", alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    ax.plot(train_loss, color="green", label="Training loss", linewidth=0.8)
    ax.plot(val_loss, color="blue", label = "Validation loss", linewidth=0.8)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set(xlabel="Epochs", ylabel="Root Mean Squared Error")
    plt.ylim(bottom = 0)
    fig.legend(loc="upper left")

    plt.savefig(os.path.join(data_dir, f"_loss_{figure}"))
    plt.close()

#%%
def plot_nn_predictions(y_tests, y_preds, history, figure = "", model = "convnet"):
    
    """
    Plot model predictions.
    """
    
    data_dir = os.path.join(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn", model)
    
    fig, ax = plt.subplots(len(y_tests), figsize=(10,10))
    fig.suptitle(f"Network Predictions")

    for i in range(len(y_tests)):
        ax[i].plot(y_tests[i], color="grey", label="targets", linewidth=0.9, alpha=0.6)
        ax[i].plot(y_preds[i], color="darkblue", label="nn predictions", linewidth=0.9, alpha=0.6)
        ax[i].plot(y_tests[i] - y_preds[i], color="lightgreen", label="absolute error", linewidth=0.9, alpha=0.6)
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(data_dir, f"_predictions_{figure}"))
    plt.close()
    
#%%
def plot_prediction_error(predictions, history, figure = "", model="convnet"):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    data_dir = os.path.join(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn", model)

    fig, ax = plt.subplots(len(predictions), figsize=(10,10))
    fig.suptitle(f"Network Prediction: Root Mean Squared Error (RMSE)")

    for i in range(len(predictions)):
        ax[i].plot(np.sqrt(np.square(predictions[i][0] - predictions[i][1])), color="green", label="rmse", linewidth=0.9, alpha=0.6)
        ax[i].set(xlabel="Day of Year", ylabel="RMSE")
    
    #handles, labels = ax[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(data_dir, f"_rmse_{figure}"))
    plt.close()
    
#%%
def plot_validation_errors(results, model, train_val = False, annotate = False):
    
    if model == "RandomForest":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_rf"
    elif model == "mlp":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn\mlp"
    elif model == "convnet":
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\fits_nn\convnet"
    else:
        data_dir = r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation"

    fig, ax = plt.subplots()
    
    
    if train_val:
        x = "rmse_val"
        y = "rmse_train"
        fig.suptitle(f"Hyperparameter Optimization (6-fold CV) \n RSME Errors ")
        data_dir = os.path.join(data_dir, f"_valtrain_errors_")
    else:
        x = "rmse_val"
        y = "mae_val"
        fig.suptitle(f"Hyperparameter Optimization (6-fold CV) \n Validation Errors ")
        data_dir = os.path.join(data_dir, f"_val_errors_")
        
    if isinstance(model, list):
        ax.scatter(results[0][x], results[0][y], color="blue", label=model[0])
        ax.scatter(results[1][x], results[1][y], color="green",label=model[1])
    else:
        ax.scatter(results[x], results[y])
    
    if annotate:
        for i, txt in enumerate(results["run"]):
            ax.annotate(txt, (results["rmse_val"][i], results["mae_val"][i]))
    
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(data_dir)
    plt.close()
    
#%%
def main(y_preds_rf, y_tests_rf, y_tests_nn, y_preds_nn, rfp, nnp):
    
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
    plt.savefig(r"OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt\plots\data_quality_evaluation\_main_")
    plt.close()
    