# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:22:10 2020

@author: marie
"""
#%% Set working directory
import os
os.getcwd()
os.chdir('OneDrive\Dokumente\Sc_Master\Masterthesis\Project\DomAdapt')

import preprocessing
#import utils
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

#%% Train the Algorithm

def random_forest_CV(X, Y, splits, shuffled):
    
    X = preprocessing.normalize_features(X)
    
    # Divide into training and test
    kf = KFold(n_splits=splits, shuffle = shuffled)
    kf.get_n_splits(X)
    regressor = RandomForestRegressor(n_estimators=1000, max_depth  = 5, criterion = "mse")

    rmse_train = np.zeros((splits))
    mae_train = np.zeros((splits))
    rmse_test = np.zeros((splits))
    mae_test = np.zeros((splits))

    y_preds = []
    y_trains = []
    y_tests = []

    i = 0
    for train_index, test_index in kf.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_tests.append(y_test)
        y_trains.append(y_train)
    
        regressor.fit(X_train, y_train.ravel())
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        y_preds.append(y_pred_test)
    

        # Evaluate the algorithm
        rmse_test[i] = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
        mae_test[i] = metrics.mean_absolute_error(y_test, y_pred_test)
        rmse_train[i] = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
        mae_train[i] = metrics.mean_absolute_error(y_train, y_pred_train)
        #yrange = np.ptp(y_train, axis=0)
        #mpe[i] = utils.percentage_error(y_test, y_pred, y_range=yrange)

        #print('Root Mean Squared Error Test:', rmse_test[i])
        #print('Mean Absolute Error Test:', mae_test[i])
        #print('Root Mean Squared Error Training:', rmse_train[i])
        #print('Mean Absolute Error Training:', mae_train[i])

        i+= 1
    
    #print('Mean Percentage Error:', np.mean(mpe))
    print('Root Mean Squared Error Training:', np.mean(rmse_train))
    print('Root Mean Squared Error Test:', np.mean(rmse_test))
    print('Mean Absolute Error Training:', np.mean(mae_train))
    print('Mean Absolute Error Test:', np.mean(mae_test))

#%%
def random_forest_fit(X_train, X_test, y_train, y_test, df = True):
    
    regressor = RandomForestRegressor(n_estimators=1000, max_depth  = 5, criterion = "mse")
    
    if df:
        regressor.fit(X_train, y_train.ravel())
    else:
        regressor.fit(X_train, y_train)
    
    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    
    # Evaluate the algorithm
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    print('Root Mean Squared Error Training:', rmse_train)
    print('Root Mean Squared Error Test:', rmse_test)
    print('Mean Absolute Error Training:', mae_train)
    print('Mean Absolute Error Test:', mae_test)
    
    return y_pred_test, y_pred_train, [rmse_train, rmse_test, mae_train, mae_test]

#%%
def plot_rf_fit(Y_train, Y_test, data):
    
    fig, ax = plt.subplots(2, figsize=(10,9))
    fig.suptitle(f"Random Forest Fit: {data} data \n RMSE Training = {np.round(errors[0], 4)}, RSME Test = {np.round(errors[1], 4)} \n MAE Training = {np.round(errors[2], 4)}, MAE Test = {np.round(errors[3], 4)}")
    ax[0].plot(Y_train, color="gray", label="Observations", linewidth=0.8)
    ax[0].plot(y_pred_train, color="darkblue", label="Predictions (train)", linewidth=0.8)
    ax[0].plot(Y_train.flatten() - y_pred_train, color="lightgreen", linewidth=0.6)
    ax[1].plot(Y_test, color="gray", linewidth=0.8)
    ax[1].plot(y_pred_test, color="lightblue", label="Predictions (test)", linewidth=0.8)
    ax[1].plot(Y_test.flatten() - y_pred_test, color="lightgreen", label="Error", linewidth=0.6)
    for a in ax.flat:
        a.set(xlabel="Time [days]", ylabel=r"GPP [g C m$^{-2}$ day$^{-1}$]")
    fig.legend(loc="upper left")
    
#%% Load data
X_profound, Y_profound = preprocessing.get_profound_data(dataset="trainval", data_dir = r'data\profound', to_numpy = True, simulation=False)
X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', to_numpy = True, preles=False)

# Merge profound and preles data into one large data set.
X_both = np.concatenate((X_profound, X_borealsites), axis=0)
Y_both = np.concatenate((Y_profound, Y_borealsites), axis=0)

#%% Crossvalidation
# Profound data
random_forest_CV(X_profound, Y_profound, splits=5, shuffled = False)
random_forest_CV(X_profound, Y_profound, splits=5, shuffled = True)
random_forest_CV(X_profound, Y_profound, splits=10, shuffled = False)
random_forest_CV(X_profound, Y_profound, splits=10, shuffled = True)
# Boreal sites
random_forest_CV(X_borealsites, Y_borealsites, splits=5, shuffled = False)
random_forest_CV(X_borealsites, Y_borealsites, splits=5, shuffled = True)
random_forest_CV(X_borealsites, Y_borealsites, splits=10, shuffled = False)
random_forest_CV(X_borealsites, Y_borealsites, splits=10, shuffled = True)
# Both
random_forest_CV(X_both, Y_both, splits=5, shuffled = False)
random_forest_CV(X_both, Y_both, splits=5, shuffled = True)
random_forest_CV(X_both, Y_both, splits=10, shuffled = False)
random_forest_CV(X_both, Y_both, splits=10, shuffled = True)


#%% Fit RF : Profound data
X_profound, Y_profound = preprocessing.get_profound_data(dataset="trainval", data_dir = r'data\profound', to_numpy = False, simulation=False)

train = X_profound['site'] != "collelongo"
test = X_profound['site'] == "collelongo"
X_train_pro = X_profound[train].drop(columns='site').to_numpy()
X_test_pro = X_profound[test].drop(columns='site').to_numpy()
Y_train_pro = Y_profound[train].to_numpy()
Y_test_pro = Y_profound[test].to_numpy()

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_pro, X_test_pro, Y_train_pro, Y_test_pro)

#Plot
plot_rf_fit(Y_train_pro, Y_test_pro, data="Profound")

#%% Fit RF : Borealsites data
X_borealsites, Y_borealsites = preprocessing.get_borealsites_data(data_dir = r'data\borealsites', to_numpy = False, preles=False)

train = X_borealsites['site'] != "kalevansuo"
test = X_borealsites['site'] == "kalevansuo"
X_train_bor = X_borealsites[train].drop(columns='site').to_numpy()
X_test_bor = X_borealsites[test].drop(columns='site').to_numpy()
Y_train_bor = Y_borealsites[train].to_numpy()
Y_test_bor = Y_borealsites[test].to_numpy()

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_bor, X_test_bor, Y_train_bor, Y_test_bor)

#Plot
plot_rf_fit(Y_train_bor, Y_test_bor, data="Borealsites")

#%% Fit RF : Both data
X_train_both = np.concatenate((X_train_pro, X_train_bor), axis=0)
X_test_both = np.concatenate((X_test_pro, X_test_bor), axis=0)
Y_train_both = np.concatenate((Y_train_pro, Y_train_bor), axis=0)
Y_test_both = np.concatenate((Y_test_pro, Y_test_bor), axis=0)

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_both, X_test_both, Y_train_both, Y_test_both)

#Plot
plot_rf_fit(Y_train_both, Y_test_both, data="Profound+Borealsites")

#%% Fit RF : Profound data / Only Hyytiala
hyytiala = (X_profound['site'] == "hyytiala")
X_hyytiala = X_profound[hyytiala].drop(columns='site').to_numpy()
Y_hyytiala = Y_profound[hyytiala].to_numpy()

X_train_hyytiala_pro, X_test_hyytiala_pro, Y_train_hyytiala_pro, Y_test_hyytiala_pro = train_test_split(X_hyytiala, Y_hyytiala, test_size=0.3, shuffle=True)

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_hyytiala_pro, X_test_hyytiala_pro, Y_train_hyytiala_pro, Y_test_hyytiala_pro, df=False)

#Plot
plot_rf_fit(Y_train_hyytiala_pro, Y_test_hyytiala_pro, data="Hyytiala Profound")

#%% Fit RF : Borealsites data / Only Hyytiala
hyytiala = (X_borealsites['site'] == "hyytiala")
X_hyytiala = X_borealsites[hyytiala].drop(columns='site').to_numpy()
Y_hyytiala = Y_borealsites[hyytiala].to_numpy()

X_train_hyytiala_bor, X_test_hyytiala_bor, Y_train_hyytiala_bor, Y_test_hyytiala_bor = train_test_split(X_hyytiala, Y_hyytiala, test_size=0.5, shuffle=True)

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_hyytiala_bor, X_test_hyytiala_bor, Y_train_hyytiala_bor, Y_test_hyytiala_bor, df=False)

#Plot
plot_rf_fit(Y_train_hyytiala_bor, Y_test_hyytiala_bor, data="Hyytiala Borealsites")

#%% Fit RF : Profound and Borealsites data / Only Hyytiala
X_train_both = np.concatenate((X_train_hyytiala_pro, X_train_hyytiala_bor), axis=0)
X_test_both = np.concatenate((X_test_hyytiala_pro, X_test_hyytiala_bor), axis=0)
Y_train_both = np.concatenate((Y_train_hyytiala_pro, Y_train_hyytiala_bor), axis=0)
Y_test_both = np.concatenate((Y_test_hyytiala_pro, Y_test_hyytiala_bor), axis=0)

y_pred_test, y_pred_train, errors = random_forest_fit(X_train_both, X_test_both, Y_train_both, Y_test_both)

#Plot
plot_rf_fit(Y_train_both, Y_test_both, data="Hyytiala Profound+Borealsites")
