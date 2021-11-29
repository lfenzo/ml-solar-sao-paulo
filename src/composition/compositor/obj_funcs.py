"""
Definições de funções objetivo para busca de hiperparametros
utilizando o Optuna
"""

import os
import joblib
import optuna
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


def objective_nn(trial, scaled_x, scaled_y, target_scaler):
    """
    Função objetivo para hiperparametrização das Redes Neurais
    """

    xtrain, xvalid, ytrain, yvalid = train_test_split(scaled_x, scaled_y,
                                                      train_size = 0.75)

    # definições de atributos customizados
    trial.set_user_attr('tol', 1e-5)
    trial.set_user_attr('max_iter', 500)
    trial.set_user_attr('hidden_layer_sizes', (100, 100, 40))

    param_grid = {
        'solver': trial.suggest_categorical('solver', ['sgd']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 0.5),
        'momentum': trial.suggest_float('momentum', 0.25, 0.9999),
        'power_t': trial.suggest_float('power_t', 0.2, 0.8),
        'alpha': trial.suggest_float('alpha', 1e-5, 0.5),

        'hidden_layer_sizes': trial.user_attrs['hidden_layer_sizes'],
        'max_iter': trial.user_attrs['max_iter'],
        'tol': trial.user_attrs['tol'],
    }

    model = MLPRegressor().set_params(**param_grid)
    model.fit(xtrain, ytrain)

    pred = target_scaler.inverse_transform( model.predict(xvalid).reshape(-1, 1) ).ravel()
    real = target_scaler.inverse_transform( yvalid )

    return mean_squared_error(pred, real, squared = False)


def objective_xgb(trial, scaled_x, scaled_y, target_scaler):
    """
    Função objetivo para hiperparametrização do Xtreme Gradient Boosting
    """

    xtrain, xvalid, ytrain, yvalid = train_test_split(scaled_x, scaled_y,
                                                      train_size = 0.75)

    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
        'eta': trial.suggest_float('eta', 0.0001, 0.5),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'subsample': trial.suggest_float('subsample', 0.00001, 1),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 5),
    }

    model = XGBRegressor().set_params(**param_grid)
    model.fit(xtrain, ytrain)

    pred = target_scaler.inverse_transform( model.predict(xvalid).reshape(-1, 1) ).ravel()
    real = target_scaler.inverse_transform( yvalid )

    return mean_squared_error(pred, real, squared = False)
