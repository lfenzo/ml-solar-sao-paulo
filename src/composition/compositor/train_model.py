"""
Treina a rede neural utilizada para fazer a interpolação
"""

import os
import joblib
import optuna
import argparse
import numpy as np

from functools import partial

from optuna.samplers import TPESampler

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# funções objetivo para o optuna
from obj_funcs import objective_nn
from obj_funcs import objective_xgb


if __name__  == '__main__':

    ap = argparse.ArgumentParser(description = 'Otimização de Hipeparametros utilizando Optuna.')

    ap.add_argument('-t', '--timeout',
                    metavar = '',
                    type = int,
                    default = 5,
                    required = False,
                    help = 'Tempo de timeout (em minutos) para a execução da otimização. Defualt = 5')

    args = vars(ap.parse_args())


    # tempo (em segundos) durante o qual o optuna vai procurar hiperparametros
    OPTUNA_HP_SEARCH_TIMEOUT = args['timeout'] * 60

    # carrega dados de treinamento
    sX = np.load(file = './data/xtrain.npy')
    sY = np.load(file = './data/ytrain.npy')

    # carrega dados de teste
    sXt = np.load(file = './data/xtest.npy')
    sYt = np.load(file = './data/ytest.npy')

    target_scaler = joblib.load('./scalers/target_scaler.dat')
    data_scaler = joblib.load('./scalers/data_scaler.dat')


    #
    #
    # Busca de Hiperparametros com o optuna
    #
    #

    # redes neurais
    nn_study = optuna.create_study(direction = 'minimize', sampler = TPESampler())

    nn_study.optimize(
        func = partial(objective_nn, scaled_x = sX, scaled_y = sY, target_scaler = target_scaler),
        timeout = OPTUNA_HP_SEARCH_TIMEOUT,
    )

    nn_best_params = nn_study.best_params

    # ***EXTREME*** GRADIENT BOOSTING!!!
    xgb_study = optuna.create_study(direction = 'minimize', sampler = TPESampler())

    xgb_study.optimize(
        func = partial(objective_xgb, scaled_x = sX, scaled_y = sY, target_scaler = target_scaler),
        timeout = OPTUNA_HP_SEARCH_TIMEOUT,
    )

    xgb_best_params = xgb_study.best_params


    #
    #
    # Treinamento do modelo utilizando os melhores parametros
    #
    nn_reg = MLPRegressor(verbose = 1).set_params(**nn_best_params)
    xgb_reg = XGBRegressor().set_params(**xgb_best_params)

    model = StackingRegressor(
        estimators = [
            ('mlp', nn_reg),
            ('xgb', xgb_reg),
        ],
    )

    model.fit(sX, sY)

    # realiza a avaliação do modelo treinado utilizando os dados de teste
    pred = target_scaler.inverse_transform( model.predict(sXt).reshape(-1, 1) ).ravel()
    real = target_scaler.inverse_transform( sYt )

    mae = mean_absolute_error(real, pred)
    rmse = mean_squared_error(real, pred, squared = False)

    print('mae =', mae)
    print('rmse =', rmse)


    if not os.path.exists('./models'):
        os.mkdir('./models')

    # salva o modelo apos o treinamento
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    joblib.dump(value = model, filename = f'./models/{model.__class__.__name__}_{timestamp}.dat')

