"""
Realiza a avaliação de performance nas previsões realizadas pela
composição
"""

import os
import joblib

import numpy as np
import pandas as pd
from performance_metrics import mean_bias_error
from sklearn.metrics import mean_squared_error, \
                            mean_absolute_error, \
                            r2_score


def get_n_stations_used(dataframe):
    """
    Obtem o número de estações de referencia utilizadas para realizar a
    previsão

    Parameters
    -------------
    dataframe : pd.DataFrame
        Dataframe contendo as previsões realizadas pela composição

    Returns
    ------------
    n_stations : int
        Número de estações utilizadas na interpolação para aquela estaçao de
        teste
    """

    return dataframe.iloc[:, 4:-2].shape[1]


def get_n_samples_used(dataframe):
    """
    Obtem o número de observações (samples) no conjunto de teste utilizado
    para medir a performance.

    Parameters
    -------------
    dataframe : pd.DataFrame
        Dataframe contendo as previsões realizadas pela composição

    Returns
    ------------
    n_samples : int
        Número de número de observações (samples) no conjunto de teste.
    """

    return dataframe.shape[0]


if __name__ == '__main__':

    comp_performance = {
        'station': [],
        'rmse': [],
        'mae': [],
        'mbe': [],
        'r2': [],
        'n_stations': [],
        'n_samples': [],
    }

    for pred_dataframe in os.listdir('./composition_compare'):

        composition_final = pd.read_csv(f'./composition_compare/{pred_dataframe}')

        real = composition_final[['real']].values
        pred = composition_final[['comp_pred']].values

        # coloca as informações no dicionario usado para construir o dataframe
        comp_performance['station'].append( pred_dataframe.split('.')[0] )
        comp_performance['rmse'].append( mean_squared_error(real, pred, squared = False) )
        comp_performance['mae'].append( mean_absolute_error(real, pred) )
        comp_performance['mbe'].append( mean_bias_error(real, pred) )
        comp_performance['r2'].append( r2_score(real, pred) )
        comp_performance['n_stations'].append( get_n_stations_used(composition_final) )
        comp_performance['n_samples'].append( get_n_samples_used(composition_final) )

    comp_performance_df = pd.DataFrame().from_dict(comp_performance)
    comp_performance_df.to_csv('./comp_performance_metrics.csv', index = False)

