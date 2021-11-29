"""
Realiza os testes do modelo de agregação utilizando IDW e os conjuntos de testes
gerados pelo script 'gen_test_sets'.
"""

import os
import argparse
from functools import reduce

import numpy as np
import pandas as pd

import haversine as hs
from haversine import Unit
from idw import InverseDistanceWeighting as IDW


def define_cli_args():

    arg_parser = argparse.ArgumentParser()

    # aeidionar argumento para a distancia máxima
    arg_parser.add_argument('-dst', '--max-distance',
                            metavar = '',
                            type = float,
                            default = 120,
                            help = 'Limite de distancia para interpolação dos valores. Default = 120Km')

    # mínimo de estações necessário, se tiver menos que isso descarta
    arg_parser.add_argument('-min', '--min-stations',
                            metavar = '',
                            type = int,
                            default = 3,
                            help = 'Nùmero mínimo de estaçoes para o cálculo da interpolação. Default = 3.')

    return arg_parser


def get_nearby_stations(pos: "(lat, lon)", max_dist, other_stations) -> (dict, list):
    """
    Obtem as estações que estão a uma distância de até 'max_dist' do ponto
    de referencia (pos) fornecido.

    Parameters
    -------------
    pos : tuple
        Posição da estação atual no formato (latitude, longitude)
    max_dist : float
        Threshold distance (em Km)

    Returns
    -----------
    nearby_stations : list
        Lista contendo os IDs das estações mais proximas ao ponto de referencia (pos).
    """

    nearby_stations = []
    distances = []

    for _, row in other_stations.iterrows():

        station_pos = ( row['lat'], row['lon'] )

        dist = hs.haversine(pos, station_pos, unit = Unit.KILOMETERS)

        # dist > 0 evita que a estação atual esteja presente nas estações mais proximas
        if dist <= max_dist and dist > 0:
            nearby_stations.append(row['station'])
            distances.append(dist)

    return distances, nearby_stations


def format_nearby_preds(nearby_preds):
    """
    Faz a junção dos dataframes de previsão colocando-os no formato

    DATA    HORA    DIA     MES     (prev ESTACAO1)  (prev ESTACAO1) ...
    --------------------------------------------------------------
    """

    dfs_to_merge = []

    for station_id, df in nearby_preds.items():

        df.rename(columns = {'pred': station_id}, inplace = True)
        df.drop(columns = ['lat', 'lon'], inplace = True)

        dfs_to_merge.append(df)

    formatted_neaby_preds = reduce(lambda left, right: pd.merge(left, right, how = 'outer',
                                                                on = ['year', 'month', 'day', 'hour']),
                                   dfs_to_merge)

    return formatted_neaby_preds


def get_estimator_errors(nearby_station_ids, estimators_info) -> list:

    nearby_station_errors = []

    for station in nearby_station_ids:
        nearby_station_errors.append( estimators_info.at[station, 'error'] )

    return nearby_station_errors


def idw_interpolate(pred_values, dists, errors):
    """
    Realiza a interpolação utilizando a técnica IDW

    Parameters
    ------------
    pred_values : array-like
        Array com as previsões realizadas (pode conter NAs)
    """

    valid_values_filter = ~np.isnan(pred_values)


    # remove os valores NA das previsões
    valid_pred_values = pred_values[ valid_values_filter ]

    # remove os valores NA das distancias
    distances = np.array(dists)
    valid_pred_distances = distances[ valid_values_filter ]

    errors = np.array(errors)
    valid_estimator_errors = errors[ valid_values_filter ]

    interpolator = IDW()

    for value, dist, error in zip(valid_pred_values, valid_pred_distances, valid_estimator_errors):
        interpolator.add_point(value = value, dist = dist, error = error)

    return interpolator.interpolate(method = 'custom')


if __name__ == '__main__':

    ap = define_cli_args()
    args = vars(ap.parse_args())

    if not os.path.exists('./composition_preds/'):
        os.mkdir('./composition_preds/')

    if not os.path.exists('./composition_compare/'):
        os.mkdir('./composition_compare/')

    # contem as informações sobre o erro
    estimators_info = pd.read_csv('./best_overall.csv')
    estimators_info = estimators_info.set_index('station')
    estimators_info['error'] = estimators_info[['rmse', 'mae']].mean(axis = 1)

    stations_info = pd.read_csv('./station_status.csv')

    test_stations = stations_info.loc[ stations_info['status'] == 'skip' ]
    train_stations = stations_info.loc[ stations_info['status'] == 'train' ]

    #
    #
    #
    # Para cada uma das estações (tanto de teste quando de treinamento) no PERIODO DE TESTES
    # gera as previsões da composição
    #
    #

    for _, row in stations_info.iterrows():

        station_pos = ( row['lat'], row['lon'] )

        distances, nearby_stations = get_nearby_stations(pos = station_pos,
                                                         max_dist = args['max_distance'],
                                                         other_stations = train_stations)

        errors = get_estimator_errors(nearby_station_ids = nearby_stations,
                                      estimators_info = estimators_info)

        # dicionario contendo dataframes com as previsões de cada uma das etações de treinamento proximas
        nearby_preds = {}

        for station_id in nearby_stations:
            nearby_preds[station_id] = pd.read_csv(f'./test_sets/preds/{station_id}.csv')

        formatted_nearby_preds = format_nearby_preds(nearby_preds)

        # o +4 vem dos 4 atributos de tempo que são utilizados em cada linha.
        # eles numca são NA então o valor mńimo de NAs deve ser o min_stations + 4 (do tempo)
        valid_nearby_preds = formatted_nearby_preds.dropna(thresh = args['min_stations'] + 4)

        # dataframe com as previsões da composição (datafarme final)
        composition_preds = valid_nearby_preds.copy()

        # pula os datafarmes que estão vazios (ou que poem ficar vazios dependendo dos
        # parametros de distancia e min_scations que são passados).
        if composition_preds.empty:
            continue

        composition_preds['comp_pred'] = composition_preds.iloc[:, 4:].apply(func = idw_interpolate,
                                                                             args = (
                                                                                 distances,
                                                                                 errors
                                                                             ),
                                                                             raw = True,
                                                                             axis = 'columns')

        composition_preds.to_csv(f'./composition_preds/{row["station"]}.csv')

        #
        #
        # Realiza o merge entre o dataframe de previsão e o dataframe real esperado
        # para aquela estação
        #
        #
        real_values = pd.read_csv(f'./test_sets/real/{row["station"]}.csv')

        composition_final = pd.merge(left = composition_preds,
                                     right = real_values,
                                     on = ['year', 'month', 'day', 'hour'],
                                     how = 'inner')

        # ultimo datafarme gerado por esse script
        if not composition_final.empty:
            composition_final.to_csv(f'./composition_compare/{row["station"]}.csv', index = False)

