"""
Gera os conjuntos de testes utilizados pelo script 'gen_comp_preds.py'. Os conjuntos
de testes são divididos em 2 tipos sendo eles:

    - Conjuntos de previsões: utilizam o formato (datetime, previsão) onde ha a associação
    entre uma timpestamp e a previsão realizda para uma estação de treinamento (no período
    de testes)

    - Conjuntos de valores reais: utilizam o formato (datetime, valor real) onde há a
    associação entre uma timestamp e o valor real esperado para aquela estação naquela
    hora específica.

Nota: para o conjunto de previsões são utilizados apenas as estações de treinamento, já no
conjunto de valores reais são utilizados tanto as estações de treinamento quanto as estaçes
não utilizadas no treinamento. A razão disto é aumentar a quantidade o tamanho do conjunto
de testes da composição.
"""

import os
import argparse

import joblib
import numpy as np
import pandas as pd
import haversine as hs

from utils import preprocess_batch, get_best_model_filepaths


def define_cli_args():

    ap = argparse.ArgumentParser(description = 'Gera os conjuntos de previsões emparelhados com das datas')

    ap.add_argument('-v', '--verbose',
                    metavar = '',
                    required = False,
                    default = 0,
                    type = int)

    return ap


if __name__ == '__main__':

    ap = define_cli_args()
    args = vars(ap.parse_args())

    # cria o diretório onde os conjuntos de teste para cada uma das estacções de teste serão criados
    if not os.path.exists('./test_sets'):
        os.mkdir('./test_sets')

    training_stations = pd.read_csv('../stations/best_overall.csv')


    # obtem o codigo das estações que foram utilizadas no treinamento (apenas as 30)
    training_stations_ids = training_stations['station'].unique()

    # obtem o codigo de todas as estações processadas (todas as 56 estçaões)
    all_stations_ids = list(map(lambda item: item.split('.')[0], os.listdir('../data/estacoes_processadas/')))

    #test_station_ids = list( set(all_stations_ids) - set(training_stations_ids) )
    test_station_ids = all_stations_ids

    #
    #
    # gera os conjuntos de dados de previsões de todas as esstações
    # utilizadas para treinamento. estes testes são referentes ao período de
    # 2019 até 2021
    #
    # previsão : apenas estações de treinamento

    # dicionário contendo os caminhos para os modelos
    models = get_best_model_filepaths(training_stations)

    if not os.path.exists('./test_sets/preds'):
        os.mkdir('./test_sets/preds')

    for train_station in training_stations_ids:

        # obtem as informações de localização que serão colocadas ao lado das previsões
        station_metadata = pd.read_csv(f'../stations/{train_station}/{train_station}_metadata.csv')

        station_lat = station_metadata.at[0, 'latitude']
        station_lon = station_metadata.at[0, 'longitude']

        # deve armazenaar as informações que serão utilizaads para construir o dataframe
        station_test_data = {
            'hour': [],
            'day': [],
            'month': [],
            'year': [],
            'pred': []
        }

        # as informações aqui já serão obtidas depois de fazer a mixagem dos melhores modelos
        # misturando dados originais e interpolação
        station_data = pd.read_csv(f'../data/estacoes_processadas/{train_station}.csv')

        model = joblib.load( models[train_station] )

        scaler_data = joblib.load(f'../stations/{train_station}/{train_station}_scaler_data.dat')
        scaler_target = joblib.load(f'../stations/{train_station}/{train_station}_scaler_target.dat')

        selected_train_period_data = station_data.loc[ station_data['year'].isin(range(2019, 2022)) ]

        if args['verbose']:
            print('Gerando conjunto de previsões da estaçao: ', train_station)

        kwargs = {
            'data': selected_train_period_data,
            'scaler_data': scaler_data,
            'scaler_target': scaler_target,
        }

        timestamps, scaled_data_batch = preprocess_batch(**kwargs)

        raw_prediction = model.predict(scaled_data_batch)

        preds = scaler_target.inverse_transform(raw_prediction.reshape(1, -1))

        final_prediction_set = timestamps.copy()
        final_prediction_set['pred'] = preds.reshape(-1, 1)

        final_prediction_set['lat'] = station_lat
        final_prediction_set['lon'] = station_lon

        final_prediction_set.to_csv(f'./test_sets/preds/{train_station}.csv', index = False)


    #
    #
    # Obtem os dados reais no formato (datetime, valor esperado)
    #
    # São utilizados tanto dados do periodo de teste (2019 até 2021) das estaçooes de
    # treinamento quanto de teste
    #
    # teste: todas as estaçoes

    if not os.path.exists('./test_sets/real'):
        os.mkdir('./test_sets/real')

    for station in all_stations_ids:

        if args['verbose']:
            print(f'Obtendo informações reais na estação: {station}.')

        station_data = pd.read_csv(f'../data/estacoes_processadas/{station}.csv')

        # filtra o dataframe para selecionar apenas o período de teste
        selected_test_period = station_data.loc[ station_data['year'].isin(range(2019, 2022)) ]

        test_station_df = selected_test_period[['datetime', 'rad_prox_hora']].copy()

        # troca o nome do atributo alvo
        test_station_df.rename(columns = {'rad_prox_hora': 'real'}, inplace = True)

        # igual ao feito na funçao 'preprocess_batch()'
        test_station_df['year']  = test_station_df['datetime'].str[ 0:4 ]
        test_station_df['month'] = test_station_df['datetime'].str[ 5:7 ]
        test_station_df['day']   = test_station_df['datetime'].str[ 8:10]
        test_station_df['hour']  = test_station_df['datetime'].str[11:13]

        test_station_df.drop(columns = ['datetime'], inplace = True)
        test_station_df.reset_index(drop = True, inplace = True)

        test_station_df.to_csv(f'./test_sets/real/{station}.csv', index = False)

