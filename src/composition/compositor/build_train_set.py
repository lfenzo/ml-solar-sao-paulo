"""
Gera o conjunto de dados que será utilizado no treinamento da neural
responsável por fazer a agregação das previsões.

Conjunto de dados de treinamento tem o seguinte formato;

                          estação1               estação2
                      ---------------        ---------------
                     /               \      /               \
DOY  HOUR  LAT LON   V1     V2      V3      V4      V5      V6 ... ...  Valor esperado
19   10    la  lo   dist   error   prev    dist   error   prev                Y

Quando não houver estações os valores devem ser preenchidos todos com zero,
os valores não nulos ficam semmpre à esqurda.

Nos conjuntos 'pred_compare' gerados com os parametros de distância de 120km e
minimo de 3 estações

Processa cada uma das estações colocando no formato especificado acima e junta
(faz um concat vertical) de todas as estações para a criação do conjunto de
dados de treinamento e teste.

Para separação entre os conjuntos de treinamento e dados, para evitar bias na
separação aleatoria que poderia ocorrer foi feito o seguinte (com o conjunto
de dados já ordenado):

Amostra 1   \
Amostra 2   | Treianmento
Amostra 3   |
Amostra 4   /
Amostra 5   -- Teste

Amostra N -> N % 5 == 0 -> teste
Amostra N -> N % 5 != 0 -> treinamento
"""

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import haversine as hs

from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler
from haversine import Unit


if __name__ == '__main__':

    stations_info = pd.read_csv('../station_status.csv')
    estimators_info = pd.read_csv('../best_overall.csv')

    # estações que podem ser utilizadas na interpolação
    # que tem pelo menos 3 estações num ario de 120 km
    compare_pred_files = list(map(lambda file: file.split('.')[0], os.listdir('../composition_compare/')))

    # informações de localização dos modelos e das métricas de desempenho
    info = pd.merge(right = stations_info,
                    left = estimators_info,
                    on = 'station')

    info = info.set_index('station', drop = True)

    # metrica de erro (media entre rmse e mae)
    info['error'] = (info['rmse'] + info['mae']) / 2


    test_sets = []
    train_sets = []


    for k, station_id in enumerate(info.index):

        print(f'Processando estação {station_id}, estação {k + 1}/{len(info.index)}')

        # comparação entre as previsões das estações mais proximas
        # e os valores reais esperados nesses locais
        try:
            pred_compare = pd.read_csv(f'../composition_compare/{station_id}.csv')
            pred_compare = pred_compare.fillna(value = 0)
        except FileNotFoundError:
            continue

        # posição da estação atual (lat, lon)
        current_station_pos = ( info.at[station_id, 'lat'], info.at[station_id, 'lon'] )

        # obtem quais são as estações mais proximas 
        nearby_station_ids = pred_compare.iloc[:, 4:-2].columns.to_list()


        nearby_info = {}

        for nearby_station_id in nearby_station_ids:

            nearby_station_pos = (
                info.at[nearby_station_id, 'lat'],
                info.at[nearby_station_id, 'lon']
            )

            nearby_info[nearby_station_id] = {
                'dist': hs.haversine(current_station_pos, nearby_station_pos, unit = Unit.KILOMETERS),
                'error': info.at[nearby_station_id, 'error'],
            }

        # ordena as estações com base no erro
        nearby_info = dict(sorted(nearby_info.items(), key = lambda item: item[1]['error']))

        # atributos do conjunto de dados de treinamento do compositor
        features = {
            'target': [],

            # atributos de tempo
            'doy':  [],
            'hour': [],

            # atributos de locaização
            'lat': [],
            'lon': [],
        }

        # atributos sobre informações de cada uma das estaçoes 
        for n in range(4):
            features[ f'stat{n + 1}_dist'  ] = []
            features[ f'stat{n + 1}_error' ] = []
            features[ f'stat{n + 1}_pred'  ] = []

        # coloca as informações no formato especificado na dox string deste script
        for _, row in pred_compare.iterrows():

            day   = str(row["day"]).split('.')[0]
            month = str(row["month"]).split('.')[0]
            year  = str(row["year"]).split('.')[0]

            features['doy'].append( pd.Period(f'{year}-{month}-{day}').dayofyear )
            features['hour'].append(row['hour'])
            features['lat'].append(info.at[station_id, 'lat'])
            features['lon'].append(info.at[station_id, 'lon'])
            features['target'].append(row['real'])

            # valores NA foram substituidos por 0s então não é para usar as estações que tem 0s
            this_row_stations = row.iloc[4:-2].index.values.tolist()
            this_row_valid_stations = list(filter(lambda station: row[station] != 0, this_row_stations))

            while len(this_row_valid_stations) > 4:
                this_row_valid_stations.pop()

            for i, nearby_station in enumerate(this_row_valid_stations):
                features[ f'stat{i + 1}_dist'  ].append( nearby_info[nearby_station]['dist'] )
                features[ f'stat{i + 1}_error' ].append( nearby_info[nearby_station]['error'] )
                features[ f'stat{i + 1}_pred'  ].append( row[nearby_station] )

            # preenche as espaços das estações vazias com zeros.
            for j in range(len(this_row_valid_stations), 4):
                features[ f'stat{j + 1}_dist'  ].append(0)
                features[ f'stat{j + 1}_error' ].append(0)
                features[ f'stat{j + 1}_pred'  ].append(0)

        formatted_station_dataset = pd.DataFrame().from_dict(features)
        #formatted_station_dataset.to_csv(f'{station_id}.csv', index = False)

        # uma em cada 5 observações será usada paa treinamento
        # em conformidade com a docstring deste script a seleção é feita desse jeito
        test_filter = formatted_station_dataset.index % 5 == 0

        test_sets.append(formatted_station_dataset.loc[test_filter, :])
        train_sets.append(formatted_station_dataset.loc[~test_filter, :])

        # salva os conjuntos de dados que serão utilizados no teste individual
        # de cada uma das estações no compositor treinado.
        formetted_station_df = formatted_station_dataset.loc[~test_filter, :]
        formetted_station_df.to_csv(f'./station_test_sets/{station_id}_formatted.csv', index = False)

    #
    # concatenando verticalmente os dataframes que foram obtidos 
    #
    train_dataset = pd.concat(objs = train_sets,
                              axis = 0,
                              ignore_index = True)

    test_dataset = pd.concat(objs = test_sets,
                             axis = 0,
                             ignore_index = True)


    #
    # Obtanendo os scalers para od dados e para os targets
    #
    whole_dataset = pd.concat(objs = test_sets + train_sets,
                              axis = 0,
                              ignore_index = True)

    data_scaler = MinMaxScaler().fit(whole_dataset.drop(columns = ['target']).values)
    target_scaler = MinMaxScaler().fit(whole_dataset.loc[:, 'target'].values.reshape(-1, 1))

    if not os.path.exists('./scalers'):
        os.mkdir('./scalers')

    joblib.dump(data_scaler, './scalers/data_scaler.dat')
    joblib.dump(target_scaler, './scalers/target_scaler.dat')

    print('Nomalizando conjuntos de dados')

    train_X = data_scaler.transform(train_dataset.drop(columns = ['target']).values)
    train_Y = target_scaler.transform(train_dataset.loc[:, 'target'].values.reshape(-1, 1))

    test_X = data_scaler.transform(test_dataset.drop(columns = ['target']).values)
    test_Y = target_scaler.transform(test_dataset.loc[:, 'target'].values.reshape(-1, 1))


    print('Salvando conjuntos de dados')

    if not os.path.exists('./data'):
        os.mkdir('./data')

    np.save(file = './data/xtrain.npy', arr = train_X)
    np.save(file = './data/ytrain.npy', arr = train_Y)

    np.save(file = './data/xtest.npy', arr = test_X)
    np.save(file = './data/ytest.npy', arr = test_Y)

