import os

import numpy as np
import pandas as pd


def get_best_model_filepaths(models_dataframe) -> dict:
    """
    Obtem o arquivo correspondente a cada um dos melhores modelos

    Parecido com a função get_best_model_directory_paths do script mixer/blend_all.py

    Returns
    ----------
    model_filepaths : dict
        Dicionário contendo o caminho para o arquivo do melhor modelo da estação

    """

    best_model_filepaths = {}

    for _, row in models_dataframe.iterrows():

        best_regressor_signature = f'{row["model"]}_{row["timestamp"]}'

        for file in os.listdir(f'../stations/{row["station"]}/models'):
            if best_regressor_signature in file:
                best_model_filepaths[ row["station"] ] = os.path.join( '..', 'stations', row['station'], 'models', file )

    return best_model_filepaths


def preprocess_batch(data,
                     scaler_data,
                     scaler_target,
                     pca_pressao = None,
                     pca_temp = None,
                     pca_umid_1 = None,
                     pca_umid_2 = None,
                     pca_orvalho = None,
                     reduce_dim = False,
                     *args, **kwargs) -> (pd.DataFrame, np.array):
    """
    Realiza o preprocesamento de um batch (dataframe)

    Parameters
    --------------

    data : DataFrame
        Dataframe contendo as observações da estação atual. Deve conter apenas o
        período de testes (2019 inclusive até 2021)

    scaler_data : Scaler Object
        Scaler utilizada para transformar os dados de entrada (data) desta estação

    scaler_target : Scaler Object
        Scaler utilizada para transformar o atributo alvo (target) da estação

    reduce_dim : bool, default=False
        Caso seja True utilizará os objetos PCA fornecidos para reduzir a
        dimensionaildade dos atributos:
            - temperatura,
            - umidade do ar,
            - pressão atmosférica,
            - ponto de Orvalho.

    Returns
    -------------
    (datetimes_df, scaled_batch_data) : tuple
        Gera uma tupla contendo dois valores:
            - datetime_df: Datafarmes contendo
            - scaled_batch__data: dados relativos ao batch padronizados conforme o
            scaler_data fornecido
    """

    if 'rad_prox_hora' in data.columns:
        data = data.drop(columns = ['rad_prox_hora'])

    if 'year' in data.columns:
        data = data.drop(columns = ['year'])

    #
    # removendo colunas do dataframe passado que não serão utilizadas
    #
    timestamp_df = pd.DataFrame()

    """
    Example
    -----------
    Timestamp: 2 0 1 9 - 0 1 - 0 1     1  0  :  0  0 (:00+00:00)
               0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15    ...
               \-----/   \-/   \-/    \---/    \--------------/
                 ano     mes   dia    hora         ignorado
    """

    timestamp_df['year']  = data['datetime'].str[ 0:4 ]
    timestamp_df['month'] = data['datetime'].str[ 5:7 ]
    timestamp_df['day']   = data['datetime'].str[ 8:10]
    timestamp_df['hour']  = data['datetime'].str[11:13]
    timestamp_df['pred']  = data['datetime'].str[11:13]

    timestamp_df.reset_index(drop = True, inplace = True)

    #
    # preprocessamento das informações (data)
    #
    batch_data = data.drop(columns = ['datetime'])
    scaled_batch_data = scaler_data.transform(batch_data.values)

    return timestamp_df, scaled_batch_data
