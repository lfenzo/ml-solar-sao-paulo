"""
Constroi a composição a partir dos modelos treinados que estão presentes em stations/
"""

import os
import shutil
import pandas as pd

def copy_to_composition(station_dir: str, model = None, timestamp = None):
    """
    Copia os arquivos e objetos necessários para funcionamento da composição.

    São transferidos:
    - estimador (modelo),
    - arquivos .csv de metadados,
    - Scalers de transformação de escala,
    - Transformadores de PCA (para reducão de dimensionalidade)
    """

    # copiando modelos
    shutil.copy(src = f'../stations/{station_dir}/models/sk_{model}_{timestamp}.dat',
                dst = f'./models/{station_dir}/sk_{model}_{timestamp}.dat')

    # copiando arquivos de metadados
    shutil.copy(src = f'../stations/{station_dir}/{station_dir}_metadata.csv',
                dst = f'./models/{station_dir}/{station_dir}_metadata.csv')

    # copiando scalers
    shutil.copy(src = f'../stations/{station_dir}/{station_dir}_scaler_data.dat',
                dst = f'./models/{station_dir}/{station_dir}_scaler_data.dat')

    shutil.copy(src = f'../stations/{station_dir}/{station_dir}_scaler_target.dat',
                dst = f'./models/{station_dir}/{station_dir}_scaler_target.dat')

    # TODO verificar a criação por que os objetos PCA não estão aparecendo em agumas estações
    # copiando transformadores de PCA
#    for feature in ['orvalho', 'pressao', 'temperatura', 'umidade']:
#        shutil.copy(src = f'../stations/{station_dir}/{station_dir}_pca_transformer_{feature}.dat',
#                    dst = f'./{station_dir}/{station_dir}_scaler_data.dat')


if __name__ == '__main__':

   # neste ponto o script 'blend_all' já deve ter sido executado para ser
    # necessário apenas copiar os melhores modelos da pasta 'stations'
    best_estimators = pd.read_csv('../stations/best_overall.csv')

    for i, row in best_estimators.iterrows():

        # cria o diretório onde todos os modelos serão armazenados
        if not os.path.exists(f'./models'):
            os.mkdir(f'./models')

        # cria o diretório relativo àquela estação caso não exista if not os.path.exists(f'./models/{row["station"]}'):
        if not os.path.exists(f'./models/{row["station"]}'):
            os.mkdir(f'./models/{row["station"]}')

        print(f'Copiando conteudos da estação {row["station"]}...')

        copy_to_composition(station_dir = str(row['station']),
                            model = row['model'],
                            timestamp = row['timestamp'])

    # copia as informações de metadados das estações
    shutil.copy(src = f'../data/station_status.csv',
                dst = f'./station_status.csv')

    # copia as informações dos melhores estimadores
    shutil.copy(src = f'../stations/best_overall.csv',
                dst = f'./best_overall.csv')
