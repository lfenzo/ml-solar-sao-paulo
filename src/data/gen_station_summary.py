import os
import numpy as np
import pandas as pd


def format_date(string):
    ano, mes, dia = string.split('-')
    data_formatada = f'{dia}/{mes}/{ano}'
    return data_formatada


def get_station_code(file: str):
    informacoes = file.split('_')
    return informacoes[3]



def get_station_city(file: str):
    informacoes = file.split('_')
    return informacoes[4]


def get_station_city_names(source_dir):
    """
    Obtains the city names from the file headers in `source_dir`.

    Parameters
    ----------
    `source_dir`: str
        Directori path containing data files

    Returns
    ----------
    `info`: pd.DataFrame
        Dataframe with the Stations Ids and the respective city name
        in each row.

    """

    info = pd.DataFrame(columns = ['city_name'])
    info.index.name = 'stations_id'

    for root, dirs, files in os.walk(source_dir):
        for file in files:

            station_code = get_station_code(file)
            station_city_name = get_station_city(file).title()

            if station_code not in info.index.to_list():
                info.at[station_code, 'city_name'] = station_city_name

    return info


if __name__ == "__main__":

    df = pd.read_feather('concatenated_dataframe.ftr')
    df = df[['station_id', 'latitude', 'longitude', 'altitude', 'data']]

    groupby = df.groupby(by = 'station_id', group_keys = False)
    df = groupby.aggregate(['min', 'count'])
    df = df.reset_index(drop = False)


    stations = pd.DataFrame()

    for feature in ['station_id', 'latitude', 'longitude', 'altitude', 'data']:

        if feature == 'data':
            stations['first_register'] = df[feature]['min']
            stations['n_samples']      = df[feature]['count']

        else:
            if feature != 'station_id':
                stations[feature] = df[feature]['min']
            else:
                stations[feature] = df['station_id']


    # colocando todas as datas no mesmo formato
    stations['first_register'] = stations['first_register'].str.replace('/', '-')
    stations['first_register'] = stations['first_register'].apply(format_date)


    # obtaining city names from the filse inside './inmet_sao_paulo'
    city_names = get_station_city_names('./inmet_sao_paulo')

    stations = stations.join(city_names, on = 'station_id')


    # saves statoins summary to csv
    stations.to_csv('station_summary.csv', index = False)
