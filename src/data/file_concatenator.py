import os
import argparse
import pandas as pd
from tqdm import tqdm


def process_header(datafile) -> dict:
    """
    Obtains information stored in the header of the .csv files. Such
    information will be further be added to the respective dataframe.

    Parameters
    ----------
    `datafile`: str
        .csv file which header will be processed.

    """

    raw_info = []

    with open(datafile, 'r', encoding = "latin-1") as file:
        for i in range(8):
            raw_info.append(file.readline())

    info = {}

    for item in raw_info:

        atributo, i = item.split(':')

        # adjusting formatting (removing 'odd' characters)
        atributo = atributo.upper()
        atributo = atributo.replace('Ã', 'A')
        atributo = atributo.replace('Ç', 'C')

        information = i[1:-1].replace(',', '.')

        info[atributo] = information

    return info


def get_station_id(datafile: str) -> str:
    """
    Obtains the Station ID from a .csv datafile name.

    Parameters
    -----------
    `filename`: str
        Name of the datafile in filesystem

    Returns
    ---------
    `station_id`: str
        Station ID obtained from the filename
    """

    return datafile.split('_')[3]


def process_dataframes(source_dir, verbose):
    """
    Concatenates all dataframes by loading each one of the datafiles
    at a time.

    Parameters
    ----------
    `source_dir`: str
        Directory containing the filtered datafiles to be concatenated.

    Returns
    ----------
    `concatenated`: pd.DataFrame
        Dataframe with all datafiles information

    """

    if verbose:
        print(f'Loading .csv files from {os.getcwd()}...')

    columns_names = [
        'data',
        'hour',
        'precipitacao_total',
        'pressao_atmosferica',
        'pressao_atmosferica_max',
        'pressao_atmosferica_min',
        'radiacao_global',
        'temperatura',
        'ponto_orvalho',
        'temperatura_max',
        'temperatura_min',
        'ponto_orvalho_max',
        'ponto_orvalho_min',
        'umidade_relativa_max',
        'umidade_relativa_min',
        'umidade_relativa',
        'vento_direcao',
        'vento_rajada',
        'vento_velocidade'
    ]

    processed_dfs = []

    # once the features names are provided another csv line must be skipped (8 + 1)
    for file in tqdm(os.listdir(source_dir), ascii = True):

        # handles inconsistency with .csv and .CVS
        if any(ext in file for ext in ('.csv', '.CSV')):

            data = pd.read_csv(filepath_or_buffer = f'{source_dir}/{file}',
                                sep = ';',
                                decimal = ',',
                                encoding = 'latin-1',
                                skiprows = 9,
                                index_col = False,
                                names = columns_names)

            header = process_header(f'{source_dir}/{file}')

            data['latitude']   = header['LATITUDE']
            data['longitude']  = header['LONGITUDE']
            data['altitude']   = header['ALTITUDE']
            data['station_id'] = get_station_id(file)

            processed_dfs.append(data)

    return pd.concat(processed_dfs)


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description = 'Concatenates all datafiles into one dataframe' + \
                                               'stored in the format \'.ftr\'')

    ap.add_argument('-src', '--source-dir',
                    metavar = '',
                    required = True,
                    help = 'Source directory for the selected datafiles.')

    ap.add_argument('-o', '--output',
                    metavar = '',
                    required = False,
                    default = 'concatenated_dataframe',
                    help = 'Concatenated datafile prefix..')

    ap.add_argument('-csv', '--save-to-csv',
                    action = 'store_true',
                    required = False,
                    default = 0,
                    help = 'Optionally saves the concatenated file in .csv format.')

    ap.add_argument('-v', '--verbose',
                    metavar = '',
                    required = False,
                    default = 0,
                    type = int,
                    help = "Verbosity level: 0 (silent) otherwise verbose.")

    args = vars(ap.parse_args())


    dataframe = process_dataframes(source_dir = args['source_dir'],
                                   verbose = args['verbose'])

    if args['verbose']:
        print(f'Saving dataframe to disk in \'{args["output"]}.ftr...')

    # safing in .ftr requires the dataframe to be reindexed
    dataframe.reset_index(inplace = True, drop = True)
    dataframe.to_feather(f'{args["output"]}.ftr')

    if args['save_to_csv']:

        if args['verbose']:
            print(f'Saving dataframe to disk in \'{args["output"]}.ftr...')

        dataframe.to_csv(f'{file_prefix}.csv', index = False)
