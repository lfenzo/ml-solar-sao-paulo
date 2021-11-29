import os
import argparse
import numpy as np
import pandas as pd


def select_by_algorithm(dataframe) -> pd.DataFrame:
    """
    Selects best regressor by algorithm.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with all regressors and metrics for a given station

    Returns
    ---------
    `best_regs`: pd.DataFrame
        Dataframe with the best regressors
    """

    best_regressors = []

    for m in dataframe['model'].unique():

        model = dataframe.loc[ dataframe['model'] == m ]
        model = model.sort_values(by = ['rmse', 'mae', 'mbe', 'r2'])

        best_regressors.append(pd.DataFrame(model.iloc[0]))

    return pd.concat(best_regressors, axis = 1).T


def get_best_regressor(dataframe, crit) -> pd.Series:
    """
    Selects the best regressor of a station.
    """

    if crit == 'rmse':
        dataframe.sort_values(by = ['rmse', 'mae', 'mbe', 'r2'], inplace = True)

    elif  crit == 'mae':
        dataframe.sort_values(by = ['mae', 'rmse', 'mbe', 'r2'], inplace = True)

    elif  crit == 'mbe':
        dataframe.sort_values(by = ['mbe', 'rmse', 'mae', 'r2'], inplace = True)

    else:
        raise ValueError(f'Invalid criterions passed: criterion {crit} is not available.')

    return dataframe.iloc[0]


def define_cli_args():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-crit', '--criterion',
                            metavar = '',
                            required = False,
                            default = 'rmse',
                            type = str,
                            help = 'Selection criteria of the best estimator. Default = rmse')

    arg_parser.add_argument('-t', '--target',
                            metavar = '',
                            choices = ['original', 'imputed', 'merged'],
                            required = True,
                            help = "Station directory targeted: (original, imputed or merged).")

    arg_parser.add_argument('-tex', '--tex-table',
                            required = False,
                            action = 'store_true',
                            help = 'Saves the performance table with Latex formatting.')

    return arg_parser


def save_latex_format(filepath, dataframe):
    """
    Formats the the `dataframe` in a format ready to be used as a LaTeX table
    """

    df = dataframe.copy()

    df['mean'] = (df['rmse'] + df['mae']) / 2
    df['mean'] = df['mean'].round(decimals = 3)

    with open(filepath, 'w') as file:

        for _, row in df.iterrows():
            formmatted_string = '\\texttt{' + str(row['station']) + '} & ' + \
                                '\\texttt{' + str(row['model']) + '} & ' + \
                                '\\texttt{' + str(row['rmse']) + '} & ' + \
                                '\\texttt{' + str(row['mae']) + '} & ' + \
                                '\\texttt{' + str(row['mbe']) + '} & ' + \
                                '\\texttt{' + str(row['r2']) + '} & ' + \
                                '\\texttt{' + str(row['mean']) + '} \\\\\n'

            file.write(formmatted_string)

        file.write('\\midrule \n')

        # calculates and writes the mean values of each performance metrics in `dataframe`
        formmatted_string = '\\multicolumn{3}{c}{\\textbf{Média}} ' + \
                            '\\texttt{\\textbf{' + str( round(df['rmse'].mean(), 3) ) + '}} & ' + \
                            '\\texttt{\\textbf{' + str( round(df['mae'].mean(), 3) ) + '}} & ' + \
                            '\\texttt{\\textbf{' + str( round(df['mbe'].mean(), 3) ) + '}} & ' + \
                            '\\texttt{\\textbf{' + str( round(df['r2'].mean(), 3) ) + '}} & ' + \
                            '\\texttt{\\textbf{' + str( round(df['mean'].mean(), 3) ) + '}} \\\\\n'

        file.write(formmatted_string)


if __name__ == '__main__':

    arg_parser = define_cli_args()
    args = vars(arg_parser.parse_args())

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    if args['target'] == 'original':
        SOURCE_DIR = os.path.join('mixer', 'original')

    elif args['target'] == 'imputed':
        SOURCE_DIR = os.path.join('mixer', 'imputed')

    elif args['target'] == 'merged':
        SOURCE_DIR = os.path.join('merged_stations')

    os.chdir(SOURCE_DIR)


    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    station_dirs = []

    for item in os.listdir():
        if os.path.isdir(item) and len(item) == len('A000'):
            station_dirs.append(item)

    best_by_station = []

    # for every best staion model...
    for station_id in sorted(station_dirs):

        results = pd.read_csv(f'{station_id}/{station_id}_results.csv')

        best_by_algorithm = select_by_algorithm(results)
        best_by_algorithm.to_csv(f'{station_id}/{station_id}_best_by_algorithm.csv', index = False)

        # gets the best model in order to save the list afterwards
        best_model = pd.DataFrame(get_best_regressor(best_by_algorithm,
                                                        crit = args['criterion'])).T
        best_model['station'] = station_id

        # reordering dataframe features
        columns = ['station',
                   'model',
                   'timestamp',
                   'rmse',
                   'mae',
                   'mbe',
                   'r2',
                   'mape',
                   'h-params']

        best_model = best_model[columns]

        best_by_station.append(best_model)


    best_models = pd.concat(best_by_station, axis = 0, ignore_index = True)

    # performance feature decimal rouding
    for feature in ['rmse', 'mae', 'mbe', 'r2', 'mape']:
        best_models[feature] = pd.to_numeric(best_models[feature]).round(decimals = 3)

    if args['tex_table']:
        save_latex_format(filepath = 'best_models_table.tex',
                          dataframe = best_models)

    best_models.to_csv('best_models.csv', index = False)


    #
    #
    # Produces the hyperparameter report
    #
    #
    for model_name in best_models['model'].unique():

        models_hparams = {
            'model': [],
            'station': []
        }

        for station_id in sorted(station_dirs):

            results = pd.read_csv( os.path.join(station_id, f'{station_id}_results.csv') )

            best_by_algorithm = select_by_algorithm(results)
            best_by_algorithm = best_by_algorithm.loc[ best_by_algorithm['model'] == model_name ]

            # obtains the best hyperparameter dictionary from the string in the dataframe
            hp_dict = eval( best_by_algorithm.head(1)['h-params'].values[0].replace('nan', 'None') )

            models_hparams['model'].append(model_name)
            models_hparams['station'].append(station_id)

            for key, value in hp_dict.items():

                if key not in models_hparams.keys():
                    models_hparams[key] = []

                models_hparams[key].append(value)

        models_hparams_df = pd.DataFrame().from_dict(models_hparams)

        # creates the directory that contains the h-param reports
        # already inside the target data directory (original or imputed)
        if not os.path.exists('hparams'):
            os.mkdir('hparams')

        models_hparams_df.to_csv(f'./hparams/{model_name}_hparams.csv', index = False)
