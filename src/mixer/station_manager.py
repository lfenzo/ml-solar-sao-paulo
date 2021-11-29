"""
Script providing functionalities to manage the stations inside
the directory 'mixer/'

The optins include:

-> checkpoint cleaning (remove all models fitted before a certain timestamp)
-> station file cleaning (remove data, models, and performance reports)
-> verify the models fitted inside each station directory
-> verify the number of samples in each station datafiles (x, y, train, test)

"""

import os
import re
import joblib
import argparse
import numpy as np


def define_cli_args():
    """
    Encapsulates CLI argument definitions
    """

    ap = argparse.ArgumentParser(description = 'Station Manager')

    ap.add_argument('-d', '--data',
                    metavar = '',
                    choices = ['original', 'imputed'],
                    required = True,
                    type = str,
                    help = 'Set of stations to apply operations on.' \
                           ' Must be specified before the action')


    subparser = ap.add_subparsers(dest = 'command', help = 'Checkpoint')


    checkpoint_parser = subparser.add_parser('checkpoint',
                                             help = 'Clear models before \'timestamp\'.')

    checkpoint_parser.add_argument(
        '-ts', '--timestamp',
        metavar = '',
        type = str,
        required = True,
        help = "Checkpoint timestamp (models before this date will be deleted)"
    )



    verify_parser = subparser.add_parser('verify',
                                         help = 'Verify sample counts or models')

    verify_parser.add_argument(
        '-t', '--target',
        metavar = '',
        choices = ['stations', 'models'],
        type = str,
        required = True,
        help = "Verify train/test samples available in each station."
    )

    verify_parser.add_argument(
        '-csv', '--save-to-csv',
        action = 'store_true',
        default = False,
        help = 'Save summary to csv file. Default = False'
    )



    clear_parser = subparser.add_parser('clear',
                                        help = 'Clean files inside the specified station set')

    clear_parser.add_argument('-m', '--models',
                              action = 'store_true',
                              required = False,
                              help = 'Remove all saved models in all stations.')

    clear_parser.add_argument('-p', '--perf-reports',
                              action = 'store_true',
                              required = False,
                              help = 'Remove all performance report csvs.')

    clear_parser.add_argument('-s', '--scalers',
                              action = 'store_true',
                              required = False,
                              help = 'Remove all scaler objects saved.')

    clear_parser.add_argument('-yes', '--assumeyes',
                              action = 'store_true',
                              required = False,
                              help = 'Assume \'yes\' for all confirmation questions.')

    return ap


def verify_stations(source_dir, save_to_csv = False):

    os.chdir(source_dir)

    station_dirs = []

    for item in sorted(os.listdir()):
        if os.path.isdir(item) and len(item) == len('A000'):
            station_dirs.append(item)

    print('Station\tTrain Shape\tTest Shape')

    train_acc, test_acc = 0, 0

    train_min, train_max = np.inf, 0
    test_min, test_max = np.inf, 0

    serie_train = []
    serie_test = []
    serie_station = []

    for station in station_dirs:

        xtrain = np.load(f'{station}/x_train.npy')
        xtest = np.load(f'{station}/x_test.npy')

        print(f'{station}\t{xtrain.shape}\t{xtest.shape}')

        if args['save_to_csv']:
            serie_train.append(xtrain.shape[0])
            serie_test.append(xtest.shape[0])
            serie_station.append(station)

        train_acc += xtrain.shape[0]
        test_acc  += xtest.shape[0]

        if xtrain.shape[0] < train_min:
            train_min = xtrain.shape[0]

        if xtrain.shape[0] > train_max:
            train_max = xtrain.shape[0]

        if xtest.shape[0] < test_min:
            test_min = xtest.shape[0]

        if xtest.shape[0] > test_max:
            test_max = xtest.shape[0]

        del xtrain, xtest

    if args['save_to_csv']:

        dataset = pd.DataFrame(data = {
            'station': serie_station,
            'train': serie_train,
            'test': serie_test
        })

        dataset.to_csv(f'{source_dir}__samples_by_station.csv', index = False)

    print('=' * 34)

    print(f'Total = {train_acc}\t\t{test_acc}')
    print(f'Mean  = {train_acc / len(station_dirs):.2f}\t{test_acc / len(station_dirs):.2f}')
    print(f'Max   = {train_max}\t\t{test_max}')
    print(f'Min   = {train_min}\t\t{test_min}')

    # exists the verify station set (directory)
    os.chdir('..')


def verify_models(source_dir):

    os.chdir(source_dir)

    most_recent_timestamp = '2000-01-01'

    for i, item in enumerate( sorted(os.listdir('.')) ):

        if os.path.isdir(item) and len(item) == len('A000'):

            print(f'({i + 1}) -- {item}', end = '\t')

            if os.path.exists(f'{item}/models/.'):

                for im, model in enumerate( sorted(os.listdir(f'{item}/models/.')) ):
                    print(f'{model}') if im == 0 else print(f'\t\t{model}')

                    timestamp = get_timestamp(model)

                    if timestamp > most_recent_timestamp:
                        most_recent_timestamp = timestamp

            print()

    print(f"Most recent timestamp = {most_recent_timestamp}")

    # exists the verify station set (directory)
    os.chdir('..')


def get_timestamp(filename):
    return re.split('[_]|[.]', filename)[2]


def can_delete_model(model_timestamp, checkpoint):
    """
    Recebe uma timestamp de um arquivo de um modelo e verifica se esse modelo
    deve ser apagado com base na timestamp do 'checkpoint'.

    Parametros
    --------------
    model_timestamp : str
        timestamp do modelo atual
    checkpoint : str
        timestamp de checkpoint (modelos mais novos que ela devem ser deletados)

    Retorna
    ------------
    bool
        True caso o modelo atual deve ser apagado.

    Nota: O timestamp 'maior' é aquela mais recente
    """
    return model_timestamp > checkpoint


def checkpoint_clear(source_dir, checkpoint_timestamp):

    os.chdir(source_dir)

    for station_dir in [d for d in os.listdir('.') if re.match('^[A-Z][0-9]{3}', d)]:

        os.chdir(f'./{station_dir}')

        for m in os.listdir('./models/'):
            if can_delete_model(get_timestamp(m), checkpoint_timestamp):
                os.remove(os.path.join('.', 'models', m))

        os.chdir('..')

    os.chdir('..')


def remove_models(source_dir):

    os.chdir(source_dir)

    for station_dir in [d for d in os.listdir('.') if re.match('^[A-Z][0-9]{3}', d)]:
        for file in os.listdir( os.path.join(station_dir, 'models') ):
            os.remove( os.path.join(station_dir, 'models', file) )

    os.chdir('..')


def remove_perf_logs(source_dir):

    os.chdir(source_dir)

    for station_dir in [d for d in os.listdir('.') if re.match('^[A-Z][0-9]{3}', d)]:
        for file in os.listdir(station_dir):
            if '.csv' in file and 'metadata' not in file:
                os.remove( os.path.join(station_dir, file) )

        # removing predictions data files from all models
        for pred_file in os.listdir( os.path.join(station_dir, 'predictions') ):
            os.remove( os.path.join(station_dir, 'predictions', pred_file) )

    os.chdir('..')


def remove_scalers(source_dir):

    os.chdir(source_dir)

    for station_dir in [d for d in os.listdir('.') if re.match('^[A-Z][0-9]{3}', d)]:
        for file in os.listdir(station_dir):
            if 'scaler' in file:
                os.remove(file)

    os.chdir('..')


if __name__ == "__main__":

    ap = define_cli_args()
    args = vars(ap.parse_args())

    if args['command'] == 'verify':

        print(f'Verifying {args["target"]} in {args["data"]} dataset')

        if args['target'] == 'stations':
            verify_stations(
                source_dir = args['data'],
                save_to_csv = args['save_to_csv']
            )

        elif args['target'] == 'models':
            verify_models(source_dir = args['data'])

        scaler = joblib.load(f'./{station_id}/{station_id}_scaler_target.dat')

    elif args['command'] == 'checkpoint':

        print(f'Checkpoint-cleaning ({arga["checkpoint"]}) in {args["data"]} dataset')

        checkpoint_clear(
            source_dir = args['data'],
            checkpoint_timestamp = args['checkpoint']
        )


    elif args['command'] == 'clear':

        print("Cleaning ", end = '')

        if args['models']:
            print("models, ", end = '')

        if args['perf_reports']:
            print("performance reports, ", end = '')

        if args['scalers']:
            print("scalers ", end = '')

        print("...")


        if not args['assumeyes']:

            escolha = input('\nAre sure you want to remove everything? [Y/n] ')

            if escolha not in ['Y', 'y', 's', 'S', 'yes', 'sim']:
                print('Operation aborted...')
                exit()

        if args['models']:
            print(f'Removing all fitted models in {args["data"]} datasets')
            remove_models(source_dir = args['data'])

        elif args['scalers']:
            print(f'Deleting scaler objects in {args["data"]} datasets')
            remove_scalers(source_dir = args['data'])

        elif args['perf_reports']:
            print(f'Deleteing all csvs in {args["data"]} datasets')
            remove_perf_logs(source_dir = args['data'])

