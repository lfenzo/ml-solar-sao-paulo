import os
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd

from numpy import nan
from tqdm import tqdm
from datetime import datetime
from argparse import RawTextHelpFormatter

from sklearn.model_selection import GridSearchCV, \
                                    RandomizedSearchCV, \
                                     cross_validate

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor, \
                             ExtraTreesRegressor, \
                             StackingRegressor


def get_best_hiperparams(station_id):
    """
    Obtains the best hyperparameters for the each model in fitted with `station_id`
    data.

    Parameters
    ----------
    `station_id`: str
        Station ID in the format "A000".

    Returns
    ---------
    `best_hp`: dict
        Dictionary with the best hyperparameters for each fitted model in `station_id`
    """

    best_hp = {}
    info = pd.read_csv(f'{station_id}_best_by_algorihm.csv')

    # 'eval' converts the string to a python dict inside the `info` dataframe
    for _, row in info.iterrows():
        best_hp[ row['model'] ] = eval(row['h-params'])

    return best_hp


def fit_selected_model(model, X, Y, strategy: str, param_grid = None, n_folds = 5):
    """
    Função responsável por treinar os modelos da classe do Scikit-Learn.

    Parâmetros
    -------------
    `model`: Scikit-Learn estimator
        Sctikit-learn estimator implementing `.fit()` method.

    `X`: array-like
        Training data

    `Y`: array-like
        Training targets

    `strategy`: str
        Training strategy: `cross_validation` or `grid_search`.

    `n_folds`: int, default = 5
        núemro de dobras que ocorrerão nos processos de cross-validation ou grid-search

    `param_grid`: default = None
        Grid Search hyperparameter search space (specified as python dictionary). Only
        used when `strategy = grid_search`.

    Returns
    -------------
    A fitted instance of `model` with `X` and `Y` using `stategy`.
    """

    if strategy == 'cross_validation':

        cross_validation = cross_validate(model,
                                          X,
                                          Y,
                                          scoring = 'neg_mean_squared_error',
                                          n_jobs = -1,
                                          cv = n_folds,
                                          verbose = verbose,
                                          return_estimator = True)

        # selects the best performing models among the folds
        model = cross_validation['estimator'][ np.argmax(cross_validation['test_score']) ]

    elif strategy == 'fit':
        model.fit(X, Y)

    elif strategy == 'grid_search':
        grid_search = GridSearchCV(estimator = model,
                                   param_grid = param_grid,
                                   scoring = 'neg_mean_squared_error',
                                   cv = n_folds,
                                   verbose = verbose)

        grid_search.fit(X, Y)

        model = grid_search.best_estimator_

    return model


def save_model(model, verbose = 0):
    """
    Saves the fitted `model` to disk in the format <name>_<timestamp>.dat.
    Assumes that the current working directory is directory in which the model
    will be saved.

    Parameters
    --------------
    `model`:
        modelo treinado a ser salvo em disco.

    `verbose`: int, default = 0
        Verbosity level: 0 (silent) otherwise verbose.
    """

    model_name = model.__class__.__name__

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = os.path.join('models', f'sk_{model_name}_{timestamp}.dat')

    if verbose:
        print(f'Saving model (\'{filename}\') ...', end = ' ')

    joblib.dump(value = model, filename = filename)

    if verbose:
        print('done!')


def choose_model(name: str, station_id, use_best_hparams):
    """
    Determines the training algorithm to be used given its `name`.

    Parameters
    ------------
    name : str
        Algorithm name to be used.

    station_id : str
        Station ID which data will be used to train the model.

    use_best_params : bool
        Uses best h-params found during Grid-Serach routines

    Returns
    ---------
    `model`
        Fitted instance of the model
    """

    # once the stacking regressor will only be fitted after the execution 
    # of grid search in the atomic estimators it is not necessary to train
    # such model with h-params other than the best h-params obtained
    if name == 'sr':
        best_hparams = get_best_hiperparams(station_id = station_id)

        model = StackingRegressor(
            final_estimator = MLPRegressor(
                early_stopping = True,
                max_iter = 1000,
                tol = 5e-5,
                n_iter_no_change = 20,
                hidden_layer_sizes = (50, 30)
            ),
            estimators = [
                ('svr', SVR(cache_size = 1024).set_params(**best_hparams['SVR'])),
                ('mlp', MLPRegressor().set_params(**best_hparams['MLPRegressor'])),
                ('xgb', XGBRegressor().set_params(**best_hparams['XGBRegressor'])),
                ('rf',  RandomForestRegressor().set_params(**best_hparams['RandomForestRegressor'])),
                ('ext', ExtraTreesRegressor().set_params(**best_hparams['ExtraTreesRegressor']))
            ],
            n_jobs = -1,
        )

        return model

    elif name == 'rf':
        model = RandomForestRegressor(n_jobs = -1)

    elif name == 'mlp':
        model = MLPRegressor(n_iter_no_change = 25,
                             tol = 1e-4,
                             max_iter = 2000,
                             early_stopping = True)

    elif name == 'xgb':
        model = XGBRegressor(n_jobs = -1)

    elif name == 'extr':
        model = ExtraTreesRegressor(n_jobs = -1)

    elif name == 'svr':
        model = SVR(cache_size = 1024)

    if use_best_hparams:
        best_hparams = get_best_hiperparams(station_id = station_id)
        model.set_params( **best_hparams[model.__class__.__name__] )

    return model


def load_param_grid(filepath):
    """
    Loads the param grid specified in `filepath` to a python dictionary.
    This function assumes that there are no errors in `filepath` json file

    Parameters
    -----------
    `filepath`: str
        Filepath containing the h-param search space formated in .json

    Returns
    -----------
    `param_grid`: dict
        Dictionary with the h-param search space specifications
    """

    with open(os.path.join(os.getcwd(), filepath), 'rb') as file:
        param_grid = pickle.load(file)

    return param_grid


def define_cli_arguments():
    """
    Encapsulates the definition of the CLI arguments used
    """

    # RawTextHelpFormatter for more specific (manual?) stdout print formatteing
    arg_parser = argparse.ArgumentParser(formatter_class = RawTextHelpFormatter)

    # =============================================================
    # ==================== TRAINING STATEGIES =====================
    # =============================================================

    train_methods = arg_parser.add_mutually_exclusive_group(required = True)

    train_methods.add_argument('-cv', '--cross-validation',
                               action = 'store_true',
                               help = 'Training using \'cross-validation\' routines.')

    train_methods.add_argument('-fit', '--standard-fit',
                               action = 'store_true',
                               help = 'Training using \'fit\' routines (default behavior).')

    train_methods.add_argument('-gs', '--grid-search',
                               action = 'store_true',
                               help = 'Training using \'grid-seach\' routines.')

    # =============================================================
    # ==================== TRAINING OPTINOS =======================
    # =============================================================

    station_train_options = arg_parser.add_mutually_exclusive_group(required = False)

    station_train_options.add_argument('-ws', '--warm-start',
                                       metavar = '',
                                       required = False,
                                       default = None,
                                       type = str,
                                       help = 'Starts training in specified station' \
                                              '(Station ID in the format A000)')

    station_train_options.add_argument('-s', '--station',
                                       metavar = '',
                                       required = False,
                                       default = None,
                                       help = 'Perform training only specified station ' \
                                              '(in thr format A000).')

    # =============================================================
    # ============++++===== OTHER OPTIONS ===++++==================
    # =============================================================

    arg_parser.add_argument('-d', '--data',
                            metavar = '',
                            required = True,
                            choices = ['original', 'imputed'],
                            type = str,
                            help = 'Set of datasets to use during training: \'original\' or \'imputed\'')

    arg_parser.add_argument('-hp', '--best-hparams',
                            action = 'store_true',
                            required = False,
                            help = 'Use best h-params (only after Grid-Search).')

    arg_parser.add_argument('-n', '--n-folds',
                            metavar = '',
                            required = False,
                            default = 5,
                            type = int,
                            help = 'Number of folds in cross-validation execution. Default: 5.')

    arg_parser.add_argument('-sm', '--save-model',
                            metavar = '',
                            type = bool,
                            default = True,
                            help = 'Saves the best model in the format [model name]_[timestamp].dat. ' \
                                   'The name is automatically assigned. Default = True')

    arg_parser.add_argument('-pf', '--param-file',
                            metavar = '',
                            required = False,
                            default = 'param_grid.pkl',
                            type = str,
                            help = 'File with custom h-param search space.')

    arg_parser.add_argument('-dr', '--dry-run',
                            metavar='',
                            required = False,
                            default = 0,
                            type = int)

    arg_parser.add_argument('-alg', '--training-algorithm',
                            metavar = '',
                            required = True,
                            type = str,
                            help = 'Object instantiated at the moment of fitting, the following are available:' +
                                    '\n\t\'svr\': Support Vector Regressor' +
                                    '\n\t\'rf\': Random Forest Regressor' +
                                    '\n\t\'extr\': Extra Tree Regressor' +
                                    '\n\t\'xbg\': Extreme Gradient Boosting' +
                                    '\n\t\'mlp\': Multi Layer Perceptron' +
                                    '\n\t\'sr\': Stacking Regressor')

    arg_parser.add_argument('-v', '--verbose',
                            metavar = '',
                            required = False,
                            default = 0,
                            help = 'Verbosity level: 0 (silent) otherwise verbose. Default = 0',
                            type = int)

    return arg_parser


def check_execution(current_station_id, warm_start, train_only):
    """

    Checks the execution of each station in the traiing loop. Depending
    on the training options not all statins are fitted. This functions
    checks which ones should be executed.

    Parameters
    -----------
    current_station_id : str
        Current Station ID and the moment of checking

    warm_start : str, None, optional
        Station ID from which training should start in case of `warm_start != None`.

    train_only : str, None, optional
        Station ID of the only station to be trained in case of `train_only != None`.

    Returns
    --------
    bool
        Specifies if the current station should have its models fitted.
    """

    if warm_start == None and train_only == None:
        return True

    elif train_only != None:
        return True if current_station_id == train_only else False

    elif warm_start != None:
        return True if current_station_id >= warm_start else False


if __name__ == '__main__':

    arg_parser = define_cli_arguments()
    args = vars(arg_parser.parse_args())

    param_grid = None

    # assumaes that if the directories exists, all data necessary to fit the models
    # has already been preprocessed
    # DATA_DIR is the directory with all preprocessed stations
    if args['data'] == 'imputed':

        DATA_DIR = os.path.join('mixer', 'imputed')

        if os.path.exists(DATA_DIR):
            os.chdir(DATA_DIR)
        else:
            raise FileNotFoundError('Data source directory \'./mixer/imputed\' not found.')

    elif args['data'] == 'original':
        DATA_DIR = os.path.join('mixer', 'original')

        if os.path.exists(DATA_DIR):
            os.chdir(DATA_DIR)
        else:
            raise FileNotFoundError('Data source directory \'./mixer/original\' not found.')


    if args['cross_validation']:
        strategy = 'cross_validation'

    elif args['standard_fit']:
        strategy = 'fit'

    elif args['grid_search']:
        strategy = 'grid_search'
        param_grid = load_param_grid(filepath = args['param_file'])

    # -----------------------------------------------------------
    # -----------------------------------------------------------

    # enters the appropriate directory based on the training data strategy:
    # -> original data or
    # -> imputed data

    station_dirs = []

    # reads only files from the `stations/` directory
    for item in os.listdir():
        if os.path.isdir(item) and len(item) == len('A000'):
            station_dirs.append(item)

    station_dirs.sort()

    if args['verbose']:
        print(f'Training {len(station_dirs)} stations with algorithm {args["training_algorithm"]}', end = '')
        print(f' on {args["data"]} data...')


    for i, station_id in enumerate(station_dirs):

        should_execute = check_execution(current_station_id = station_id,
                                         warm_start = args['warm_start'],
                                         train_only = args['station'])

        if should_execute:

            if args['verbose']:
                print(f'[{i:2} / {len(station_dirs) - 1}] -- {station_id} || ', end = '')
                print('Training started... ', end = '')

            # loading the data in this station directory
            os.chdir(station_id)

            xtrain = np.load('x_train.npy')
            ytrain = np.load('y_train.npy')

            model = choose_model(name = args['training_algorithm'],
                                 station_id = station_id,
                                 use_best_hparams = args['best_hparams'])

            if args['dry_run']:
                print(f'dry-run: {model.__class__.__name__} trained...')

            else:
                fitted_model = fit_selected_model(model = model,
                                                  X = xtrain,
                                                  Y = ytrain.ravel(),
                                                  strategy = strategy,
                                                  n_folds = args['n_folds'],
                                                  param_grid = param_grid)

                # checks the existence of `models` directory wihtin the current station directory
                if not os.path.exists('models'):
                    os.mkdir('models')

                if args['save_model']:
                    save_model(model = fitted_model, verbose = args['verbose'])

            os.chdir('..')
