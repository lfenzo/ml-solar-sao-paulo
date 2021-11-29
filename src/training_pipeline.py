import os
import argparse


ORIGINAL_DATA = 0
IMPUTED_DATA = 1


def format_command(script: str, arg_list: list) -> str:
    """
    Formats the arguments passed as a dictionary in such way that they
    are ready to be passed to `os.system(...)`
    """

    command = f'python {script}'

    for arg in arg_list:
        command += f' {str(arg)}'

    return command


def get_next_station_after_stop(algorithm, source_dir):
    """
    Should the training process (performed by `scikit_gym.py`) be interrupted
    by any reasons, this function is responsible for detecting the last fitted
    estimator and return the correct value for the `warm_start` parameter in
    the next run of script `scikit_gym.py`. Such value corresponds to an Statino
    ID in the format 'A000'.

    Parameters
    ------------
    `algorithm` : str
        Training algorithm

    `source_dir` : str
        Data source directory depending on the option 'idw' either original data
        or imputed data can be used to train the models.

    Returns
    ----------
    `station` : str, None
        ID of the next station to be passed as `warm_start` argument in the next
        run of `scikit_gym.py`. If no action is required `None` is returned.

    See also
    ----------
    Check the functionality of the function `check_execution()` in `./scikit_gym.py`
    """

    os.chdir(source_dir)

    station_dirs = []

    # read only files from the station directories
    for item in os.listdir():
        if os.path.isdir(item) and len(item) == len('A000'):
            station_dirs.append(item)

    model_names = {
        'mlp':  'MLPRegressor',
        'rf':   'RandomForestRegressor',
        'extr': 'ExtraTreesRegressor',
        'xgb':  'XGBRegressor',
        'svr':  'SVRegressor',
        'sr':   'StackingRegressor',
    }

    for station in sorted(station_dirs):

        # the current station hasn't trained any models yet
        if not os.path.exists( os.path.join(station, 'models') ):
            os.chdir( os.path.join('..', '..') )
            return station

        # model name to be compared to the model files
        model_name = model_names[algorithm]

        for model_file in os.listdir( os.path.join(station, 'models') ):

            # algorithm already training in the current station
            if model_name in model_file:
                break

        else:

            # the algorithm in the in the directory `models/` in the curernt station,
            # because of the ordering, none of the next stations will contain this
            # model, hence, the current station is the return value of the function.
            # The next run of `scikit_gym.py` will start from the current station.
            os.chdir( os.path.join('..', '..') )
            return station

    # the algorithm was already trained in all training stations
    # return to 'mixer/'
    os.chdir( os.path.join('..', '..') )

    return None


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description = 'Model Training Pipeline')

    arg_parser.add_argument('-v', '--verbose',
                            metavar = '',
                            required = False,
                            default = 0,
                            help = "Verbosity level",
                            type = int)

    arg_parser.add_argument('-dr', '--dry-run',
                            metavar = '',
                            required = False,
                            default = 0,
                            type = int)

    arg_parser.add_argument('-ff', '--final-fit',
                            metavar = '',
                            required = False,
                            default = 0,
                            type = int)

    args = vars(arg_parser.parse_args())


    pipeline_operations = {

        'original': {
            'grid_search': [],
            'performance_metrics': [],
            'cleanup': [],
            'final_fit': [],
        },

        'imputed': {
            'grid_search': [],
            'performance_metrics': [],
            'cleanup': [],
            'final_fit': [],
        }
    }

    # execute training pipeline steps with
    #  -> original data (0) and 
    #  -> imputed data (1)
    for data in ['original', 'imputed']:

        # directory with set of datasets to be used in training
        if data == 0:
            DATA_DIR = os.path.join('mixer', 'original')

        else:
            DATA_DIR = os.path.join('mixer', 'imputed')

        #
        # Grid Search steps in each of the algorithms. hyperparameters obtained
        # here are used in the remaining executions of the pipeline
        #
        for alg in ['xgb', 'mlp', 'rf', 'extr', 'svr']:

            # Checks which algorithms have to be executed again owing to any kind
            # of interruption during training.
            next_station_id = get_next_station_after_stop(alg, DATA_DIR)

            if next_station_id == None:
                continue

            pipeline_operations[ data ]['grid_search'].append(
                {
                    'script': 'scikit_gym.py',
                    'args': [
                        '-fit',
                        '-alg', alg,
                        '-pf', os.path.join('..', '..', 'param_grids', f'{alg}_param_grid.pkl'),
                        '-dr', args['dry_run'],
                        '-d', data,
                        '-ws', next_station_id,
                        '-v', args['verbose'],
                    ]
                }
            )

        # Generates intermediate files used in the remaining steps of the pipeline
        # Used twoce in the current script
        pipeline_operations[ data ]['performance_metrics'].append(
            {
                'script': 'performance_metrics.py',
                'args': [
                    '-v', args['verbose'],
                    '-t', data,
                ]
            }
        )

        pipeline_operations[ data ]['performance_metrics'].append(
            # obtains the best regressors per algorithm in each station
            {
                'script': 'consolidate_results.py',
                'args': [
                    '-d', data,
                ]
            },
        )

        # once all the estimators are obtained in the grid search routines
        # the best regressors are mantained and the remaining removed in order
        # to save space. The Stacking Ensemble is also among these kept models.
        pipeline_operations[ data ]['cleanup'].append(
            {
                'script': os.path.join('mixer', 'station_manager.py'),
                'args': [
                    '-d', data,
                    'clear',
                    '-m',
                    '-yes',
                ]
            }
        )


        # comprehends all the models with the respective best hyperparameters
        # At this point the stacking ensemble is also added to the list of models
        # to be fitted with the best hparams
        for alg in ['mlp', 'rf', 'extr', 'xgb', 'svr', 'sr']:

            # Checks which algorithms have to be executed again owing to any kind
            # of interruption during training.
            next_station_id = get_next_station_after_stop(alg, DATA_DIR)

            if next_station_id == None:
                continue

            pipeline_operations[ data ]['final_fit'].append(
                {
                    'script': 'scikit_gym.py',
                    'args': [
                        '-fit',
                        '-alg', alg,
                        '-hp',
                        '-d', data,
                        '-dr', args['dry_run'],
                        '-v', args['verbose'],
                    ]
                }
            )

    # ---------------------------------------------
    # ---------------------------------------------

    # generate hparam grids for all used algorithms
    common_operations = {
        'script': os.path.join('param_grids', 'gen_param_grid.py'),
        'args': [],
    }

    os.system(command = format_command(script = common_operations['script'],
                                       arg_list = common_operations['args']))

    if args['final_fit']:

        for data_type in pipeline_operations.keys():
            for step in pipeline_operations[data_type]['final_fit']:
                os.system(command = format_command(script = step['script'],
                                                   arg_list = step['args']))

    else:

        for data_type in pipeline_operations.keys():
            for procedure in pipeline_operations[data_type]:
                for step in pipeline_operations[data_type][procedure]:
                    command = format_command(script = step['script'],
                                                       arg_list = step['args'])
                    os.system(command = format_command(script = step['script'],
                                                       arg_list = step['args']))
