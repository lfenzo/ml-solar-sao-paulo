import os
import re
import numpy as np
import pandas as pd
import argparse
import joblib

from sklearn.metrics import mean_squared_error, \
                            mean_absolute_error, \
                            mean_absolute_percentage_error, \
                            r2_score


def mean_bias_error(y_true, y_pred):
    """
    Implements the calculation of the Mean Bias Error.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_pred - y_true)


def get_regressors(path):
    """
    Generator to yield all regressors within `path`. Also separates
    the regressors trained with scikit-learn from the ones trained
    with tensorflow (depracted)
    """

    files = os.listdir(path)

    for file in files:

        model_to_load = os.path.join(path, file)

        if 'sk_' in file:
            timestamp = ''.join(re.split('[.|_]', model_to_load)[2])
            yield joblib.load(model_to_load), timestamp

        # TODO precisa conferir se a timestamp vai estar certa
        # problema pra no build_composition.py por conta da formatação da timastamp
        elif 'tf_' in file:
            timestamp = '-'.join(re.split('[.|_]', model_to_load)[4:10])
            yield keras.models.load_model(model_to_load), timestamp


def get_scores(regressor: 'fitted regressor', X, Y, scaler: 'scaler used in the transformaations'):
    """
    Calculates and returns the performance metrics for `regressor`

    Parameters
    ----------
    `regressor`
        Fitted regressor loaded from the model files

    `X`: array-like
        Test/Validation data

    `Y`: array-like
        Test/Validation target values

    `scaler`: object
        Target scaler used to performe the `inverse_transform` in the predicted values

    Returns
    -----------
    Two dictionaries with:
        1: the predictions and excpected values for the estimators
        2: a performance dictionary with respect to the `regressor` with the following keys:
            `mae` : Mean Absolute Error\\
            `rmse`: Root Mean Squared Error\\
            `mape`: Mean Absolute Percentage Error\\
            `mbe` : Mean Bias Error\\
            `r2`  : R2 Scoring\\
            `h-params`: estimator hyperparameters\\
    """

    yhat = regressor.predict(X)

    results = {'model': [regressor.__class__.__name__]}
    predictions = {}

    # performs the inverse transform
    yhat = scaler.inverse_transform(yhat.reshape(-1, 1))
    Y = scaler.inverse_transform(Y)

    results['mae']  = [ mean_absolute_error(y_true = Y, y_pred = yhat) ]
    results['rmse'] = [ mean_squared_error(y_true = Y, y_pred = yhat, squared = False) ]
    results['mape'] = [ mean_absolute_percentage_error(y_true = Y, y_pred = yhat) ]
    results['mbe']  = [ mean_bias_error(y_true = Y, y_pred = yhat) ]
    results['r2']   = [ r2_score(y_true = Y, y_pred = yhat) ]

    # keras classes have no `get_params()` method (depracted since keras is no longer 
    # used in the project)
    if regressor.__class__.__name__ == 'Sequential':
        results['h-params'] = ' '
    else:
        results['h-params'] = str(regressor.get_params())

    predictions['estimado']   = [pred[0] for pred in yhat]
    predictions['valor_real'] = [real[0] for real in    Y]

    return results, predictions


def show_scores(estimator, results):
    """
    Shows the results in stdout from the `results` dictionary
    """

    print(f'Showing results of {estimator.__class__.__name__}:')
    print(f'\tRMSE:\t{results["rmse"]}')
    print(f'\tMAE:\t{results["mae"]}')
    print(f'\tMAPE:\t{results["mape"]}')
    print(f'\tMBE:\t{results["mbe"]}')
    print(f'\tR2:\t{results["r2"]}')


def save_predictions(regressor, values: dict, timestamp, station_id):
    """
    Saves the `regressor` predictions as well as the excpected real values in each timestamp
    """

    filename = os.path.join(
        station_id,
        'predictions',
        f'{station_id}_predictions_{regressor.__class__.__name__}_{timestamp}.csv'
    )

    dataframe = pd.DataFrame(data = values)
    dataframe.to_csv(filename, index = False)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description = 'Consolidates the results of all \
                                         fitted regressors')

    arg_parser.add_argument('-t', '--target',
                            metavar = '',
                            choices = ['original', 'imputed', 'merged'],
                            required = True,
                            help = "Station directory targeted: (original, imputed or merged).")

    arg_parser.add_argument('-v', '--verbose',
                            metavar = '',
                            required = False,
                            default = 0,
                            help = "Verbosity level: 0 (silent) otherwise verbose.",
                            type = int)

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

    for i, station_id in enumerate( sorted(station_dirs) ):

        if args['verbose']:
            print(f'[{i:2} / {len(station_dirs) - 1}]\t', end = '')
            print(f'Commencing performance evaluations on Station {station_id:2}')

        x = np.load( os.path.join(station_id, 'x_test.npy') )
        y = np.load( os.path.join(station_id, 'y_test.npy') )

        # loads the target scaler in order to perform `inverse_transform`
        scaler = joblib.load( os.path.join(station_id, f'{station_id}_scaler_target.dat') )

        final_results_per_station = pd.DataFrame()

        if not os.path.exists( os.path.join(station_id, 'predictions') ):
            os.mkdir( os.path.join(station_id, 'predictions' ) )

        # obtains all scikit-learn regressors in the current station directory
        for reg, timestamp in get_regressors( os.path.join(os.getcwd(), station_id, 'models') ):

            results, predictions = get_scores(regressor = reg,
                                              X = x,
                                              Y = y,
                                              scaler = scaler)

            results['timestamp'] = timestamp

            result_parcial = pd.DataFrame(data = results)
            final_results_per_station = final_results_per_station.append(result_parcial,
                                                                         ignore_index = True)

            if args['verbose']:
                show_scores(estimator = reg, results = results)

            final_results_per_station.to_csv(os.path.join(station_id, f'{station_id}_results.csv'),
                                             index = False)

            # saves the predictions and the expected values in a different dataframe
            if args['verbose']:
                print(f'\tSaving predictions ...')

            # called from within the selected directory ('original' or 'imputed')
            save_predictions(regressor = reg,
                             values = predictions,
                             timestamp = timestamp,
                             station_id = station_id)
