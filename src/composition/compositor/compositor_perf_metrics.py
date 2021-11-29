"""
Separa o conjunto de teste para verificar os errors estação a estação
"""

import os
import joblib
import numpy as np
import pandas as pd

from performance_metrics import mean_bias_error

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def get_n_samples_used(dataframe):
    """
    Obtem o número de observações (samples) no conjunto de teste utilizado
    para medir a performance.

    Parameters
    -------------
    dataframe : pd.DataFrame
        Dataframe contendo as previsões realizadas pela composição

    Returns
    ------------
    n_samples : int
        Número de número de observações (samples) no conjunto de teste.
    """

    return dataframe.shape[0]


if __name__ == '__main__':

    # carregando os scalers
    data_scaler = joblib.load('./scalers/data_scaler.dat')
    target_scaler = joblib.load('./scalers/target_scaler.dat')

    # mudar depois
    model = joblib.load('./models/StackingRegressor_2021-08-11-17-12-40.dat')

    compositor_performance = {
        'station': [],
        'rmse': [],
        'mae': [],
        'mbe': [],
        'r2': [],
        'n_samples': [],
    }

    for file in os.listdir('./station_test_sets'):

        station_test_dataset = pd.read_csv(f'./station_test_sets/{file}')

        # test X e text Y já transformados pelos scalers.
        test_X = data_scaler.transform(station_test_dataset.drop(columns = ['target']).values)
        test_Y = target_scaler.transform(station_test_dataset.loc[:, 'target'].values.reshape(-1, 1))

        y_real = target_scaler.inverse_transform( test_Y )
        y_hat  = target_scaler.inverse_transform( model.predict(test_X).reshape(-1, 1) )

        compositor_performance['station'].append( file.split('_')[0] )
        compositor_performance['rmse'].append( mean_squared_error(y_real, y_hat, squared = False) )
        compositor_performance['mae'].append( mean_absolute_error(y_real, y_hat) )
        compositor_performance['mbe'].append( mean_bias_error(y_real, y_hat) )
        compositor_performance['r2'].append( r2_score(y_real, y_hat) )
        compositor_performance['n_samples'].append( get_n_samples_used(station_test_dataset) )


    compositor_perf_df = pd.DataFrame().from_dict(compositor_performance)
    compositor_perf_df.to_csv('./ml_compositor_perf_metrics.csv', index = False)

