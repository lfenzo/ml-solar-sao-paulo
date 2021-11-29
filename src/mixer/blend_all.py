"""
Merges the models trained in `original/` and `imputed/` into `merged_stations/`

The ideia is to select for every station the best model comparing the models
trained with imputed data and orignal data. This models, as well as the contents
of the whole directory are copied to `merged_stations/` (which is analogue to
both `original/` ans `imputed/` in terms of contens: transformer, performance csvs).

Note that `performance_metrics.py` and `consolidate_performance.py` should have been
already executed for both imputed and original data.

"""

import os
import shutil
import numpy as np
import pandas as pd


def get_best_model_directory_paths(models_dataframe) -> dict:

    best_models_paths = {}

    for _, row in models_dataframe.iterrows():

        best_regressor_signature = f'{row["model"]}_{row["timestamp"]}'
        directory = 'imputed' if row['is_interp'] else 'original'

        for file in os.listdir( os.path.join(directory, row["station"], 'models') ):

            if best_regressor_signature in file:
                best_models_paths[ row["station"] ] = os.path.join(os.getcwd(),
                                                                   directory,
                                                                   row["station"])

    return best_models_paths


if __name__ == '__main__':

    OUTPUT_DIR = os.path.join('..', 'merged_stations')

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    best_interp   = pd.read_csv( os.path.join('imputed', 'best_models.csv') )
    best_original = pd.read_csv( os.path.join('original', 'best_models.csv') )

    all_models = pd.merge(left = best_interp,
                          right = best_original,
                          on = 'station',
                          how = 'outer',
                          suffixes = ('_interp', '_original'))

    best_info = {
        'station'   : [],
        'model'     : [],
        'timestamp' : [],
        'rmse'      : [],
        'mae'       : [],
        'mbe'       : [],
        'r2'        : [],
        'is_interp' : [],
        'h-params'  : [],
    }

    for idx, row in all_models.iterrows():

        # this station has only been added beacuse of IDW imputation,
        # so, there is not 'origial' station to be compared.

        # since it only appears in one of the rows of the dataframe
        # the value is replaced with NaN which is an instance of float
        if isinstance(row['timestamp_original'], float):

            # coloca cada coluna desta linha do dataframe
            for key in best_info.keys():

                if key == 'station':
                    best_info[key].append( row[f'{key}'] )

                elif key == 'is_interp':
                    best_info[key].append( True )

                else:
                    best_info[key].append( row[f'{key}_interp'] )


        # stations using both interolated and original data (must be selected)
        else:

            choice = 'interp' if row['rmse_interp'] < row['rmse_original'] else 'original'

            for key in best_info.keys():

                if key == 'station':
                    best_info[key].append( row[f'{key}'] )

                elif key == 'is_interp':
                    best_info[key].append( True if choice == 'interp' else False )

                else:
                    best_info[key].append( row[f'{key}_{choice}'] )


    best_overall = pd.DataFrame().from_dict(best_info)
    best_overall = best_overall.sort_values(by = ['station'])

    best_overall.to_csv('best_overall.csv', index = False)
    best_overall.to_csv(os.path.join(OUTPUT_DIR, 'best_overall.csv'), index = False)


    best_stations_src = get_best_model_directory_paths(best_overall)

    # copies the whole directoiies relative to the best models into `merged_stations/`
    for station_id, path in best_stations_src.items():
        shutil.copytree(src = path,
                        dst = os.path.join(OUTPUT_DIR, station_id),
                        dirs_exist_ok = True)

