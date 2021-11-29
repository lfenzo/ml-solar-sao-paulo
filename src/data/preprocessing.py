import os
import joblib
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, \
                                  MinMaxScaler, \
                                  RobustScaler

import haversine as hs
from haversine import Unit


from idw import InverseDistanceWeighting as IDW


def define_args():
    """
    Define all Command-Line Interface arguments used in the script.
    """

    ap = argparse.ArgumentParser(description = 'Performs preprocessing transformations',
                                 formatter_class = argparse.RawTextHelpFormatter)

    ap.add_argument('-i', '--input',
                    metavar = '',
                    required = True,
                    type = str,
                    help = 'Concatenated datafile to be preprocessed,')

    ap.add_argument('-sv', '--save-station',
                    metavar = '',
                    required = False,
                    default = True,
                    type = bool,
                    help = 'Saves a copy of the preprocessed site-specific dataframe. Default = True')

    ap.add_argument('-idw', '--idw-imputation',
                    metavar = '',
                    required = False,
                    default = 1,
                    type = int,
                    help = 'Performs Inverse Distance Weighting (IDW) impuattion. Default = True')

    ap.add_argument('-nrm', '--normalization_method',
                    metavar = '',
                    required = False,
                    default = 'rbst',
                    type = str,
                    help = 'Normalization methods to be used. Avaliable data scaling methods:' +
                        '\n\t\'std\': Standard Scaler' +
                        '\n\t\'rbst\': Robust Scaler' +
                        '\n\t\'mM\': Min Max Scaler')

    ap.add_argument('-v', '--verbose',
                    metavar = '',
                    required = False,
                    default = 0,
                    type = int,
                    help = "Verbosity level: 0 (silent) otherwise verbose.")

    return ap


def apply_pca(dataframe, features_to_transform: list, new_features: list):
    """
    Applies Principal Component Analysis to the selected set of features

    Parameters
    ----------
    `dataframel`: pd.DataFrame
        Site-specific dataframe to apply PCA to

    `features_to_transform`: list
        Names of Lisfeatures in which PCA will be applied

    `new_features`: list
        List of (new) transformed features in the dataframe

    Returns
    ----------
    `transformed_dataframe`: pd.DataFrame
        Pandas dataframe with the new PCA-transformed features without the
        transformed features.

        all_featuers = dataframe features - features_to_convert + new_features

    `pcs_transformer`: scikit-learn transformer
        Fitted PCA transformer object ready for use.

    """

    # gets columns values to be transformed
    columns_to_transform = dataframe[features_to_transform].copy()

    pca_transformer = PCA(n_components = len(new_features))

    transformed = pca_transformer.fit_transform(columns_to_transform)


    if len(new_features) == 1:
        dataframe[f'{new_features[0]}'] = transformed

    else:
        # Attention! numpy indexing notation used (instead of pandas'
        # notation) that heappens because 'transformed' is a numpy array
        for i, new_column in enumerate(new_features):
            dataframe[new_column] = transformed[:, i]

    return dataframe.drop(columns = features_to_transform), pca_transformer


def drop_ramaining_nas(dataframe):
    """
    Drop ramaining NAs left after imputation.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe in which NAs will be dropped

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe without NA values

    """

    return dataframe.dropna()


def fill_remaining_nas(dataframe):
    """
    Interpolates the NA values that could not be filled using
    IDW impuattion

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe in which NAs will be interpolated

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe without NA values

    """

    return dataframe.interpolate(method = 'pad', limit = 2)


def assign_nas(dataframe):
    """
    Assign NAs to -9999.0 values in `dataframe`. Values registered
    after 01/01/2019 have such values already corresponding to NAs

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe in which -9999.0 will be replaced

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with no -9999.0 values.

    """

    return dataframe.replace(to_replace = -9999.0, value = np.nan)


def set_features_to_float32(dataframe, feature_list: list):
    """
    Applies type conversion to all features in `features_list` to Float32

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which features in `feature_list` will be
        converted.

    `feature_list`: list
        List of features to be converted.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with converted features.

    """

    for feature in feature_list:
        dataframe[f'{feature}'] = pd.to_numeric(dataframe[f'{feature}'], downcast = 'float')

    return dataframe


def filtra_radiacao_solar(dataframe):
    """
    Replaces invalid Solar Radiation values with NAs which will be later
    interpolated either with IDW or with pandas' interpolation method.

    Also selects observation in the period 10-22 UTC, dropping the remaining

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which invalid solar radiation feature values be
        replace with NAs.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with replaced values in the valid period.

    """

    dataframe = dataframe.copy()

    dataframe.loc[ dataframe['radiacao_global'] >= 8000, 'radiacao_global' ] = np.nan

    # os valores precisam ser esses para os limites [11, 21]
    selected_hours = dataframe['hour'].isin(range(10, 22))

    return dataframe.loc[selected_hours]


def handle_inconsistencies(dataframe):
    """
    Handle inconsistent observation values e.g. relative humidigy < 0
    Such values are replaced with NAs and are later interpolated either
    with IDW imputation of with pandas interpolation method.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which invalid feature values be replace
        with NAs.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with replaced values.

    """

    # permite que as mudanças feitas na função permaneçam no dataframe
    dataframe = dataframe.copy()

    dataframe.loc[ dataframe['precipitacao_total'] < 0, 'precipitacao_total' ] = np.nan

    dataframe.loc[ dataframe['pressao_atmosferica'] < 0, 'pressao_atmosferica' ] = np.nan
    dataframe.loc[ dataframe['pressao_atmosferica_max'] < 0, 'pressao_atmosferica_max' ] = np.nan
    dataframe.loc[ dataframe['pressao_atmosferica_min'] < 0, 'pressao_atmosferica_min' ] = np.nan

    dataframe.loc[ dataframe['umidade_relativa'] < 0, 'umidade_relativa' ] = np.nan
    dataframe.loc[ dataframe['umidade_relativa_max'] < 0, 'umidade_relativa_max' ] = np.nan
    dataframe.loc[ dataframe['umidade_relativa_min'] < 0, 'umidade_relativa_min' ] = np.nan

    dataframe.loc[ dataframe['temperatura'] < 10, 'temperatura' ] = np.nan
    dataframe.loc[ dataframe['temperatura_max'] < 10, 'temperatura_max' ] = np.nan
    dataframe.loc[ dataframe['temperatura_min'] < 10, 'temperatura_min' ] = np.nan

    dataframe.loc[ dataframe['vento_velocidade'] < 0, 'vento_velocidade' ] = np.nan

    return dataframe


def _remove_utc_from_hour(hour_field):
    """
    Auxiliary function for 'process_time_units()'
    """
    return hour_field.replace('UTC', '') if 'UTC' in hour_field else hour_field


def _standardize_date_format(date_field):
    """
    Auxiliary function for 'process_time_units()'
    """
    return date_field.replace('-', '/') if '-' in date_field else date_field


def process_time_units(dataframe):
    """
    Processes time units by parsing the formatted datetime features. This
    function standardizes the datetime record format and creates the feature
    'datetime' from which other time-related features can be obtained.

    Note that this function also sets the 'datetime' feature as index of the
    dataframe. This allows time-related sorting.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which time units will be processed.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with processed datetmie-related features

    """

    dataframe['hour'] = dataframe['hour'].apply(_remove_utc_from_hour)
    dataframe['data'] = dataframe['data'].apply(_standardize_date_format)

    # adding 'format' kw speeds up dramatically parsing times
    dataframe['datetime'] = pd.to_datetime(arg = dataframe['data'] + ' ' + dataframe['hour'],
                                           utc = True,
                                           format = '%Y/%m/%d %H:%M')

    dataframe.drop(columns = ['data', 'hour'], inplace = True)

    dataframe['doy']  = dataframe['datetime'].dt.dayofyear.astype('int32')
    dataframe['year'] = dataframe['datetime'].dt.year.astype('int32')
    dataframe['hour'] = dataframe['datetime'].dt.hour.astype('int32')

    dataframe.sort_values(by = ['datetime'], inplace = True)

    return dataframe.set_index(keys = ['datetime'], drop = True)


def remove_features(dataframe, feature_list: list):
    """
    Removes all features in `feature_list` from `dataframe`.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which features will be removed.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe without the features specified in `feature_list`.

    """

    return dataframe.drop(columns = feature_list)


def shift_solar_radiation(dataframe, new_feature: str, hours_ahead: int):
    """
    Shifts solar radiation by creating the target feature: solar radiation
    in the next hour. This operation requires the dataframe to be sorted by
    datetime.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe with global radiation feature to be shifted

    `hours_ahead`: int
        Number of hours (in the future) to shift in relation to the current
        hour.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with new target feature.

    Example
    ----------

    Hour    Current Radiation   Radiation in the Next Hour (radiation(h + 1))
    10      100                 101
    11      101                 102
    12      102                 103
    13      103                 ...

    """

    dataframe[f'{new_feature}'] = dataframe['radiacao_global'].shift(periods = -hours_ahead)
    return dataframe


def handle_radiation_outliers(dataframe, feature: str, threshold):
    """
    Handle outliers in `feature`. A `threshold` is also used in the upper limmit
    to drop such outlier values..

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Site-specific dataframe which outliers in `feature` will be dropped

    `feature`: int
        Feature to be evaluated

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe without the features specified in `feature_list`.

    """

    for hora in range(10, 22):

        Q3 = dataframe.loc[dataframe['hour'] == hora][f'{feature}'].quantile(0.75)
        Q1 = dataframe.loc[dataframe['hour'] == hora][f'{feature}'].quantile(0.25)

        IQR = Q3 - Q1

        upper_limit = (Q3 + 1.5 * IQR)
        lower_limit = (Q1 - 1.5 * IQR)

        # obtain all indexes that should be dropped owing to the outlier values
        outlier_rows = dataframe[  (dataframe['hour'] == hora) &
                                 ( (dataframe[f'{feature}'] <= lower_limit) |
                                   (dataframe[f'{feature}'] >= (upper_limit) + threshold) )  ].index

        dataframe.loc[outlier_rows, feature] = np.nan

    return dataframe


def save_dataframe(dataframe, file_prefix, fmt = 'ftr'):
    """
    Saves the preprocessed `dataframe` to disk.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Dataframe to be saved.

    `file_prefix`: str
        File prefix, filename without extension.

    `fmt`: str
        FIle extension: '.csv' or '.ftr'.

    """

    # to be created inside 'data/'
    if not os.path.exists('processed_stations'):
        os.mkdir('processed_stations')

    if fmt == 'ftr':
        # necessario resetar o index para salvar como feather
        dataframe.reset_index(inplace = True, drop = True)
        dataframe.to_feather( os.path.join('processed_stations', f'{file_prefix}.ftr') )

    elif fmt == 'csv':
        dataframe.to_csv( os.path.join('processed_stations', f'{file_prefix}.csv') )


def normalize_dataframe(dataframe, method: 'normalization method', scaler = None):
    """
    Performs data normalization with `dataframe` data using `method`.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Dataframe to be normalized (it can contain a single feature -- series).

    `method`: str
        Normalization method to be applied to data. Avaliable methods:
            - `std` : Standardization scaling
            - `mM`  ; Min-Max scaling
            - `rbst`; Robust scaling

    `scaler`: sklearn.Transformer, default = None
        Previously fitted scaler object to be used during data normalization.

    Returns
    ---------
    `normalized_data`: np.ndarray
        Numpy array with normalized data. Note that `normalized.shape == dataframe.shape`.

    `scaler`: sklearn.Transformer
        Fitted scaler used to normalizeed the data.
    """

    data = dataframe.values

    # single feature being scaled (used when scaling the target feature)
    if isinstance(dataframe, pd.Series):
        data = data.reshape(-1, 1)

    if scaler == None:

        if method == 'std':
            scaler = StandardScaler().fit(data)

        elif method == 'mM':
            scaler = MinMaxScaler().fit(data)

        elif method == 'rbst':
            scaler = RobustScaler().fit(data)

        else:
            raise KeyError(f'Normalization method \'{method}\' not found.')

    normalized_data = scaler.transform(data)

    return normalized_data, scaler


def split_data_targets(dataframe, target, shuffle = False):
    """
    "Horizontally" splits a dataframe into `data` and `target` sets.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Dataframe to be split.

    `target`: str
        Target feature.

    `shuffle`: bool, default = False
        Shuffle the dataframe before splitting.

    Returns
    ----------
    `data_dataframe`: pd.DataFrame
        Dataframe containing all features except `target`.

    `target_dataframe`: pd.DataFrame
        Dataframe containing only `target` feature.

    """

    if shuffle:
        dataframe = dataframe.sample(frac = 1)

    return dataframe.drop(columns = target), dataframe[[target]]


def set_most_frequent_lat_lon_alt(dataframe):
    """
    Addresses geographical coordinates duality due to register faults in
    the meteorological stations.

    Duality is removed from the features:
    - latitude
    - longitude
    - altitude

    Instead of dropping the samples with different values of these features
    the approach consists in selecting the most frequent values (registered
    for the longest period of time) and overwrite the remaining different
    values.

    Parameters
    ----------
    `dataframe`: pd.DataFrame
        Dataframe to be corrected.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with corrected duality in geographical coordinates.

    """

    # `value_counts()` already sorts the different combinations in frequency order
    # once it's done one must only select the first row (index[0]) of this generate dataframe
    lat, lon, alt = dataframe[['latitude', 'longitude','altitude']].value_counts().index[0]

    dataframe['latitude']  = lat
    dataframe['longitude'] = lon
    dataframe['altitude']  = alt

    return dataframe


def get_nearby_stations(current_station, other_stations: dict, dist_thresh) -> (dict, list):
    """
    In the context of IDW imputation, this function obtains the nearby stations
    eligible with respect to the distance.

    Parameters
    ----------
    `current_station`: pd.DataFrame
        Current dataframe in the IDW imputation process. This station is the reference
        point when searching for nearby stations.

    `other_stations`: dict
        Dict with station IDs as keys and the respective dataframes as values.

    `dist_thresh`: float
        Distance threshold adopted when selecting the closest stations. Stations with
        distance to `current_station` grater than `dist_thresh` will not be selected.

    Returns
    ----------
    `distances`: dict
        Dict in the format ('station_id': float(distance)).

    `nearby_scations`: list
        List of dataframes corresponding to the closest stations in relation
        to `current_station`.

    """

    ref_lat, ref_lon = *current_station['latitude'].unique(), *current_station['longitude'].unique()

    distances = {}
    nearby_scations = []

    for id_est in other_stations.keys():

        station = other_stations[id_est]
        est_lat, est_lon = *station['latitude'].unique(), *station['longitude'].unique()

        dist = hs.haversine((ref_lat, ref_lon), (est_lat, est_lon),
                            unit = Unit.KILOMETERS)

        if dist <= dist_thresh:
            distances[id_est] = dist
            nearby_scations.append(station.copy())

    return distances, nearby_scations


def interpolate_feature(current_station, feature, distances: dict, nearby_stations: list, min_stations) -> pd.DataFrame:
    """
    Performs IDW-based imputation in a `df_station` dataframe with respect to `feature` in
    all its available NA observations when possible.

    Parameters
    ----------
    `current_station`: pd.DataFrame
        Site-specific dataframe to have its NA values to be imputed by IDW interpolation.

    `distances`: dict
        Dict in the format ('station_id': float(distance)) with nearby station distances.

    `nearby_stations`: list
        List of dataframes corresponding to the nearby stations.

    `min_stations`: int
        Minimum number of reference stations in order to interpolate `feature` in `current_station`.

    Returns
    ----------
    `dataframe`: pd.DataFrame
        Dataframe with IDW interpolated values in `feature`.

    """

    if len(nearby_stations) >= min_stations:

        interpolator = IDW()
        na_indexes = current_station.index[ current_station[feature].apply(np.isnan) ]

        for index in na_indexes:
            for station in nearby_stations:

                # it is possible that 'station' doesn't have that index
                try:

                    if not np.isnan(station.at[index, feature]):

                        interpolator.add_point(
                            dist = distances[ station.at[index, 'station_id'] ],
                            value = station.at[index, feature]
                        )

                except Exception:
                    continue

            if interpolator.total_points() >= min_stations:
                current_station.at[index, feature] = interpolator.interpolate()

            interpolator.dispose_points()

    return current_station


def separate_stations(all_stations) -> dict:
    """
    Obtains a dictionary with all stations dataframes in formar:
        - key: Station ID
        - value: station dataframe

    Parameters
    ----------
    `all_stations`: pd.DataFrame
        Dataframe containing all avaliable data.

    Returns
    ----------
    `dataframe_dict`: dict
        Dict with dataframes of each station
    """

    dataframe_dict = {}

    # the name 'station_id' is first set in './file_concatenator.py'
    for station_id in all_stations['station_id'].unique():
        dataframe_dict[station_id] = all_stations[ all_stations['station_id'] == station_id ].copy()

    return dataframe_dict


def get_remaining_stations(all_stations: dict, exclude: str) -> dict:
    """
    In the context of IDW, this function selects all stations except the one being
    interpolated (`exclude`).

    Parameters
    ----------
    `all_stations`: dict
        Dict containing station id as keys and station dataframes as values.

    `exclude`: str
        Station ID to be excluded.

    Returns
    ----------
    `remaining_stations`: dict
        Dict with all stations except `excluded`.

    """
    return {key: value for key, value in all_stations.items() if key != exclude}


if __name__ == '__main__':

    ap = define_args()
    args = vars(ap.parse_args())


    if args['idw_imputation']:
        OUTPUT_DIR = os.path.join('..', 'mixer', 'imputed')
    else:
        OUTPUT_DIR = os.path.join('..', 'mixer', 'original')

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


#    if not os.path.exists('../stations'):
#        os.mkdir('../stations')

    if args['input'] not in os.listdir(os.getcwd()):
        raise KeyError(f'Input file \'{args["input"]}\' not found in current path ')

    elif args['input'].split('.')[1] != 'ftr':
        raise KeyError(f'File \'{args["input"]}\' has no extension \'.ftr\'.')


    all_stations = pd.read_feather(args['input'])

    stations_dict = separate_stations(all_stations = all_stations)

    if args['verbose']:
        print('Commencing preprocessing procedures ...')

    for station_id in sorted( list(stations_dict.keys()) ):

        stations_dict[station_id] = set_most_frequent_lat_lon_alt(
            dataframe = stations_dict[station_id],
        )

        stations_dict[station_id] = process_time_units(
            dataframe = stations_dict[station_id],
        )

        features_to_convert = [
            'precipitacao_total',
            'pressao_atmosferica',
            'pressao_atmosferica_max',
            'pressao_atmosferica_min',
            'radiacao_global',
            'temperatura',
            'temperatura_max',
            'temperatura_min',
            'ponto_orvalho',
            'ponto_orvalho_max',
            'ponto_orvalho_min', 'umidade_relativa',
            'umidade_relativa_max',
            'umidade_relativa_min',
            'vento_direcao',
            'vento_velocidade',
            'latitude',
            'longitude',
            'altitude',
        ]

        stations_dict[station_id] = set_features_to_float32(dataframe = stations_dict[station_id],
                                                            feature_list = features_to_convert)

        stations_dict[station_id] = filtra_radiacao_solar(dataframe = stations_dict[station_id])

        stations_dict[station_id] = shift_solar_radiation(dataframe = stations_dict[station_id],
                                                          hours_ahead = 1,
                                                          new_feature = 'rad_prox_hora')

        stations_dict[station_id] = handle_inconsistencies(dataframe = stations_dict[station_id])

        stations_dict[station_id] = assign_nas(dataframe = stations_dict[station_id])


        for radiationn_feature in ['radiacao_global', 'rad_prox_hora']:

            stations_dict[station_id] = handle_radiation_outliers(
                dataframe = stations_dict[station_id],
                feature = radiationn_feature,
                threshold = 300
            )

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    if args['idw_imputation']:

        if args['verbose']:
            print("Execution Imputation Routines...")

        for num_est, station_id in enumerate( sorted(list(stations_dict.keys())) ):

            if args['verbose']:
                print(f"imputation in station {station_id}: {num_est} of ", end = '')
                print(f'{len(stations_dict.keys())}', end = '\t')

            features_to_interpolate = [
                'precipitacao_total',
                'pressao_atmosferica',
                'pressao_atmosferica_max',
                'pressao_atmosferica_min',
                'radiacao_global',
                'temperatura',
                'temperatura_max',
                'temperatura_min',
                'ponto_orvalho',
                'ponto_orvalho_max',
                'ponto_orvalho_min',
                'umidade_relativa',
                'umidade_relativa_max',
                'umidade_relativa_min',
                'vento_direcao',
                'vento_velocidade',
                'rad_prox_hora',
            ]

            other_stations = get_remaining_stations(all_stations = stations_dict,
                                                    exclude = station_id)

            distances, nearby_stations = get_nearby_stations(
                current_station = stations_dict[station_id],
                other_stations = other_stations,
                dist_thresh = 150,
            )

            for feature in features_to_interpolate:

                stations_dict[station_id] = interpolate_feature(
                    current_station = stations_dict[station_id].copy(),
                    feature = feature,
                    distances = distances,
                    nearby_stations = nearby_stations,
                    min_stations = 3,
                )

            if args['verbose']:
                print("done!")

        stations_dict[station_id] = fill_remaining_nas(dataframe = stations_dict[station_id])

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # used for checking which stations have been used for training and which were disposed
    station_status_info = {
        'station': [],
        'status': [],
        'lat': [],
        'lon': [],
    }

    if args['verbose']:
        print("Reducing dimensionality with PCA...")

    for station_id in sorted(list(stations_dict.keys())):

        stations_dict[station_id] = drop_ramaining_nas(dataframe = stations_dict[station_id])

        stations_dict[station_id], pca_transformer_pressao = apply_pca(
            dataframe = stations_dict[station_id],
            features_to_transform = ['pressao_atmosferica',
                                     'pressao_atmosferica_max',
                                     'pressao_atmosferica_min'],
            new_features = ['pca_pressao'],
        )

        stations_dict[station_id], pca_transformer_umidade = apply_pca(
            dataframe = stations_dict[station_id],
            features_to_transform = ['umidade_relativa',
                                     'umidade_relativa_max',
                                     'umidade_relativa_min'],
            new_features = ['pca_umidade_1', 'pca_umidade_2'],
        )

        stations_dict[station_id], pca_transformer_temperatura = apply_pca(
            dataframe = stations_dict[station_id],
            features_to_transform = ['temperatura',
                                     'temperatura_max',
                                     'temperatura_min'],
            new_features = ['pca_temperatura'],
        )

        stations_dict[station_id], pca_transformer_orvalho = apply_pca(
            dataframe = stations_dict[station_id],
            features_to_transform = ['ponto_orvalho',
                                     'ponto_orvalho_max',
                                     'ponto_orvalho_min'],
            new_features = ['pca_ponto_orvalho'],
        )


        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------

        station = stations_dict[station_id].copy()

        # obtains metadata that should be stored inside the station directories
        station_metadata = stations_dict[station_id][ ['latitude',
                                                       'longitude',
                                                       'altitude',
                                                       'station_id'] ].drop_duplicates()

        # drops no longer relevant features in the dataframes
        ready_dataframe = station.drop(columns = ['station_id',
                                                  'latitude',
                                                  'longitude',
                                                  'altitude'])

        if args['save_station']:
            save_dataframe(dataframe = ready_dataframe,
                           file_prefix = station_id,
                           fmt = 'csv')


        # train/test size os defined by the amount of samples in each split
        df_test  = ready_dataframe[ ready_dataframe['year'].isin(range(2019, 2022)) ]
        df_train = ready_dataframe[ ready_dataframe['year'] < 2019 ]

        # ------------------------------------------------------------
        # ------------------------------------------------------------

        if df_test.shape[0] == 0 or df_train.shape[0] == 0:
            print('Not enough samples in test split of this statinn')
            continue

        MIN_REQUIRED_SAMPLES = (21 - 11 + 1) * 365 * 7

        station_status_info['lat'].append(station['latitude'].unique()[0])
        station_status_info['lon'].append(station['longitude'].unique()[0])

        if not stations_dict[station_id].shape[0] >= MIN_REQUIRED_SAMPLES:

            # station wont be used but keep the metadata
            station_status_info['station'].append(station_id)
            station_status_info['status'].append('skip')

            if args['verbose']:
                print(f"Insufficient training sample count for station {station_id}.")

            continue

        # ------------------------------------------------------------
        # ------------------------------------------------------------

        # creates the output directori inside 'mixer' directory
        if not os.path.exists( os.path.join(OUTPUT_DIR, station_id) ):
            os.mkdir( os.path.join(OUTPUT_DIR, station_id) )


        station_metadata.to_csv( os.path.join(OUTPUT_DIR,
                                              station_id,
                                              f'{station_id}_metadata.csv') )


        station_status_info['station'].append(station_id)
        station_status_info['status'].append('train')

        df_test.drop(columns = ['year'])
        df_train.drop(columns = ['year'])


        # fitting scalers with train data only
        _, scaler_data   = normalize_dataframe(dataframe = df_train.drop(columns = 'radiacao_global'),
                                               method = args['normalization_method'])

        _, scaler_target = normalize_dataframe(dataframe = df_train['radiacao_global'],
                                               method = args['normalization_method'])



        xtrain_df, ytrain_df = split_data_targets(dataframe = df_train,
                                                  target = 'rad_prox_hora')

        xtest_df,  ytest_df  = split_data_targets(dataframe = df_test,
                                                  target = 'rad_prox_hora')



        # norm train data
        xtrain, _ = normalize_dataframe(dataframe = xtrain_df,
                                        method = args['normalization_method'],
                                        scaler = scaler_data)
        # norm train target
        ytrain, _ = normalize_dataframe(dataframe = ytrain_df,
                                        method = args['normalization_method'],
                                        scaler = scaler_target)

        # norm test target
        xtest, _  = normalize_dataframe(dataframe = xtest_df,
                                        method = args['normalization_method'],
                                        scaler = scaler_data)

        # norm test target
        ytest, _  = normalize_dataframe(dataframe = ytest_df,
                                        method = args['normalization_method'],
                                        scaler = scaler_target)



        normalized_arrays = [xtrain, xtest, ytrain, ytest]
        files = ['x_train.npy', 'x_test.npy', 'y_train.npy', 'y_test.npy']

        for file, dataset in zip(files, normalized_arrays):
            np.save(file = os.path.join(OUTPUT_DIR, station_id, file), arr = dataset)



        joblib.dump(value = scaler_data,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_scaler_data.dat'))

        joblib.dump(value = scaler_target,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_scaler_target.dat'))



        joblib.dump(value = pca_transformer_pressao,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_pca_transformer_pressao.dat'))

        joblib.dump(value = pca_transformer_umidade,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_pca_transformer_umidade.dat'))

        joblib.dump(value = pca_transformer_temperatura,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_pca_transformer_temperatura.dat'))

        joblib.dump(value = pca_transformer_orvalho,
                    filename = os.path.join(OUTPUT_DIR,
                                            station_id,
                                            f'{station_id}_pca_transformer_orvalho.dat'))


    # out of for loop
    pd.DataFrame().from_dict(station_status_info).to_csv(
        path_or_buf = 'station_status.csv',
        index = False
    )
