import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from empirical_vs_ml.empirical_model import CPRGModel


def mean_bias_error(preds, real) -> float:

    preds = np.asarray(preds)
    real = np.asarray(real)

    # preds é unidimensional então não e possível subscrever o .size
    return np.sum(preds - real) / preds.size


if __name__ == "__main__":

    perf_info = {
        'station': [],
        'lat': [],
        'lon': [],
        'rmse': [],
        'mae': [],
        'mbe': [],
        'r2': [],
    }

    if not os.path.exists('./empirical_preds_python/'):
        os.mkdir('./empirical_preds_python/')


    model = CPRGModel()

    for file in os.listdir("../input_data"):

        data = pd.read_csv(f"../input_data/{file}")
        station = file.split(".")[0]

        # pula os datamframes vazios
        if data.empty:
            continue

        this_site_preds = {
            'doy': [],
            'hour': [],
            'year': [],
            'emp_pred': [],
            'real': [],
        }

        lat = data.loc[0, 'lat']
        lon = data.loc[0, 'lon']

        for _, row in data.iterrows():

            localtime = row['hour'] - 3 # correção para o fuso-horário 
            standard_meridian_time = -3
            doy = row['doy']
            total_daily = row['daily_gsr']

            # localtime - 3 pois as previsões pareciam estar deslocadas em 3 horas mesmo fazendo
            # a correção para o fuso horário local. Com esse valor os erros são mehores
            emp_pred = model.predict(lat, lon, localtime - 3, standard_meridian_time, doy, total_daily)

            this_site_preds['doy'].append( row['doy'] )
            this_site_preds['hour'].append( row['hour'] )
            this_site_preds['year'].append( row['year'] )
            this_site_preds['emp_pred'].append( emp_pred )
            this_site_preds['real'].append( row['radiacao_global'] )


        this_site_preds = pd.DataFrame().from_dict( this_site_preds )
        this_site_preds.to_csv(f'./empirical_preds_python/{station}.csv', index = False)

        perf_info['station'].append( station )
        perf_info['lat'].append( lat )
        perf_info['lon'].append( lon )
        perf_info['rmse'].append( mean_squared_error(this_site_preds['emp_pred'], this_site_preds['real'], squared = False) )
        perf_info['mae'].append( mean_absolute_error(this_site_preds['emp_pred'], this_site_preds['real']) )
        perf_info['mbe'].append( mean_bias_error(this_site_preds['emp_pred'], this_site_preds['real']) )
        perf_info['r2'].append( r2_score(this_site_preds['emp_pred'], this_site_preds['real']) )


    perf_info = pd.DataFrame().from_dict(perf_info)
    perf_info.to_csv('./emp_perf_table.csv', index = False)

   # latex_table_format(perf_info, output_file = 'data.tex')
