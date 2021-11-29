"""
Gera uma tabela comparando a performance dos modelos empíricos com
a performance dos modelos de machine learning.

Deve ter o seguinte formato:

estação     metrica1_empirico   métrica2_ml .....

"""

import os
import numpy as np
import pandas as pd

def latex_table_format(df, output_file) -> None:

    with open(output_file, mode = 'w') as file:

        formatted_string = ''

        for _, row in df.iterrows():
            interp_status = "\\cmark" if row['is_interp'] == True else "\\xmark"
            formmatted_string = str(row['station'])  + ' & ' + \
                                str(row['lat'])      + ' & ' + \
                                str(row['lon'])      + ' & ' + \
                                str(row['rmse_ml'])  + ' & ' + \
                                str(row['rmse_emp']) + ' & ' + \
                                str(row['mae_ml'])   + ' & ' + \
                                str(row['mae_emp'])  + ' & ' + \
                                str(row['mbe_ml'])   + ' & ' + \
                                str(row['mbe_emp'])  + ' & ' + \
                                str(row['r2_ml'])    + ' & ' + \
                                str(row['r2_emp'])   + ' & ' + \
                                interp_status        + ' \\\\ \n'

            file.write(formmatted_string)



if __name__ == "__main__":

    # essas informações ja incluem o 'is_interp'
    empirical_perf = pd.read_csv('./emp_perf_table.csv')
    mach_lear_perf = pd.read_csv('../../mixer/best_overall.csv')
    #station_metadata = pd.read_csv('../../data/station_status.csv')

    table = pd.merge(empirical_perf, mach_lear_perf,
                     on = 'station',
                     how = 'outer',
                     suffixes = ('_emp', '_ml'))

    table.drop(columns = ['h-params', 'timestamp', 'model'], inplace = True)

    # adicionando as latitudes e longitues
#    table = pd.merge(table, station_metadata,
#                     on = 'station',
#                     how = 'inner')

    table.dropna(inplace = True)

    table = table[[
        'station',
        'lat',
        'lon',
        'rmse_ml',
        'rmse_emp',
        'mae_ml',
        'mae_emp',
        'mbe_ml',
        'mbe_emp',
        'r2_ml',
        'r2_emp',
        'is_interp',
    ]]

    table.to_csv('./comp_perf_table.csv', index = False)

    latex_table_format(table, output_file = 'data.tex')


