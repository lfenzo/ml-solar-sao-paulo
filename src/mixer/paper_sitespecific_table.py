"""
Gera a tabela da performance dos melhores modelos que será colocada
no paper na seção "site specific"

obtem o arquivo 'best_overall.csv' que é gerado pelo script 'blend_all.py'
e junta com informações do arquivo '../data/estacoes_inmet.csv' tais como
latitude e longitude.
"""

import numpy as np
import pandas as pd

def latex_table_format(df, output_file) -> None:

    with open(output_file, mode = 'w') as file:

        formatted_string = ''

        for _, row in df.iterrows():
            formmatted_string = str(row['station']) + ' & ' + \
                                str(row['Latitude']) + ' & ' + \
                                str(row['Longitude']) + ' & ' + \
                                str(row['rmse']) + ' & ' + \
                                str(row['mae']) + ' & ' + \
                                str(row['mbe']) + ' & ' + \
                                str(row['r2']) + ' \\\\ \n'

            file.write(formmatted_string)


if __name__ == "__main__":

    perf_data = pd.read_csv("./best_overall.csv")
    site_data = pd.read_csv('../data/estacoes_inmet.csv')

    for feature in ['Latitude', 'Longitude']:
        site_data[feature] = site_data[feature].round(3)

    # juntando os dois dataframes
    data = pd.merge(perf_data, site_data,
                    left_on = 'station',
                    right_on = 'Id Estação')

    selected_features = [
        'station',
        'Latitude',
        'Longitude',
        'rmse',
        'mae',
        'mbe',
        'r2',
   ]

    data = data[selected_features]

    latex_table_format(df = data, output_file = 'paper_sitespecific_table.tex')
