"""
Gera faz a comparação entre a performance da generalização via ml
com os resultados do modelo empirico.


Como o modelo é emírico não é necessário "rodar de novo" as previsões
pois ele dará o mesmo resultado, assim basta comparar o erro das previsões
que o MLCOMPOSITOR nas estações que ele tinha dados para testar com as
estações correspondentes que tem previsões feitas pelo modelo empírico
"""

import os
import numpy as np
import pandas as pd


def latex_table_format(df, output_file) -> None:

    with open(output_file, mode = 'w') as file:

        formatted_string = ''

        for _, row in df.iterrows():
            formmatted_string = str(row['station'])     + ' & ' + \
                                str(row['rmse_ml'])     + ' & ' + \
                                str(row['rmse_idw'])    + ' & ' + \
                                str(row['rmse_empi'])   + ' & ' + \
                                str(row['mae_ml'])      + ' & ' + \
                                str(row['mae_idw'])     + ' & ' + \
                                str(row['mae_empi'])    + ' & ' + \
                                str(row['mbe_ml'])      + ' & ' + \
                                str(row['mbe_idw'])     + ' & ' + \
                                str(row['mbe_empi'])    + ' & ' + \
                                str(row['r2_ml'])       + ' & ' + \
                                str(row['r2_idw'])      + ' & ' + \
                                str(row['r2_empi'])     + ' \\\\ \n'

            file.write(formmatted_string)


if __name__ == "__main__":

    # pega a informação de quais estações foram usadas no teste do arquivo de informações de performance
    # do ml_compositor
    ml_compositor_metrics = pd.read_csv("../../composition/compositor/ml_compositor_perf_metrics.csv")

    idw_compositor_metrcs = pd.read_csv("../../composition/comp_performance_metrics.csv")

    empirical_metrics = pd.read_csv('../site_specific/emp_perf_table.csv')


    for feature in ['rmse', 'mae', 'mbe', 'r2']:
        empirical_metrics[f"{feature}_empi"] = empirical_metrics[feature]
        empirical_metrics.drop(columns = feature, inplace = True)


    ml_e_idw = pd.merge(ml_compositor_metrics, idw_compositor_metrcs,
                        on = 'station',
                        how = 'inner',
                        suffixes = ('_ml', '_idw'))

    all_methods = pd.merge(ml_e_idw, empirical_metrics,
                           on = 'station',
                           how = 'inner')


    new_order = ['station']

    for metric in ['rmse', 'mae', 'mbe', 'r2']:
        for method in ['ml', 'idw', 'empi']:
            new_order.append( f"{metric}_{method}" )

    all_methods = all_methods[new_order]
    all_methods.to_csv('all_methods.perfs.csv', index = False)

    latex_table_format(all_methods, output_file = 'all_methods.tex')
