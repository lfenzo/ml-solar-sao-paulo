import os
import numpy as np
import pandas as pd
from functools import partial


def save_latex_format(filepath, dataframe):
    """
    Salva os dados que seriam salvos em um arquivo .csv em um formato que o uma tabela
    do latex consegue compreender.
    """
    with open(filepath, 'w') as file:

        for _, row in dataframe.iterrows():

            formmatted_string = ''

            for feature in row.index:
                formmatted_string += '\\texttt{' + str(row[feature]) + '} & '

            formmatted_string += '\\\\\n'

            file.write(formmatted_string)


if __name__ == '__main__':

    idw_perf = pd.read_csv('./comp_performance_metrics.csv')
    ml_perf = pd.read_csv('./ml_compositor_perf_metrics.csv')

    # gera aa tabela comparativa com mas MEDIAS (tabela pequena)
    perf_features = ['rmse', 'mae', 'mbe', 'r2']
    table_info = {}

    for feature in perf_features:
        table_info[f'{feature}_mean'] = []
        table_info[f'{feature}_std'] = []

    for df in [idw_perf, ml_perf]:
        for feature in perf_features:
            table_info[f'{feature}_mean'].append( df[feature].mean() )
            table_info[f'{feature}_std'].append( df[feature].std() )

    table_info = pd.DataFrame().from_dict(table_info)
    table_info = table_info.rename(index = {0:'idw', 1:'ml'})

    table_info.to_csv('mean_table.csv')
    print(table_info)

    print(idw_perf.shape)
    print(ml_perf.shape)

    # tabela com a tabela que compara a performance estação a estação
    comparative_table = pd.merge(right = idw_perf,
                                 left = ml_perf,
                                 on = 'station',
                                 suffixes = ('_ml', '_idw'),
                                 how = 'inner')

    new_column_order = ['station']

    for feature in perf_features:
        new_column_order.append( f'{feature}_idw' )
        new_column_order.append( f'{feature}_ml' )

    # selecionando apenas os atributos de performance
    comparative_table = comparative_table.loc[:, new_column_order]

    # faz os arredondamentos nos atributos numéricos (r2 com 3 casas)
    for feature in comparative_table.columns:

        if 'r2' in feature:
            comparative_table[[feature]] = comparative_table[[feature]].round(3)

        elif feature != 'station':
            comparative_table[[feature]] = comparative_table[[feature]].round(2)

    # preenchendo os valores vazios (pela diferença entre os conjuntos de dados ficaram faltando estações no ML)
    comparative_table = comparative_table.fillna(value = '--')

    comparative_table.to_csv('comparative_table.csv', index = False)
    print(comparative_table)

    save_latex_format(dataframe = comparative_table,
                      filepath = 'comparative_table.tex')
