"""
Script que gera um gráfico comparando as estações que mais melhraram
os erros graças ao ensemble vs as 4 estações que tiveram os menores
ganhos quando foi utilizado o ensemble
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #
    # obtendo as 4 estações com melhores e piores melhorias 
    # considerando o uso de aprend. supervisionada com compositor
    #

    # tabela gerada pelo script ../composition/ml_vs_idw/compadador.py
    data = pd.read_csv("./comparative_table.csv")

    for feature in ['rmse', 'mae', 'r2', 'mbe']:
        data[f'diff_{feature}'] = data[f'{feature}_idw'] - data[f'{feature}_ml']

    data = data.sort_values(by = ['diff_rmse', 'diff_mae'], ascending = False)

    N_SELETED_STATIONS = 18 # con 36 estações a divisão fica correta sem sobrar estações

    # selecionando as 4 primeiras e 4 ultimas estações com as diferenças calculadas.
    melhores = data.loc[data.head(N_SELETED_STATIONS).index, :]
    melhores = melhores.iloc[::-1, :].reset_index() # invertendo a ordem das linhas do DF

    piores = data.loc[data.tail(N_SELETED_STATIONS).index, :]
    piores = piores.iloc[::-1, :].reset_index()

    fig, axs = plt.subplots(1, 2, figsize = (10, 11), dpi = 400, sharex = True)

    sep = 105

    for i, data in enumerate([melhores, piores]):

        axs[i].grid(axis = 'x')
        axs[i].set_axisbelow(True)

        axs[i].scatter(data.loc[:, 'rmse_ml'], np.arange(N_SELETED_STATIONS),
                       label = "ML",
                       marker = matplotlib.markers.CARETRIGHT,
                       s = 200,
                       linewidth = 0.1,
                       color = 'royalblue')

        for j in range(0, N_SELETED_STATIONS):
            axs[i].text(s = f"{round(data.loc[j, 'rmse_ml'], 1)}",
                        x = data.loc[j, 'rmse_ml'] - sep,
                        y = j,
                        fontsize = 8,
                        va = 'center',
                        ha = 'center')

        #
        #
        #

        for j in range(0, N_SELETED_STATIONS):
            axs[i].text(s = f"{round(data.loc[j, 'rmse_idw'], 1)}",
                        x = data.loc[j, 'rmse_idw'] + sep,
                        y = j,
                        fontsize = 8,
                        va = 'center',
                        ha = 'center')


        axs[i].scatter(data.loc[:, 'rmse_idw'], np.arange(N_SELETED_STATIONS),
                       label = "IDW",
                       marker = matplotlib.markers.CARETLEFT,
                       s = 200,
                       color = 'orangered',
                       edgecolor = 'k')
        #
        #
        #

        # diferença
        axs[i].barh(np.arange(N_SELETED_STATIONS), data.loc[:, 'diff_rmse'],
                    left = data.loc[:, 'rmse_ml'],
                    label = "Diff.",
                    color = 'lightgreen',
                    edgecolor = 'k',
                    height = 0.3)

        axs[i].set_yticks(np.arange(N_SELETED_STATIONS))
        axs[i].set_yticklabels( data.loc[:, 'station'].tolist() )

        axs[i].set_xlabel("RMSE (kJ/m²)")

        axs[i].set_xlim(0, 1000)

        if data is piores:
            axs[i].legend(markerscale = 0.5)




fig.savefig("./jpeg/melhores_melhorias.jpeg", bbox_inches = 'tight')

fig.savefig("./pdf/melhores_melhorias.pdf", bbox_inches = 'tight')


