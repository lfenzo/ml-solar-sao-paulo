import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

from matplotlib.gridspec import GridSpec


if __name__ == '__main__':

    mpl.use('pgf')
    mpl.rcParams['axes.unicode_minus'] = False
    utils.check_img_dst()

    #melhores_modelos = pd.read_csv('../stations/melhores_modelos.csv')
    best_overall = pd.read_csv('../mixer/best_overall.csv')
    samples_por_estacao = pd.read_csv('../stations/samples_by_station.csv')

    data = pd.merge(best_overall, samples_por_estacao,
                    left_on = 'station',
                    right_on = 'station',
                    how = 'inner')

    data['mean'] = (data['rmse'] + data['mae']) / 2

    mean_train_size = data[['train']].mean().values

    #
    #
    # gráfico 1: scatter plot da performance de cada uma das estações pelo tamanho
    #            do conuunto de treinamento utilizado
    #

    fig, axs = plt.subplots(figsize = utils.set_size(width = 3, kind = 'square'),
                            dpi = 400)

    # informação que será utilizada em cada iteração para plotar os gráficos
    plot_info = [
        {
            'variable': 'r2',
            'mean_color': 'red',
            'color': 'orange',
        },
        {
            'variable': 'mae',
            'mean_color': 'red',
            'color': 'dodgerblue',
        },
        {
            'variable': 'mbe',
            'mean_color': 'red',
            'color': 'purple',
        },
        {
            'variable': 'rmse',
            'mean_color': 'red',
            'color': 'green',
        },
    ]

    for info in plot_info:

        axs.clear()

        axs.scatter(data['train'], data[f'{info["variable"]}'],
                    c = f'{info["color"]}',
                    linewidth = .5,
                    edgecolor = 'k',
                    alpha = 0.5,
                    label = 'Melhor modelo',
                    s = 25)

        axs.scatter(mean_train_size, data[[f'{info["variable"]}']].mean().values,
                    c = info['mean_color'],
                    edgecolor = 'k',
                    linewidth = .5,
                    alpha = 0.5,
                    s = 100,
                    label = 'Valor médio')

        axs.legend(markerscale = 0.5,
                   fontsize = utils.LEGEND_FONTSIZE - 1)

        axs.set_ylabel(f'{info["variable"].upper() + " (kJ/m$^2$)" if info["variable"] != "r2" else "R$^2$ Score"}',
                          weight = 'bold',
                          fontsize = utils.LABEL_FONTSIZE)

        axs.set_xlabel('Tamanho do Conjunto de Treinamento por Estação',
                          fontsize = utils.LABEL_FONTSIZE)

        axs.tick_params(axis = 'both', labelsize = utils.TICK_FONTSIZE)

        axs.grid(True, linewidth = 0.5)
        axs.set_axisbelow(True)

        fig.tight_layout()

        fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', f'_{info["variable"]}.jpeg'),
                    bbox_inches = 'tight')

        fig.savefig('./pgf/' + os.path.basename(__file__).replace('.py', f'_{info["variable"]}.pgf'),
                    format = 'pgf',
                    bbox_inches = 'tight')

    #
    #
    # gráfico 2: Scatter plot da performance média dos modelos com anotações de qual esta~ao é qual 
    #
    #            
    mean_perf = (data[['mae']].mean().values + data[['rmse']].mean().values) / 2
    mean_nsamples = data['train'].mean() / 1000

    fig = plt.figure(figsize = utils.set_size(width = 7, kind = 'golden'), dpi = 170)

    gs = GridSpec(9, 10, figure = fig, hspace = 10)

    axs_big = fig.add_subplot(gs[:, 0:6])
    axs_zoom = fig.add_subplot(gs[3:8, 7:10])

    for axs in [axs_big, axs_zoom]:

        axs.grid(True, linewidth = 0.3)
        axs.set_axisbelow(True)

        # faz o scatter plot dos pontos relativos a cada um dos algoritmos utilizados
        for interp_status in data['is_interp'].unique():

            info = data[ data['is_interp'] == interp_status ].copy()

            # plota os pontos corresopndentes a cada um dos melhores modelos
            axs.scatter(info['train'] / 1000, info['mean'],
                        c = 'gold' if interp_status == True else 'purple',
                        s = 20,
                        alpha = 0.8,
                        edgecolor = 'k',
                        linewidth = 0.5,
                        label = 'Dados Interpolados' if interp_status == True else 'Dados Originais')

        # marca a média dos atributos
        axs.scatter(mean_train_size / 1000, mean_perf,
                    c = 'red',
                    edgecolor = 'k',
                    alpha = 0.8,
                    s = 100,
                    label = 'Valor Médio')

        axs.set_xticks(np.arange(20, 75, 5))

        if axs is axs_big:

            axs.set_yticks(np.arange(200, 450, 25))

            axs.set_ylim(bottom = min(data['mean']) - 10,
                         top = max(data['mean']) + 10)

            axs.set_xlim(left = min(data['train'] / 1000) - 2.5,
                         right = max(data['train'] / 1000) + 2.5)

            axs.set_ylabel('$(RMSE + MAE)^{-2}$ kJ/m$^2$')
            axs.set_xlabel('N$^o$ de Amostras no Treinamento em Cada Estação ($\\times 1000$)')

        elif axs is axs_zoom:

            axs.legend(markerscale = 0.8,
                       loc = 'lower center',
                       bbox_to_anchor = (0.5, 1.1),
                       fontsize = utils.LEGEND_FONTSIZE)

            axs.set_yticks(np.arange(200, 450, 5))

            axs.set_ylim(bottom = mean_perf - 20, top = mean_perf + 20)
            axs.set_xlim(left = mean_nsamples - 10, right = mean_nsamples + 10)

    fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', '_mean.jpeg'), bbox_inches = 'tight')
    fig.savefig('./pgf/' + os.path.basename(__file__).replace('.py', '_mean.pgf'), format = 'pgf', bbox_inches = 'tight')
    fig.savefig('./pdf/' + os.path.basename(__file__).replace('.py', '_mean.pdf'), bbox_inches = 'tight')
