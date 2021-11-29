import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

import matplotlib.font_manager as font_manager

if __name__ == '__main__':

    mpl.use('pgf')
    utils.check_img_dst()

    try:
        melhores_modelos = pd.read_csv('../stations/melhores_modelos.csv')
    except FileNotFoundError:
        melhores_modelos = pd.read_csv('../stations/best_overall.csv')
    except FileNotFoundError:
        print('Arquivos de performance dos modelos não encontrado. ')
        exit()

    melhores_modelos.sort_values(by = ['rmse','mae', 'r2'],
                                 ascending = False,
                                 inplace = True)

    fig, axs = plt.subplots(figsize = utils.set_size(width = 7, kind = 'half'), dpi = 220)

    axs.bar(melhores_modelos['station'],
            edgecolor = 'k',
            linewidth = 0.5,
            width = 0.7,
            height = melhores_modelos['rmse'],
            label = 'RMSE')

    axs.bar(melhores_modelos['station'],
            edgecolor = 'k',
            linewidth = 0.5,
            width = 0.7,
            height = melhores_modelos['mae'],
            label = 'MAE')

    axs.legend(fontsize = utils.LEGEND_FONTSIZE,
               prop = font_manager.FontProperties(family = 'monospace',
                                                  size = utils.LEGEND_FONTSIZE))

    axs.set_ylabel('Erro (kJ/m$^2$)',
                   weight = 'bold',
                   fontsize = utils.LABEL_FONTSIZE)

    axs.set_xlabel('Ponto de Previsão (ID da Estação)',
                   weight = 'bold',
                   fontsize = utils.LABEL_FONTSIZE)

    axs.grid(axis = 'y',
             linewidth = 0.5)

    axs.set_yticks(np.arange(50, 600, 50))

    axs.set_ylim(ymin = min(melhores_modelos['mae']) - 40,
                 ymax = max(melhores_modelos['rmse']) + 20)

    axs.set_xticklabels(labels = melhores_modelos['station'],
                        size = utils.LABEL_FONTSIZE,
                        rotation = 90,
                        fontdict = {'fontfamily': 'monospace'})

    axs.tick_params(axis = 'both',
                    labelsize = utils.TICK_FONTSIZE)


    fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', '.jpeg'),
                bbox_inches = 'tight')

    fig.savefig('./pgf/' + os.path.basename(__file__).replace('.py', '.pgf'),
                bbox_inches = 'tight')
