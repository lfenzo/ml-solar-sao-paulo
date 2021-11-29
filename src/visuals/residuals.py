import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

def get_station_preds(station_id):
    """
    Dado un ID de estação, retorna as previsões e os error residuasis
    da estação especificada com o melhor regressor encontrado.
    """

    try:
        melhores_modelos = pd.read_csv('../mixer/best_overall.csv')
    except FileNotFoundError:
        melhores_modelos = pd.read_csv('../stations/best_overall.csv')
    except FileNotFoundError:
        print('Arquivos de performance dos modelos não encontrado.')

    for _, row in melhores_modelos.iterrows():

        if row['station'] == station_id:

            best_regressor_signature = f'{row["model"]}_{row["timestamp"]}'

            for file in os.listdir(f'../stations/{row["station"]}/previsoes'):
                if 'previsoes' in file and best_regressor_signature in file:
                    predictions = file

            pred_info = pd.read_csv(f'../stations/{row["station"]}/previsoes/{predictions}')

            info = {
                'station': row['station'],
                'real': pred_info['valor_real'],
                'pred': pred_info['estimado'],
                'residuals': pred_info['valor_real'] - pred_info['estimado'],
            }

            return info

    else:
        raise KeyError(f"Estação {args['station']} não encontrada.")



if __name__ == '__main__':


    ap = argparse.ArgumentParser()

    ap.add_argument('-s', '--station',
                    metavar = 'STRING',
                    required = False,
                    default = 'A711',
                    help = "Código da estação a ser utilizada para plotar os valores residuais.")

    args = vars(ap.parse_args())


    #mpl.use('pgf')
    #mpl.rcParams['axes.unicode_minus'] = False
    utils.check_img_dst()

    info = get_station_preds(args['station'])

    # ======================================================================
    # ======================================================================

    fig, axs = plt.subplots(2, 1, figsize = utils.set_size(width = 6, kind = 'simple'),
                            dpi = 120,
                            sharex = True)


    axs[0].hist2d(np.arange(0, len(info['pred']), 1), info['residuals'],
                  bins = (150, 60),
                  cmap = 'jet')

    axs[1].scatter(np.arange(0, len(info['pred']), 1), info['residuals'],
                   alpha = 0.3,
                   s = 2)

    axs[0].tick_params(axis = 'both',
                       labelsize = utils.TICK_FONTSIZE)

    # devido a valores faltantes no conjunto de treinamento (não tem exatamente 2 anos inteiros)
    # predisa ser ajustado manualmente
    axs[0].set_xticks([0, 2015, 4040, 6090, 8400])
    axs[0].set_xticklabels(['Jan', 'Jun', 'Dec', 'Jun', 'Dec'])


    axs[1].tick_params(axis = 'both',
                       labelsize = utils.TICK_FONTSIZE)

    axs[0].set_ylabel('Residual Error (kJ/m$^2$)',
                      fontsize = utils.LABEL_FONTSIZE)



    axs[1].set_ylabel('Residual Error (kJ/m$^2$)',
                      fontsize = utils.LABEL_FONTSIZE)

    axs[1].set_xlabel('Test Set (Month of the Year)',
                      fontsize = utils.LABEL_FONTSIZE)

    axs[0].set_ylim(-1500, 1500)

    fig.tight_layout()

    fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', '_example.jpeg'),
                 bbox_inches = 'tight')

    fig.savefig('./pdf/' + os.path.basename(__file__).replace('.py', '_example.pdf'),
                format = 'pdf',
                bbox_inches = 'tight')
