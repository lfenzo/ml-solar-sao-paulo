import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils


def get_predictions(models_dataframe):

    info = []

    for _, row in models_dataframe.iterrows():

        best_regressor_signature = f'{row["model"]}_{row["timestamp"]}'

        for file in os.listdir(f'../stations/{row["station"]}/previsoes'):
            if 'previsoes' in file and best_regressor_signature in file:
                predictions = file

        pred_info = pd.read_csv(f'../stations/{row["station"]}/previsoes/{predictions}')

        info.append({
            'real': pred_info['valor_real'],
            'pred': pred_info['estimado'],
            'residuals': pred_info['valor_real'] - pred_info['estimado'],
            'estacao': row['station'],
            'model': row['model'],
            'rmse': row['rmse'],
            'mae': row['mae'],
        })

    return info

if __name__ == '__main__':

    mpl.use('pgf')
    utils.check_img_dst()

    all_models = pd.read_csv('../stations/best_overall.csv')
    all_models.sort_values(by = 'rmse', inplace = True)

    best_models = all_models.head(3)
    worst_models = all_models.tail(3)

    best_models_info = get_predictions(best_models)
    worst_models_info = get_predictions(worst_models)

    fig, axs = plt.subplots(2, 3, figsize = utils.set_size(width = 6.3, kind = 'golden'),
                            dpi = 150,
                            sharex = True,
                            sharey = True)

    axs[0, 0].set_ylabel('Valor Previsto (MJ/m$^2$)', fontsize = utils.LEGEND_FONTSIZE)
    axs[1, 0].set_ylabel('Valor Previsto (MJ/m$^2$)', fontsize = utils.LEGEND_FONTSIZE)


    # gráfico para os melhores modelos
    for i, info in enumerate(best_models_info):
        axs[0, i].scatter(info['real'] / 1000, info['pred'] / 1000,
                          s = 5,
                          alpha = 0.1)
        axs[0, i].set_title(f'{info["estacao"]} - RMSE: {info["rmse"]:.2f}',
                            fontsize = utils.LEGEND_FONTSIZE,fontfamily = 'monospace')
        axs[0, i].set_xticks(range(0, 6))
        axs[0, i].set_yticks(np.arange(0, 5))
        axs[0, i].grid()


    # gráfico para os piores modelos
    for i, info in enumerate(worst_models_info):
        axs[1, i].scatter(info['real'] / 1000, info['pred'] / 1000,
                          s = 5,
                          alpha = 0.1)
        axs[1, i].set_title(f'{info["estacao"]} - RMSE: {info["rmse"]:.2f}',
                            fontsize = utils.LEGEND_FONTSIZE,fontfamily = 'monospace')
        axs[1, i].set_xlabel('Valor Real (MJ/m$^2$)', fontsize = utils.LEGEND_FONTSIZE)
        axs[1, i].set_yticks(np.arange(0, 5))
        axs[1, i].grid()

    fig.tight_layout()


    fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', '.jpeg'),
                bbox_inches = 'tight')

    fig.savefig('./pgf/' + os.path.basename(__file__).replace('.py', '.pgf'),
                format = 'pgf',
                bbox_inches = 'tight')
