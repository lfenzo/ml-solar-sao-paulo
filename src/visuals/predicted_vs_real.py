import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':

    mpl.use('pgf')
    utils.check_img_dst()

    melhores_modelos = pd.read_csv('../clusters/melhores_modelos.csv')

    info = []

    for _, row in melhores_modelos.iterrows():

        best_regressor_signature = f'{row["model"]}_{row["timestamp"]}'

        for file in os.listdir(f'../clusters/{row["cluster"]}/previsoes'):
            if 'previsoes' in file and best_regressor_signature in file:
                predictions = file

        pred_info = pd.read_csv(f'../clusters/{row["cluster"]}/previsoes/{predictions}')

        info.append({
            'real': pred_info['valor_real'].to_list(),
            'pred': pred_info['estimado'].to_list(),
            'estacao': row['cluster'],
        })

    fig, axs = plt.subplots(6, 5, figsize = (18, 18),
                            dpi = 120,
                            sharex = True,
                            sharey = True)

    for i in range(6):
        for j in range(5):

            data = info.pop()

            axs[i, j].scatter(data['real'], data['pred'],
                              alpha = 0.4,
                              s = 10)

            axs[i, j].set_aspect(1)
            axs[i, j].set_title(data['estacao'], size = 14, weight = 'bold')
            axs[i, j].set_xticks(np.arange(0, 6000, 1000))


    fig.tight_layout()

    fig.savefig(os.path.basename(__file__).replace('.py', '.jpeg'), bbox_inches = 'tight')
