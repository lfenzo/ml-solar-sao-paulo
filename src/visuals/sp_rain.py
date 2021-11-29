import os
import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # gera o conjunto de dados utilizado para o plot, caso ele não exista
    if not os.path.exists('data_sp_rain.csv'):

        names = [
            'data',
            'hora',
            'precip',
            'press_atm',
            'press_atm_max',
            'press_atm_min',
            'rad_global',
            'temp',
            'orvalho',
            'temp_max',
            'temp_min',
            'orvalho_max',
            'orvalho_min',
            'umid_max',
            'umid_min',
            'umid',
            'vento_dir',
            'vento_raj',
            'vento_vel',
            'lat',
            'lon',
            'alt',
            'id',
        ]

        data = pd.read_feather('../data/dataframe_concatenado.ftr')
        data.columns = names

        data.dropna(inplace = True)

        data['hora'] = data['hora'].str.replace('UTC', '')
        data['data'] = data['data'].str.replace('-', '/')

        data['datetime'] = pd.to_datetime(data['data'] + ' ' + data['hora'],
                                      format = '%Y-%m-%d %H:%M',
                                      utc = True)

        data.drop(columns = ['data', 'hora'], inplace = True)

        data.to_csv('data_sp_rain.csv', index = False)



    # se um deles existe então ambos existem. A não ser que haja um erro ambos sempre 
    # vão existir
    if os.path.exists('./sp_rain_ready1.pkl') and os.path.exists('./sp_rain_ready2.pkl'):

        with open("./sp_rain_ready1.pkl", 'rb') as file:
            chuva4 = pickle.load(file)

        with open("./sp_rain_ready2.pkl", 'rb') as file:
            valores_por_mes = pickle.load(file)

    else:
        data = pd.read_csv('data_sp_rain.csv')

        data['datetime'] = pd.to_datetime(data['datetime'],
                                      format = '%Y-%m-%d %H:%M:%S',
                                      utc = True)

        data['mes'] = data['datetime'].dt.month.astype('int32')
        data['hora'] = data['datetime'].dt.hour.astype('int32')
        data['ano'] = data['datetime'].dt.year.astype('int32')
        data['dia'] = data['datetime'].dt.day.astype('int32')

        data.at[data['precip'] < 0, 'precip'] = np.nan
        data.at[data['rad_global'] < 0, 'rad_global'] = np.nan
        data.at[data['umid'] < 0, 'umid'] = np.nan
        data.at[data['umid_max'] < 0, 'umid_max'] = np.nan
        data.at[data['umid_min'] < 0, 'umid_min'] = np.nan

        data.at[data['temp'] < 0, 'temp'] = np.nan
        data.at[data['temp_max'] < 0, 'temp_max'] = np.nan
        data.at[data['temp_min'] < 0, 'temp_min'] = np.nan

        data.at[data['orvalho'] < 0, 'orvalho'] = np.nan
        data.at[data['orvalho_max'] < 0, 'orvalho_max'] = np.nan
        data.at[data['orvalho_min'] < 0, 'orvalho_min'] = np.nan

        data.dropna(inplace = True)

        data = data[ data['hora'].isin(range(10, 22)) ]

        # =======================================
        # =======================================
        # =======================================

        data = data[ data['hora'].isin(range(10, 22)) ]
        data_gb = data.groupby(by = ['id', 'ano', 'mes', 'hora'])

        chuva1 = data_gb.agg(sum)

        chuva2 = chuva1.groupby(by = ['id', 'ano', 'mes']).agg(sum)
        chuva3 = chuva2.groupby(by = ['id', 'mes']).mean()
        chuva4 = chuva3.groupby(by = ['mes']).mean()

        chuva4 = chuva4[['precip']]
        with open('./sp_rain_ready1.pkl', 'wb') as file:
            pickle.dump(obj = chuva4, file = file)

        # ================================================================

        solar = data.copy()
        solar.at[solar['rad_global'] < -10, 'rad_global'] = np.nan
        solar = solar.dropna()

        grad1 = solar.groupby(by = ['id', 'ano', 'mes', 'dia', 'hora']).sum()
        grad2 = grad1.groupby(by = ['id', 'ano', 'mes', 'dia']).sum()
        grad3 = grad2.groupby(by = ['id', 'ano', 'mes']).mean()
        grad4 = grad3.groupby(by = ['id', 'mes']).mean()

        valores_por_mes = []

        for mes in range(1, 13):
            valores = []
            for estacao in solar['id'].unique():
                try:
                    valores.append(grad4.at[(estacao, mes), 'rad_global'])
                except Exception:
                    continue

            valores_por_mes.append(valores)

        with open('./sp_rain_ready2.pkl', 'wb') as file:
            pickle.dump(obj = valores_por_mes, file = file)

    # ================================================================
    # ================================================================
    # ================================================================

    meses_num = np.arange(1, 13)
    meses = [
       'Jan', 'Fev', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Set', 'Oct', 'Nov', 'Dec',
    ]

    fig, axs = plt.subplots(figsize = utils.set_size(width = 6.5, kind = 'golden'),
                            dpi = 150,
                            sharex = True)

    axs.plot(meses_num, chuva4.values.ravel(),
             label = 'Precipitation',
             linestyle = 'dashed',
             color = 'tab:blue',)

    axs.set_xticks(meses_num)
    axs.set_xticklabels(meses, size = utils.TICK_FONTSIZE)

    axs.tick_params(axis = 'both', labelsize = utils.TICK_FONTSIZE)

    axs.set_ylabel('Mean Precipitation (mm)',
                   fontsize = utils.LABEL_FONTSIZE)

    axs.yaxis.label.set_color('tab:blue')
    axs.tick_params(axis = 'y', colors = 'tab:blue')

    axs.legend(loc = 'lower right', fontsize = utils.LEGEND_FONTSIZE)

    axs.set_xlabel("Month of the Year", fontsize = utils.LABEL_FONTSIZE)
    # ================================================================
    # ================================================================
    # ================================================================

    filtered = []

    for m in valores_por_mes:
        mininum = min(m)
        filtered.append([v for v in m if v > mininum])

    valores_por_mes = filtered


    axs2 = axs.twinx()

    box = axs2.boxplot(valores_por_mes,
                       patch_artist = True,
                       showmeans = False,
                       labels = meses,
                       medianprops = {'color': 'red'},
                       boxprops = {'alpha': 0.3})

    colors = ['tab:orange' for _ in range(12)]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    axs2.set_xticks(meses_num)

    axs2.set_yticklabels([f'{valor:.1f}' for valor in np.arange(5, 27.5, 2.5)],
                         fontsize = utils.LABEL_FONTSIZE)

    axs2.set_ylabel('Global Radiation (MJ/m²)',
                    fontsize = utils.LABEL_FONTSIZE)

    axs2.yaxis.label.set_color('tab:red')
    axs2.tick_params(axis = 'y', colors = 'tab:red')

    fig.tight_layout()

    fig.savefig('./jpeg/' + os.path.basename(__file__).replace('.py', '.jpeg'),
                bbox_inches = 'tight')

    #fig.savefig('./pgf/' + os.path.basename(__file__).replace('.py', '.pgf'), format = 'pgf', bbox_inches = 'tight')

    fig.savefig('./pdf/' + os.path.basename(__file__).replace('.py', '.pdf'),
                format = 'pdf',
                bbox_inches = 'tight')
