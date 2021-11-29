import os
import utils
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


def process_feature_names(name):
    """
    Remove odd characters from feature names
    """
    name = name.lower()
    name = name.replace(' ', '_')
    name = name.replace('ã', 'a')
    name = name.replace('ẽ', 'e')
    name = name.replace('õ', 'o')
    name = name.replace('ç', 'c')
    name = name.replace('í', 'i')

    return name


def rescale_legend_labels(labels: list, factor):
    """
    Correctly assigns the legend size for the scatter plot
    """

    items = []

    for label in labels:

        left = label.split('{')[0]
        right = label.split('}')[1]
        num = ''.join(c for c in label if c.isdigit())

        items.append(left + '{' + str(int(num) * factor) + '}' + right)

    return items


if __name__ == '__main__':

    # comentado para a geração dos PDF  (conflito no backend -- xelatex utilizado por padrão)
    #mpl.use('pgf')
    utils.check_img_dst()
    mpl.rcParams['axes.unicode_minus'] = False

    try:
        melhores_modelos = pd.read_csv('../stations/melhores_modelos.csv')
    except FileNotFoundError:
        melhores_modelos = pd.read_csv('../stations/best_overall.csv')
    except FileNotFoundError:
        print('Arquivos de performance dos modelos não encontrado. ')
        exit()

    estado = gpd.read_file('resources/35MUE250GC_SIR.shp')
    estacoes = pd.read_csv('resources/estacoes_inmet.csv')

    estacoes.columns = map(process_feature_names, estacoes.columns)

    estacoes_perf = pd.merge(melhores_modelos, estacoes, right_on = 'id_estacao', left_on = 'station')
    estacoes_perf['perf'] = (estacoes_perf['rmse'] + estacoes_perf['mae']) / 2

    #
    #
    #
    # Plota o mapa com as performances dos modelos INDIVIDUAIS
    #
    #
    #
    fig, axs = plt.subplots(figsize = utils.set_size(width = 7.5, kind = 'half'), dpi = 200)

    estado.plot(ax = axs,
                color = 'whitesmoke',
                edgecolor = 'k',
                linewidth = 0.2)

    axs.grid(True, linewidth = 0.4)
    axs.set_axisbelow('line')

    # plota as estações que foram excluidas (os Xs)
    axs.scatter(estacoes['longitude'], estacoes['latitude'],
                marker = 'x',
                c = 'k',
                s = 40,
                alpha = 0.8,
                linewidth = 3,
                edgecolor = 'k',
                zorder = 2,
                label = 'Not during P1 training')

    im = axs.scatter(estacoes_perf['longitude'], estacoes_perf['latitude'],
                     s = 90,
                     c = estacoes_perf['rmse'],
                     cmap = 'jet',
                     edgecolor = 'k',
                     zorder = 2,
                     label = 'P1 prediction sites')


    axs.set_ylabel('Latitude', fontsize = utils.LABEL_FONTSIZE)
    axs.set_xlabel('Longitude', fontsize = utils.LABEL_FONTSIZE)

    axs.tick_params(axis = 'both',
                    labelsize = utils.TICK_FONTSIZE)

    axs.legend(loc = 'lower left',
               fontsize = utils.LEGEND_FONTSIZE,
               markerscale = 0.8)

    cbar = fig.colorbar(im, ax = axs,
                        label = '$RMSE$ (kJ/m$^2$)',
                        orientation = 'vertical')

    cbar.ax.tick_params(axis = 'both', labelsize = utils.LABEL_FONTSIZE)

    fig.tight_layout()


    fig.savefig(fname = './jpeg/' + os.path.basename(__file__).replace('.py', '.jpeg'),
                bbox_inches = 'tight')

    fig.savefig(fname = './pgf/' + os.path.basename(__file__).replace('.py', '.pgf'),
                bbox_inches = 'tight', format = 'pgf')

    fig.savefig(fname = './pdf/' + os.path.basename(__file__).replace('.py', '.pdf'),
                bbox_inches = 'tight', format = 'pdf')

    #
    #
    #
    # Plota o mapa com a comparação entre os erros da interpolação numerica e com o 
    # compositor baseado em aprendizado de máquina
    #
    #
    #
    fig, axs = plt.subplots(figsize = utils.set_size(width = 7.5, kind = 'square'),
                            sharex = True,
                            dpi = 300)

    # plota o mapa da localização geográfica das estações de previsão usando os dois
    # métodos testados: 
    #  1- mapa para método numérico 
    #  2- mapa para machine learning 


    idw_compositor = pd.read_csv('../composition/comp_performance_metrics.csv')
    idw_compositor['mean'] = (idw_compositor['rmse'] + idw_compositor['mae']) / 2

    ml_compositor = pd.read_csv('../composition/compositor/ml_compositor_perf_metrics.csv')
    ml_compositor['mean'] = (ml_compositor['rmse'] + ml_compositor['mae']) / 2

    compositor_compare = pd.merge(left = idw_compositor,
                                  right = ml_compositor,
                                  on = 'station',
                                  how = 'inner',
                                  suffixes = ('_idw', '_ml'))

    # dataframe pronto para ser utilzado para plotar
    compositor_compare = pd.merge(compositor_compare, estacoes,
                                  left_on = 'station',
                                  right_on = 'id_estacao',
                                  how = 'inner')

    # plota o estado no plano de ufndo
    estado.plot(ax = axs,
                color = 'whitesmoke',
                edgecolor = 'k',
                linewidth = 0.15,
                zorder = -1)

    # comprimir a altura das barras verticais
    bar_scale_factor = 0.001
    bar_edge_distance = 0.03
    bar_width = 0.11

    for i, row in compositor_compare.iterrows():

        # barras da previsão do compositor (barra da direita)
        axs.add_patch(
            Rectangle(xy = (row['longitude'] + bar_edge_distance, row['latitude'] - 0.1),
                      width = bar_width,
                      facecolor = 'royalblue',
                      linewidth = 0.3,
                      edgecolor = 'k',
                      height = row['mean_ml'] * bar_scale_factor,
                      label = 'Superv. Learning' if i == compositor_compare.shape[0]-1 else '',
                      zorder = i)
        )

        # barras da previsão com IDW
        axs.add_patch(
            Rectangle(xy = (row['longitude'] - bar_edge_distance - bar_width, row['latitude'] - 0.1),
                      width = bar_width,
                      facecolor = 'orangered',
                      linewidth = 0.3,
                      edgecolor = 'k',
                      height = row['mean_idw'] * bar_scale_factor,
                      label = 'IDW' if i == compositor_compare.shape[0]-1 else '',
                      zorder = i)
        )

        axs.plot([row['longitude'] + (bar_edge_distance * 6),
                  row['longitude'] - (bar_edge_distance * 6)],
                 [row['latitude'] - 0.1] * 2,
                 color = 'k',
                 linewidth = 0.5,
                 zorder = i)

    axs.set_ylabel('Latitude')
    axs.set_xlabel('Longitude')

    axs.legend(loc = 'lower left')


    fig.savefig(fname = './jpeg/' + os.path.basename(__file__).replace('.py', '_comp_compare.jpeg'),
                bbox_inches = 'tight')

    fig.savefig(fname = './pgf/' + os.path.basename(__file__).replace('.py', '_comp_compare.pgf'),
                bbox_inches = 'tight', format = 'pgf')

    fig.savefig(fname = './pdf/' + os.path.basename(__file__).replace('.py', '_comp_compare.pdf'),
                bbox_inches = 'tight', format = 'pdf')

