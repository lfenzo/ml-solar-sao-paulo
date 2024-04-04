import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_bins(width, xymax):
    binwidth = width
    lim = (int(xymax / binwidth) + 1) * binwidth
    return np.arange(-lim, lim + binwidth, binwidth)


def scatter_hist(x, y, ax, ax_histx, ax_histy, color):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # now determine nice limits by hand:
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    up_bins = set_bins(width=5, xymax=xymax)
    right_bins = set_bins(width=15, xymax=xymax)

    ax_histx.hist(x, bins=up_bins, color=color, edgecolor='k', linewidth=0.5)
    ax_histy.hist(y, bins=right_bins, orientation='horizontal', color=color, edgecolor='k', linewidth=0.5)


if __name__ == "__main__":
    samples_by_station = pd.read_csv('./samples_by_station.csv')
    best_models = pd.read_csv('./best_overall.csv')
    df = pd.merge(best_models, samples_by_station, on='station', how='left')
    df['train'] = df['train'] / 1_000

    interp_mask = df['is_interp']
    interpolated = df[interp_mask].copy()
    original = df[~interp_mask].copy()

    fig = plt.figure(figsize=(7, 7), dpi=200)

    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(df['train'], df['mae'], ax, ax_histx, ax_histy, color='darkgray')
    ax.scatter(
        original['train'], original['mae'],
        label='Original data',
        edgecolors='k',
        linewidths=0.5,
        s=40,
    )
    ax.scatter(
        interpolated['train'], interpolated['mae'],
        marker='D',
        label='IDW-imputed',
        edgecolors='k',
        linewidths=0.5,
        s=40,
    )

    ax.set_yticks(np.arange(150, 315, 15))
    ax.set_ylim(159, 300)
    ax.set_xticks(np.arange(20, 70, 5))
    ax.set_xlim(20, 70)
    ax.grid()
    ax.set_ylabel("MAE (kJ/m²)")
    ax.set_xlabel("Number of training samples ($\\times$1000)")
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig('./perf_scatter.png', bbox_inches='tight')
