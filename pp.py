# Purpose of script:
# This script post-processes the data created by `main.py`.
#
# Context:
# Test of LMCV/LCMP/simpler dMWF fusion.
#
# Created on: 01/04/2025.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import pickle
import mypalettes
import numpy as np
from pathlib import Path

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

pathToData = f'{Path(__file__).parent}/out/data/'
DATA_FILE = f'{pathToData}/run_20250401_093538_batch.pkl'
DATA_FILE = f'{pathToData}/run_20250401_094835_online.pkl'
DATA_FILE = f'{pathToData}/run_20250401_144200_wDANSE_seq.pkl'

PALETTE = mypalettes.get_palette('seabed')

def main():
    """Main function (called by default when running script)."""
    with open(DATA_FILE, 'rb') as f:
        msed = pickle.load(f)

    if isinstance(msed[0][list(msed[0].keys())[0]][0], float):
        fig = plot_batch(msed)
    else:
        fig = plot_online(msed)

    # Save figure
    exportFileName = f'{pathToData}/../{DATA_FILE.split("/")[-1].split(".")[0]}'
    fig.savefig(f'{exportFileName}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{exportFileName}.pdf', dpi=300)
    print(f"Figure saved to {pathToData}/../{DATA_FILE.split('/')[-1].split('.')[0]}.png")


def discretized_palette(palette, n_colors):
    """
    Discretize a given color palette to `n_colors` colors.
    """
    return [
        palette[ii]
        for ii in np.linspace(
            0, len(palette) - 1, n_colors
        ).astype(int)
    ]


def plot_online(msed):
    """Online-mode run plot: line plot."""
    # Select len(msed[0].keys()) equally spaced colors from the palette
    paletteCurr = discretized_palette(PALETTE, len(msed[0].keys()))
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*']

    order = {
        'Unprocessed': 'Unprocessed',
        'Local': 'Local MWF',
        'dMWF': 'dMWF as in [1]',
        # 'LCMV',
        'Simple': 'dMWF with proposed fusion',
        'DANSE': 'DANSE (sequential)',
        'Centralized': 'Centralized'
    }

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(7, 3.5)
    for ii, BFtype in enumerate(order.keys()):
        # Data
        data = np.mean(
            np.array([msed[jj][BFtype] for jj in range(len(msed))]),
            axis=-1
        )  # [nMCruns x nSamples]
        # Turn to dB
        data = 10 * np.log10(data)
        # Mean and standard deviation over MC runs
        m = np.mean(data, axis=0)
        s = np.std(data, axis=0)
        # Plot
        col = paletteCurr[ii]
        if BFtype == 'Centralized':
            col = 'k'
        axes.plot(
            m,
            label=order[BFtype],
            color=col,
            marker=markers[ii % len(markers)],
            markersize=5,
            markeredgecolor=col,
            markerfacecolor='none',
            markevery=0.1,
        )
        # # Add shaded area for standard deviation
        # axes.fill_between(
        #     np.arange(len(m)), m - s, m + s,
        #     alpha=0.1, color=col, edgecolor='none'
        # )
    axes.set_xlabel('Frame index')
    axes.set_ylabel('MSE_d [dB]')
    axes.set_xlim([0, 100])
    # axes.legend()
    # axes.set_title(f'MSE_d over {len(msed)} MC runs (averages over {msed[0]["Unprocessed"].shape[-1]} nodes)')
    fig.tight_layout()
    # Turn off outer edges
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.show()
    
    return fig


def plot_batch(msed):
    """Batch-mode run plot: violins."""
    # Select len(msed[0].keys()) equally spaced colors from the palette
    paletteCurr = discretized_palette(PALETTE, len(msed[0].keys()))

    # Plot results
    fig, axes = plt.subplots(1, 2, gridspec_kw={
        'width_ratios': [len(msed[0].keys()) - 1, 1]
    })
    fig.set_size_inches(8.5, 3.5)
    violinWidth = 0.8
    xScatter = np.random.normal(0, 0.1, len(msed))
    for ii, BFtype in enumerate(msed[0].keys()):
        # Data
        data = np.mean(np.array([msed[jj][BFtype] for jj in range(len(msed))]), axis=1)
        if ii == len(msed[0].keys()) - 1:
            idxAx = 1
            posViolin = [0]
        else:
            idxAx = 0
            posViolin = [ii]
        col = paletteCurr[ii]
        # Violin plot over MC runs, averaged over nodes
        violinparts = axes[idxAx].violinplot(data, positions=posViolin, widths=violinWidth, showmeans=True, showextrema=False)
        for pc in violinparts['bodies']:
            pc.set_facecolor(col)
            pc.set_edgecolor(paletteCurr[-1])
            pc.set_alpha(0.5)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if partname not in violinparts.keys():
                continue
            violinparts[partname].set_edgecolor('black')  # Change edges to black
            violinparts[partname].set_linewidth(2)
        # Plot scatter of individual runs, with random jitter
        xScatterCurr = xScatter + posViolin[0]
        axes[idxAx].scatter(xScatterCurr, data, marker='.', color=col)
        # Add markers on y-axis to show mean values
        axes[idxAx].text(posViolin[0], np.amax(data), f"Mean:\n{np.mean(data):.2e}", ha='center', va='bottom')
        # Adjust y-axis upper limit if text is too close to the top
        if np.amax(data) > 0.9 * axes[idxAx].get_ylim()[1]:
            axes[idxAx].set_ylim(top=1.2 * np.amax(data))
    axes[0].set_xticks(np.arange(len(msed[0].keys()) - 1))
    axes[0].set_xticklabels(list(msed[0].keys())[:-1])
    axes[0].set_xlim(-violinWidth/2, len(msed[0].keys()) - 2 + violinWidth/2)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels([list(msed[0].keys())[-1]])
    # Set axes[1] y-axis ticks and labels on the right
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position('right')
    for ax in axes:
        ax.grid(axis='y')
    fig.suptitle(f'Distribution of MSE_d over {len(msed)} MC runs (averages over {len(msed[0]["Unprocessed"])} nodes)')
    fig.tight_layout()
    # axes[0].set_ylim(bottom=0, top=0.0015)
    plt.show()

    return fig

if __name__ == '__main__':
    sys.exit(main())