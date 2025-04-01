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
DATA_FILE = f'{pathToData}/run_20250401_093538.pkl'

PALETTE = 'seabed'

def main():
    """Main function (called by default when running script)."""
    with open(DATA_FILE, 'rb') as f:
        msed = pickle.load(f)

    if isinstance(msed[0][list(msed[0].keys())[0]][0], float):
        plot_batch(msed)
    else:
        plot_online(msed)


def plot_online(msed):
    pass


def plot_batch(msed):
    """
    Batch-mode run plot: violins.
    """
    palette = mypalettes.get_palette(PALETTE)
    # Select len(msed[0].keys()) equally spaced colors from the palette
    palette = [
        palette[ii]
        for ii in np.linspace(
            0, len(palette) - 1, len(msed[0].keys())
        ).astype(int)
    ]

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
        col = palette[ii]
        # Violin plot over MC runs, averaged over nodes
        violinparts = axes[idxAx].violinplot(data, positions=posViolin, widths=violinWidth, showmeans=True, showextrema=False)
        for pc in violinparts['bodies']:
            pc.set_facecolor(col)
            pc.set_edgecolor(palette[-1])
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

if __name__ == '__main__':
    sys.exit(main())