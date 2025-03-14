# Purpose of script:
# This script is the main entry point for testing the option of
# using an LCMV beamformer for the local fusion problem of dMWF.
#
# Context: dMWF practical aspects development.
#
# Created on: 14/03/2025
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys, time
import numpy as np
import mypalettes
from tools.algos import *
from tools.base import *

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

PATH_TO_CFG = ".\\config\\cfg.yml"  # Path to the configuration file

NMC = 100  # number of Monte Carlo runs

PALETTE = 'seabed'

def main():
    """Main function (called by default when running script)."""
    cfg = Parameters()
    cfg.load_from_yaml(PATH_TO_CFG)

    np.random.seed(cfg.seed)
    
    msed = []
    for ii in range(NMC):
        t0 = time.time()
        cfg.seed += 1  # change seed for each run
        # Launch the simulation
        msed.append(Run(cfg).launch())
        print(f"Monte Carlo run {ii + 1}/{NMC} completed in {time.time() - t0:.2f} s.")
    print("All Monte Carlo runs completed.")

    post_processing(msed, cfg)
    pass

def post_processing(msed: list[dict], cfg: Parameters):
    """Post-processing of the results."""

    palette = mypalettes.get_palette(PALETTE)
    # Select len(msed[0].keys()) equally spaced colors from the palette
    palette = [
        palette[ii]
        for ii in np.linspace(
            0, len(palette) - 1, len(msed[0].keys())
        ).astype(int)
    ]

    # Plot results
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 3.5)
    violinWidth = 0.8
    for ii, BFtype in enumerate(msed[0].keys()):
        col = palette[ii]
        # Violin plot over MC runs, averaged over nodes
        data = np.mean(np.array([msed[jj][BFtype] for jj in range(NMC)]), axis=1)
        violinparts = axes.violinplot(data, positions=[ii], widths=violinWidth, showmeans=True, showextrema=False)
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
        x = np.random.normal(ii, 0.1, len(data))
        axes.scatter(x, data, marker='.', color=col)
    axes.set_xticks(np.arange(len(msed[0].keys())))
    axes.set_xticklabels(list(msed[0].keys()))
    axes.set_xlim(-violinWidth/2, len(msed[0].keys()) - 1 + violinWidth/2)
    # Add markers on y-axis to show mean values
    for ii, BFtype in enumerate(msed[0].keys()):
        data = np.mean(np.array([msed[jj][BFtype] for jj in range(NMC)]), axis=1)
        axes.text(ii, np.amax(data), f"Mean:\n{np.mean(data):.2e}", ha='center', va='bottom')
    # Adjust y-axis upper limit if text is too close to the top
    if np.amax(data) > 0.9 * axes.get_ylim()[1]:
        axes.set_ylim(top=1.2 * np.amax(data))
    axes.set_title(f'Distribution of MSE_d over {NMC} MC runs (averages over {cfg.K} nodes)')
    axes.grid()
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    sys.exit(main())