# Purpose of script:
# This script is the main entry point for testing the option of
# using an LCMV beamformer for the local fusion problem of dMWF.
#
# Context: dMWF practical aspects development.
#
# Created on: 14/03/2025
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import time
import pickle
import mypalettes
import numpy as np
from tools.algos import *
from tools.base import *
from pyinstrument import Profiler

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

PATH_TO_CFG = ".\\config\\cfg.yml"  # Path to the configuration file

NMC = 100  # number of Monte Carlo runs

def main():
    """Main function (called by default when running script)."""
    cfg = Parameters()
    cfg.load_from_yaml(PATH_TO_CFG)

    np.random.seed(cfg.seed)
    
    # profiler = Profiler()
    # profiler.start()

    msed = []
    t0 = time.time()
    for ii in range(NMC):
        cfg.seed += 1  # change seed for each run
        # Launch the simulation
        msed.append(Run(cfg).launch())
        print(f"Monte Carlo run {ii + 1}/{NMC} completed (total time: {time.time() - t0:.2f} s).", end='\r')
    print("\nAll Monte Carlo runs completed.")

    # Export
    with open(f'./out/data/run_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(msed, f)

    return 0

if __name__ == '__main__':
    sys.exit(main())