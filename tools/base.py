# Contents of package:
# Classes and functions related to the basic functionality for ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import yaml
import pickle
import numpy as np
from dataclasses import dataclass

@dataclass
class Parameters:
    """A dataclass for simulation parameters."""
    K: int = 5      # number of nodes
    Mk: int = 10    # number of sensors per node (same for all nodes)
    D: int = 1      # number of target signal channels at every node
    Qd: int = 5     # number of desired sources in environment
    Qn: int = 3     # number of noise sources in environment
    Nbatch: int = int(1e4)    # number of samples for batch processing
    Nonline: int = int(1e2)    # number of samples for online processing (frame length)
    nFrames: int = int(1e2)    # number of time frames (only for online processing)
    upEvery: int = 1    # number of time frames between consecutive updates of the fusion matrices (only for online processing)
    beta: float = 0.995    # exponential averaging factor (only for online processing)
    scmEst: str = 'theoretical'    # type of SCM estimation method ('theoretical', 'batch', or 'online')

    seed: int = 42  # random number generator seed
    outputDir: str = ""  # path to output directory

    def __post_init__(self):
        np.random.seed(self.seed)
        self.M = self.Mk * self.K
        assert self.Mk >= self.Qd + self.Qn, "Number of sensors per node must be greater than number of sources."

    def load_from_yaml(self, path: str):
        """Load parameters from a YAML file."""
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        for key, value in data.items():
            setattr(self, key, value)
        self.__post_init__()

    def export_to_pickle(self, name: str):	
        """Export parameters to a Pickle archive."""
        with open(f'{self.outputDir}/{name}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)