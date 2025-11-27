import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from config import Config


class MPRConfig(Config):
    def __init__(self, initial_conditions_seed=42, noise_seed=42):
        super().__init__(initial_conditions_seed=initial_conditions_seed, noise_seed=noise_seed)
        self.tau = 1.0
        self.I = 0.0
        self.Delta = 1.0
        self.J = 15.0
        self.eta = -5.0
        self.cr = 1.0
        self.cv = 0.0

    def init_config_for_connectivity(self):
        return super().init_config_for_connectivity()
