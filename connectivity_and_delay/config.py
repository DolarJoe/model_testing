from jax import config
import numpy as np
from tvb.simulator.lab import connectivity


class Config:
    def __init__(self, random_seed):
        self.dt = 0.1
        self.sim_steps = 1
        self.speed = 1.0
        self.a = 0.35
        self.w = 0.2
        self.np_rng = np.random.default_rng(seed=random_seed)
        self.history_length = 10

    def __config_connectivity(self):
        self.conn = connectivity.Connectivity.from_file()
        np.fill_diagonal(self.conn.weights, 0)  # remove self-connections
        self.conn.speed = np.r_[self.speed]
        self.size = self.conn.weights.shape[0]

    def init_config_for_connectivity(self):
        self.__config_connectivity()
        self.conn.tract_lengths = np.zeros_like(self.conn.weights)  # because of neurolib
        self.init_cond = np.r_[
            [[self.np_rng.random((self.size, 1)), self.np_rng.random((self.size, 1))]]
        ]

    # def init_config_for_delays(self):
    #     self.__config_connectivity()
    #     init_hist_shape = (
    #         np.rint((self.conn.tract_lengths / self.speed) / self.dt).astype(np.int32).max() + 1
    #     )
    #     self.init_cond = self.np_rng.random((init_hist_shape, 2, self.size, 1))
    #     max_len = np.max(self.conn.tract_lengths)
    #     self.conn.tract_lengths /= max_len
    #     self.conn.tract_lengths *= self.history_length - 1
