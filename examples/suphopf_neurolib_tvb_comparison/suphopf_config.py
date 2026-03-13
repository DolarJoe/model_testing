import numpy as np
from conn_no_warning import ConnNoWarnings


class SupHopfConfig:
    def __init__(self, initial_conditions_seed, noise_seed=42):
        self.dt = 0.1
        self.speed = 2.0
        self.a = 0.35
        self.w = 0.2
        self.init_cond_rng = np.random.default_rng(seed=initial_conditions_seed)
        self.history_length = 10
        self.coupling_strength = 1.0
        self.noise = 0.0
        self.noise_seed = noise_seed
        self.number_of_state_variables = 2
        self.conn = None

    def _config_connectivity(self):
        if self.conn is None:
            self.conn = ConnNoWarnings().from_file()
        np.fill_diagonal(self.conn.weights, 0)  # remove self-connections
        self.conn.speed = np.r_[self.speed]
        self.size = self.conn.weights.shape[0]

    def init_config_for_dfun(self):
        self.conn = ConnNoWarnings()
        self.size = 1000
        self.conn.weights = np.zeros((self.size, self.size))
        self.conn.tract_lengths = np.zeros((self.size, self.size))
        self.conn.centres_spherical(number_of_regions=self.size)
        self.conn.compute_region_labels()
        self.conn.try_compute_hemispheres()
        self.conn.configure()
        # Replace init conditions for a sampling reflecting your model state variable range
        self.init_cond = np.r_[[[np.ones((self.size, 1)), np.ones((self.size, 1))]]]

    def init_config_for_connectivity(self):
        self._config_connectivity()
        self.conn.tract_lengths = np.zeros_like(self.conn.weights)  # because of neurolib
        self.init_cond = self.init_cond_rng.random((1, self.number_of_state_variables, self.size, 1))

    def init_config_for_delays(self):
        self._config_connectivity()
        max_len = np.max(self.conn.tract_lengths)
        self.conn.tract_lengths /= max_len
        self.conn.tract_lengths *= self.history_length - 1
        self.conn.speed = np.r_[self.speed]
        self.init_cond = self.init_cond_rng.random(self._get_good_history_shape())
        self.conn.configure()

    def init_cond_for_noise(self):
        self.conn = ConnNoWarnings()
        self.conn.weights = np.zeros((1000, 1000))
        self.conn.centres_spherical(number_of_regions=1000)
        self.conn.compute_tract_lengths()
        self.conn.compute_region_labels()
        self.conn.try_compute_hemispheres()
        self.conn.configure()
        self._config_connectivity()
        self.init_cond = self.init_cond_rng.random((1, self.number_of_state_variables, self.size, 1))

    def _get_good_history_shape(self):
        self._config_connectivity()
        delays = self.conn.tract_lengths / self.speed
        idelays = np.rint(delays / self.dt).astype(np.int32)
        init_hist_shape = np.max(idelays) + 1
        init_cond_shape = (init_hist_shape, self.number_of_state_variables, self.size, 1)
        return init_cond_shape
