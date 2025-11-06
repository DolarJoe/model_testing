import numpy as np
import tvb.simulator.lab as tvbl
from tvb.simulator.models.oscillator import SupHopf
from conn_no_warning import ConnNoWarnings


class Config:
    def __init__(self, initial_conditions_seed, noise_seed=42):
        self.dt = 0.1
        self.sim_steps = 1
        self.speed = 2.0
        self.a = 0.35
        self.w = 0.2
        self.init_cond_rng = np.random.default_rng(seed=initial_conditions_seed)
        self.history_length = 10
        self.coupling_strength = 1.0
        self.noise = 0.0
        self.noise_seed = noise_seed

    def __config_connectivity(self):
        self.conn = ConnNoWarnings().from_file()
        np.fill_diagonal(self.conn.weights, 0)  # remove self-connections
        self.conn.speed = np.r_[self.speed]
        self.size = self.conn.weights.shape[0]

    def init_config_for_connectivity(self):
        self.__config_connectivity()
        self.conn.tract_lengths = np.zeros_like(self.conn.weights)  # because of neurolib
        self.init_cond = np.r_[
            [[self.init_cond_rng.random((self.size, 1)), self.init_cond_rng.random((self.size, 1))]]
        ]

    # TODO state variable count is hardcoded here, fix
    def init_config_for_delays(self):
        self.__config_connectivity()
        max_len = np.max(self.conn.tract_lengths)
        self.conn.tract_lengths /= max_len
        self.conn.tract_lengths *= self.history_length - 1
        # Maybe this is the right way to do it, I don't know
        # init_hist_shape = int((np.max(self.conn.tract_lengths) / self.dt) / self.dt + 1)
        # self.init_cond = self.np_rng.random((init_hist_shape, 2, self.size, 1))
        self.init_cond = self.init_cond_rng.random(self.get_good_history_shape())
        self.conn.configure()

    # TODO model is hardcoded here, fix
    def get_good_history_shape(self):
        self.__config_connectivity()
        # There is most assuredly a way to calculate this directly from the
        # connectivity and dt, but I have made a mistake here so many times already
        # that I will simply use TVB to do it for me.
        temp_sim = tvbl.simulator.Simulator(
            connectivity=self.conn,
            model=SupHopf(),
            integrator=tvbl.integrators.EulerDeterministic(dt=self.dt),
            conduction_speed=self.speed,
            simulation_length=self.dt * self.sim_steps,
        ).configure()
        return temp_sim.good_history_shape
