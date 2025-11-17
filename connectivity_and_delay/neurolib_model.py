import numpy as np
from config import Config
from neurolib.models.hopf import HopfModel


class NeurolibModel:
    def __init__(self, config: Config):
        self.config = config
        self._configure_sim()

    def _configure_sim(self):
        self.model = HopfModel(
            Cmat=self.config.conn.weights,
            Dmat=self.config.conn.tract_lengths,
        )
        self.model.params["dt"] = self.config.dt
        self.model.params["xs_init"] = self.config.init_cond[:, 0, :, 0].T
        self.model.params["ys_init"] = self.config.init_cond[:, 1, :, 0].T
        self.model.params["duration"] = self.config.dt * self.config.sim_steps
        self.model.params["a"] = self.config.a
        self.model.params["w"] = self.config.w
        self.model.params["coupling"] = "additive"
        self.model.params["K_gl"] = self.config.coupling_strength
        self.model.params["signalV"] = self.config.speed
        if self.config.noise != 0:
            self.model.params["sigma_ou"] = self.config.noise
            self.model.params["x_ou_mean"] = self.config.noise
            self.model.params["y_ou_mean"] = self.config.noise
            self.model.params["x_ou"] = np.random.uniform(
                -self.config.noise, self.config.noise, (self.config.size,)
            )
            self.model.params["y_ou"] = np.random.uniform(
                -self.config.noise, self.config.noise, (self.config.size,)
            )

    def run(self):
        self.model.run()
        return self.model.x
