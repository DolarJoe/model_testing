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
        self.model.params["xs_init"] = self.config.init_cond[0, 0]
        self.model.params["ys_init"] = self.config.init_cond[0, 1]
        self.model.params["duration"] = self.config.dt * self.config.sim_steps
        self.model.params["a"] = self.config.a
        self.model.params["w"] = self.config.w
        self.model.params["coupling"] = "additive"
        self.model.params["K_gl"] = 1.0
        self.model.params["signalV"] = self.config.speed
        self.model.params.tau_ou = 1.0

    def run(self):
        self.model.run()
        return self.model.x
