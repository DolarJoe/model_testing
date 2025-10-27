from jax import config
import numpy as np
from tvb.simulator import simulator, coupling
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.monitors import Raw
from tvb.simulator.models.oscillator import SupHopf
from config import Config


class TvbModel:
    def __init__(self, config: Config):
        self.config = config
        self._configure_sim()

    def _configure_sim(self):
        self.sim = simulator.Simulator(
            connectivity=self.config.conn,
            model=SupHopf(a=np.r_[self.config.a], omega=np.r_[self.config.w]),
            integrator=EulerDeterministic(dt=self.config.dt),
            initial_conditions=self.config.init_cond,
            monitors=[Raw()],
            simulation_length=self.config.dt * self.config.sim_steps,
            coupling=coupling.Scaling(a=np.r_[1.0]),
        )
        self.sim.configure()

    def run(self):
        return self.sim.run()
