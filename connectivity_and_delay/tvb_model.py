import numpy as np
from tvb.simulator import simulator, coupling
from tvb.simulator.integrators import EulerDeterministic, EulerStochastic
from tvb.simulator.monitors import Raw
from tvb.simulator.models.oscillator import SupHopf
from config import Config
import tvb.simulator.lab as tvbl


class TvbModel:
    def __init__(self, config: Config):
        self.config = config
        self._configure_sim()

    def _configure_sim(self):
        self.sim = simulator.Simulator(
            connectivity=self.config.conn,
            model=SupHopf(a=np.r_[self.config.a], omega=np.r_[self.config.w]),
            integrator=EulerStochastic(
                dt=self.config.dt,
                noise=tvbl.noise.Additive(
                    nsig=np.r_[self.config.noise],
                ),
            ),
            initial_conditions=self.config.init_cond,
            conduction_speed=self.config.speed,
            monitors=[Raw()],
            simulation_length=self.config.dt * self.config.sim_steps,
            coupling=coupling.Scaling(a=np.r_[self.config.coupling_strength]),
        )
        self.sim.configure()

    def run(self):
        return self.sim.run()[0][1].T
