import numpy as np
import tvb.simulator.lab as tvbl
from suphopf_config import SupHopfConfig
from tvb.simulator import coupling, simulator
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.models.oscillator import SupHopf
from tvb.simulator.monitors import Raw


class TvbModel:
    def __init__(self, config: SupHopfConfig):
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
                    noise_seed=self.config.noise_seed,
                ),
            ),
            initial_conditions=self.config.init_cond,
            conduction_speed=self.config.speed,
            monitors=[Raw()],
            simulation_length=self.config.dt,
            coupling=coupling.Scaling(a=np.r_[self.config.coupling_strength]),
        )
        self.sim.configure()

    def run(self):
        return self.sim.run()[0][1].T
