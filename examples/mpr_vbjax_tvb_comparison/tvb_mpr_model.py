import numpy as np
import tvb.simulator.lab as tvbl
from mpr_config import MPRConfig
from tvb.simulator import coupling, simulator
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.monitors import Raw


class TvbMPRModel:
    def __init__(self, config: MPRConfig):
        self.config = config
        self._configure_sim()

    # @patrik takto sa scaluje noise strength v TVB, napr. VBJax nic take nerobi
    # Najdes to v tvbl.noise.Additive
    # g_x = numpy.sqrt(2.0 * self.nsig)

    def _configure_sim(self):
        self.sim = simulator.Simulator(
            connectivity=self.config.conn,
            model=MontbrioPazoRoxin(
                tau=np.r_[self.config.tau],
                I=np.r_[self.config.I],
                Delta=np.r_[self.config.Delta],
                J=np.r_[self.config.J],
                eta=np.r_[self.config.eta],
                cr=np.r_[self.config.cr],
                cv=np.r_[self.config.cv],
            ),
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
        return self.sim.run()[0][1]
