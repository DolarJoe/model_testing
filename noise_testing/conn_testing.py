# change to the root directory of the project

from tvb.simulator.models.oscillator import SupHopf
from tvb.simulator.lab import simulator, connectivity, coupling
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.monitors import Raw
import tvb.simulator.lab as tvbl

import numpy as np
from neurolib.models.hopf import HopfModel


class ModelWrapper:

    def __init__(self):
        self.dt = 0.1
        self.noise = 0.0
        self.size = 5
        self.conn = np.zeros((self.size, self.size))
        interim_conn = connectivity.Connectivity.from_file()
        np.fill_diagonal(interim_conn.weights, 0)
        self.conn = interim_conn.weights
        self.size = self.conn.shape[0]
        self.tract_len = np.zeros((self.size, self.size))
        self.a = 0.35
        self.w = 0.2
        self.sim_steps = 1
        self.init_cond = np.r_[[[np.ones((self.size, 1)), np.ones((self.size, 1))]]]

    def run(self):
        pass


class TvbModel(ModelWrapper):
    def __init__(self):
        super().__init__()

        conn = connectivity.Connectivity(
            weights=self.conn,
            tract_lengths=self.tract_len,
        )

        conn.centres_spherical(number_of_regions=self.size)
        conn.create_region_labels()

        hiss = tvbl.noise.Additive(nsig=np.r_[self.noise])

        self.sim = simulator.Simulator(
            connectivity=conn,
            model=SupHopf(a=np.r_[self.a], omega=np.r_[self.w]),
            integrator=EulerStochastic(dt=self.dt, noise=hiss),
            # integrator=EulerDeterministic(dt=self.dt),
            initial_conditions=self.init_cond,
            monitors=[Raw()],
            simulation_length=self.dt * self.sim_steps,
            coupling=coupling.Scaling(a=np.r_[1.0]),
        )
        self.sim.configure()

    def run(self):
        return self.sim.run()


class NeurolibModel(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.model = HopfModel(
            Cmat=self.conn,
            Dmat=self.tract_len,
        )
        self.model.params["xs_init"] = self.init_cond
        self.model.params["ys_init"] = self.init_cond
        self.model.params["duration"] = self.dt * self.sim_steps
        self.model.params["a"] = self.a
        self.model.params["w"] = self.w
        self.model.params["coupling"] = "additive"
        self.model.params["K_gl"] = 1.0
        self.model.params.sigma_ou = self.noise
        self.model.params.tau_ou = 1.0

    def run(self):
        self.model.run()
        return self.model.x


# TODO init model 10 times then run it once
def run_multi_nlb():
    arr = np.array([])
    for _ in range(10):
        neurolibModel = NeurolibModel()
        neurolib_result = neurolibModel.run()
        np.append(arr, neurolib_result)
    return arr


def run_multi_tvb():
    arr = np.array([])
    for _ in range(10):
        neurolibModel = NeurolibModel()
        neurolib_result = neurolibModel.run()
        np.append(arr, neurolib_result)
    return arr


tvbModel = TvbModel()

tvb_result = np.array(
    [np.reshape(tvbModel.run()[0][1].T, neurolib_result[0].shape) for _ in range(10)]
)


print(f"nlb shape: {neurolib_result.shape}\ntvb shape: {tvb_result.shape}")
np.testing.assert_allclose(tvb_result, neurolib_result)
