# change to the root directory of the project

from re import S
from tvb.simulator.models.oscillator import SupHopf
from tvb.simulator.lab import simulator, connectivity, coupling
from tvb.simulator.integrators import EulerStochastic, EulerDeterministic
from tvb.simulator.monitors import Raw
import tvb.simulator.lab as tvbl

import numpy as np
from neurolib.models.hopf import HopfModel


class ModelWrapper:

    def __init__(self):
        self.dt = 0.1
        self.make_conn()
        self.size = self.conn.weights.shape[0]
        self.a = 0.35
        self.w = 0.2
        self.sim_steps = 1
        np_rng = np.random.default_rng(seed=46)
        self.init_cond = np.r_[[[np_rng.random((self.size, 1)), np_rng.random((self.size, 1))]]]

    def make_conn(self):
        # weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = np.zeros((3, 3))
        self.conn = connectivity.Connectivity(weights=weights)
        self.conn.compute_region_labels()
        self.conn.centres_cubic()
        self.conn = connectivity.Connectivity.from_file()
        self.conn.tract_lengths = np.zeros_like(self.conn.weights)  # because of neurolib
        np.fill_diagonal(self.conn.weights, 0)  # remove self-connections
        self.conn.tract_lengths *= 0.0

    def run(self):
        pass


class TvbModel(ModelWrapper):
    def __init__(self):
        super().__init__()

        self.sim = simulator.Simulator(
            connectivity=self.conn,
            model=SupHopf(a=np.r_[self.a], omega=np.r_[self.w]),
            integrator=EulerDeterministic(dt=self.dt),
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
            Cmat=self.conn.weights,
            Dmat=self.conn.tract_lengths,
        )
        self.model.params["dt"] = self.dt
        self.model.params["xs_init"] = self.init_cond[0, 0]
        self.model.params["ys_init"] = self.init_cond[0, 1]
        self.model.params["duration"] = self.dt * self.sim_steps
        self.model.params["a"] = self.a
        self.model.params["w"] = self.w
        self.model.params["coupling"] = "additive"
        self.model.params["K_gl"] = 1.0
        self.model.params.tau_ou = 1.0

    def run(self):
        self.model.run()
        return self.model.x


def run_nlb():
    neurolibModel = NeurolibModel()
    return neurolibModel.run()


def run_tvb(neurolib_result_shape):
    tvbModel = TvbModel()
    return np.reshape(tvbModel.run()[0][1].T, neurolib_result_shape)


if __name__ == "__main__":

    tvbModel = TvbModel()

    neurolib_result = run_nlb()
    tvb_result = run_tvb(neurolib_result.shape)

    print(f"nlb shape: {neurolib_result.shape}\ntvb shape: {tvb_result.shape}")
    np.testing.assert_allclose(neurolib_result, tvb_result)
