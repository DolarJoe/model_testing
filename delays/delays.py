# change to the root directory of the project

from tvb.simulator.models.oscillator import SupHopf
from tvb.simulator.lab import simulator, connectivity, coupling
from tvb.simulator.integrators import EulerStochastic, EulerDeterministic
from tvb.simulator.monitors import Raw
import tvb.simulator.lab as tvbl

import numpy as np
from neurolib.models.hopf import HopfModel


class ModelWrapper:

    def __init__(self):
        self.sim_steps = 1
        self.history_length = 5
        self.dt = 0.1
        self.noise = 0.0
        self.size = 5
        self.conn = np.zeros((self.size, self.size))
        self.conn = connectivity.Connectivity.from_file()
        self.normalize_tract_lengths()
        self.conn.speed = np.r_[1.0]
        self.size = self.conn.weights.shape[0]
        self.tract_len = np.zeros((self.size, self.size))
        self.a = 0.35
        self.w = 0.2
        rng = np.random.default_rng()
        self.init_cond = rng.random((self.history_length, 2, self.size, 1))
        print(f"init cond shape: {self.init_cond.shape}")

    def normalize_tract_lengths(self):
        max_len = np.max(self.conn.tract_lengths)
        self.conn.tract_lengths /= max_len
        self.conn.tract_lengths *= self.history_length - 1

    def run(self):
        pass


class TvbModel(ModelWrapper):
    def __init__(self):
        super().__init__()

        self.sim = simulator.Simulator(
            connectivity=self.conn,
            model=SupHopf(a=np.r_[self.a], omega=np.r_[self.w]),
            # integrator=EulerStochastic(dt=self.dt, noise=hiss),
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
        self.model.params["xs_init"] = self.init_cond[:, 0, :, 0].T
        self.model.params["ys_init"] = self.init_cond[:, 1, :, 0].T
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
    print(arr)
    return arr


def run_multi_tvb(shape):
    return np.array([np.reshape(tvbModel.run()[0][1].T, shape) for _ in range(10)])


if __name__ == "__main__":

    tvbModel = TvbModel()

    neurolib_result = run_multi_nlb()
    tvb_result = run_multi_tvb(neurolib_result.shape)

    print(f"nlb shape: {neurolib_result.shape}\ntvb shape: {tvb_result.shape}")
    np.testing.assert_allclose(tvb_result, neurolib_result)
