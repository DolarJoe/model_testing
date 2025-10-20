from datetime import datetime
from typing import override
from matplotlib import pyplot as plt
import numpy as np
import scipy
import numpy as np
import tvb.simulator.lab as tvbl
from tvb.simulator.integrators import EulerStochastic
from tvb.simulator.lab import coupling, simulator
from tvb.simulator.models.oscillator import SupHopf
from tvb.simulator.monitors import Raw

from conn_testing import ModelWrapper
from scipy.stats import gaussian_kde

from ConnNoWarnings import ConnNoWarnings


class TvbModel(ModelWrapper):
    def __init__(self, noise_seed):
        super().__init__()

        self.rng = np.random.RandomState(seed=noise_seed)
        self.__set_noise(noise_seed=self.rng)
        self._init_sim()

    def _init_sim(self):

        self.sim = simulator.Simulator(
            connectivity=ConnNoWarnings.from_file(),
            model=SupHopf(a=np.r_[self.a], omega=np.r_[self.w]),
            integrator=EulerStochastic(dt=self.dt, noise=self.hiss),
            initial_conditions=self.init_cond,
            monitors=[Raw()],
            simulation_length=self.dt * self.sim_steps,
            coupling=coupling.Scaling(a=np.r_[1.0]),
        )
        self.sim.configure()

    def __set_noise(self, noise_seed):
        print("TVBMODEL")
        self.hiss = tvbl.noise.Additive(random_stream=noise_seed)

    def run(self):
        return self.sim.run()


class TvbDifferent(TvbModel):
    def __init__(self, noise_seed):
        super().__init__(noise_seed)
        self.__set_noise(noise_seed=self.rng)
        super()._init_sim()

    @override
    def __set_noise(self, noise_seed):
        print("TVBMODEL DIFFERENT")
        self.hiss = tvbl.noise.Additive(nsig=np.r_[9.2], random_stream=noise_seed)


def debug(init_seed, sample_size, model):
    res = []
    for i in range(init_seed, init_seed + sample_size):
        m = model(noise_seed=i)
        res.append(m.run()[0][1][:, :, :, 0][0][0][0])
    return np.r_[res]


if __name__ == "__main__":
    begin = datetime.now()
    sample_size = 300
    data = debug(0, sample_size, TvbModel)
    data2 = debug(sample_size, sample_size, TvbDifferent)
    print(scipy.stats.ks_2samp(data, data2))
    print(datetime.now() - begin)
    kde = gaussian_kde(data)
    kde2 = gaussian_kde(data2)
    x = lambda data_input: np.linspace(data_input.min(), data_input.max(), 200)
    plt.switch_backend("TkAgg")
    # plt.plot(x(data), kde(x(data)), label="KDE Estimate")
    # plt.plot(x(data2), kde2(x(data2)), label="KDE Estimate 2")
    plt.hist(data, bins=50, density=True, alpha=0.5)
    plt.hist(data2, bins=50, density=True, alpha=0.5)
    plt.legend()
    plt.title("Exponential Distribution (KDE vs Histogram)")
    plt.show()
    # plt.savefig("fig.png")
