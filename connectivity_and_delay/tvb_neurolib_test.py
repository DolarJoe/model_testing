from matplotlib import pyplot as plt
import scipy
from config import Config
from neurolib_model import NeurolibModel
from tvb_model import TvbModel
import numpy as np


def run_test(config):
    neurolib_result = NeurolibModel(config).run()
    tvb_result = np.reshape(TvbModel(config).run(), neurolib_result.shape)
    np.testing.assert_allclose(neurolib_result, tvb_result, atol=1e-6)


def run_noise_comparison(initial_conditons_seed, number_of_tests):
    seeds_for_noise = list(range(number_of_tests))
    results_no_noise = []
    results_with_noise = []
    for i in seeds_for_noise:
        config = Config(initial_conditions_seed=initial_conditons_seed)
        config.init_config_for_connectivity()

        config.noise_seed = i
        config.noise = 0.8999
        results_no_noise.append(TvbModel(config).run()[0, 0, 0, 0])

        config.noise_seed = i + number_of_tests
        config.noise = 9.0
        results_with_noise.append(TvbModel(config).run()[0, 0, 0, 0])

    print(scipy.stats.ks_2samp(results_no_noise, results_with_noise))
    plt.hist(results_no_noise, bins=50, density=True, alpha=0.5, color="green")
    plt.hist(results_with_noise, bins=50, density=True, alpha=0.5, color="red")
    plt.show()


if __name__ == "__main__":

    config = Config(initial_conditions_seed=46)
    config.init_config_for_connectivity()
    print("############### Connectivity test ###############")
    run_test(config)
    print("############### Neurolib and TVB results are close enough! ###############")

    config_delay = Config(initial_conditions_seed=46)
    config_delay.init_config_for_delays()
    print("############### Delay test ###############")
    run_test(config_delay)
    print("############### Neurolib and TVB results are close enough! ###############")

    print("############### Noise test ###############")
    run_noise_comparison(initial_conditons_seed=26, number_of_tests=1000)
    print("############### Low pvalue means the distributions don't match ###############")
