from matplotlib import pyplot as plt
import scipy.stats
from config import Config
from neurolib_model import NeurolibModel
from tvb_model import TvbModel
import numpy as np
import seaborn as sns


def run_test(config):
    neurolib_result = NeurolibModel(config).run()
    tvb_result = np.reshape(TvbModel(config).run(), neurolib_result.shape)
    np.testing.assert_allclose(neurolib_result, tvb_result, atol=1e-6)


def run_noise_comparison(initial_conditons_seed, number_of_tests):

    results_no_noise = []
    results_with_noise = []

    config = Config(initial_conditions_seed=initial_conditons_seed)
    config.init_cond_for_noise()

    # Run model with noise
    config.noise_seed = 0
    config.noise = 10.0
    results_with_noise = TvbModel(config).run().flatten()

    # Run model without noise
    config.noise_seed = 0 + number_of_tests
    config.noise = 0.1

    config.noise = 0
    results_no_noise = TvbModel(config).run().flatten()

    # Statistical test
    ks_stat, ks_p = scipy.stats.ks_2samp(results_no_noise, results_with_noise)
    print(f"KS Test → Statistic: {ks_stat:.4f}, p-value: {ks_p:.4e}")


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
