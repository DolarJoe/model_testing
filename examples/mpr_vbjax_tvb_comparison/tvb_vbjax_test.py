from matplotlib import pyplot as plt
import scipy.stats
from mpr_config import MPRConfig
from vbjax_model import VBJaxModel
from tvb_mpr_model import TvbMPRModel
import numpy as np


def run_test(config: MPRConfig):
    config.coupling_strength = 1.0

    vbjax_result = VBJaxModel(config).run()
    tvb_result = np.reshape(TvbMPRModel(config).run(), vbjax_result.shape)
    np.testing.assert_allclose(vbjax_result, tvb_result, atol=1e-6)


def run_noise_comparison(initial_conditons_seed, number_of_tests):

    for state_var in [0, 1]:
        for dt in [0.001, 0.01, 0.1]:
            results_no_noise = []
            results_with_noise = []

            config = MPRConfig()
            config.dt = dt
            config.init_cond_for_noise()

            # Run model with noise
            config.noise_seed = 0
            config.noise = 10.0
            results_with_noise = TvbMPRModel(config).run()[0][state_var].flatten()

            # Run model without noise
            config.noise_seed = 0 + number_of_tests
            results_no_noise = VBJaxModel(config).run()[state_var].flatten()

            # Statistical test
            ks_stat, ks_p = scipy.stats.ks_2samp(results_no_noise, results_with_noise)
            print(f"KS Test dt={dt} → Statistic: {ks_stat:.4f}, p-value: {ks_p:.4e}")


def connectivity_test(number_of_tests):
    for i in range(number_of_tests):
        config = MPRConfig(initial_conditions_seed=i)
        print(f"Test {i+1}")
        config.init_config_for_connectivity()
        run_test(config)


def delay_test():
    # We're not testing delays for MPR VBjax
    pass


if __name__ == "__main__":

    print("############### Connectivity test ###############")
    connectivity_test(100)
    print("############### VBJax and TVB results are close enough! ###############")

    # No delay testing

    print("############### Noise test ###############")
    run_noise_comparison(initial_conditons_seed=26, number_of_tests=1000)
    print("############### Low pvalue means the distributions don't match ###############")
