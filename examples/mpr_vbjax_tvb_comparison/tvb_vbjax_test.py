import numpy as np
import scipy.stats
from mpr_config import MPRConfig
from tvb_mpr_model import TvbMPRModel
from vbjax_model import VBJaxModel


def run_test(config: MPRConfig):
    config.coupling_strength = 1.0

    vbjax_result = VBJaxModel(config).run()
    tvb_result = np.reshape(TvbMPRModel(config).run(), vbjax_result.shape)
    np.testing.assert_allclose(vbjax_result, tvb_result, rtol=5e-6)


def noise_test(number_of_tests):

    for state_var in [0, 1]:
        for noise_strength in [1, 10]:
            for dt in [0.001, 0.01, 0.05]:
                vbjax_result = []
                tvb_result = []

                config = MPRConfig()
                config.dt = dt
                config.init_cond_for_noise()

                # Run model with noise
                config.noise_seed = 13
                config.noise = noise_strength
                tvb_result = TvbMPRModel(config).run()[0][state_var].flatten()

                # Run model without noise
                config.noise_seed = 0 + number_of_tests

                # SCALE THE NOISE TO MATCH TVB
                # @patrik ked si spustis test tak uvidis ze failuje na niektorych dt hodnotach,
                # konkretne MPR je velmi citlive na dt
                # Ale ak by si odstranil tento scaling (je vytiahnuty z TVB kodu, VBJax nic take nerobi)
                # tak to bude failovat uplne
                config.noise = np.sqrt(2.0 * config.noise)

                vbjax_result = VBJaxModel(config).run()[state_var].flatten()

                # Statistical test
                ks_stat, ks_p = scipy.stats.ks_2samp(vbjax_result, tvb_result)
                print(f"KS Test dt={dt:.3f}, state_var={state_var}, noise_strength={noise_strength} → Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")


def connectivity_test(number_of_tests):
    print("running connectivity testing")
    for i in range(number_of_tests):
        config = MPRConfig(initial_conditions_seed=i)
        # print(f"Test {i+1}")
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
    noise_test(initial_conditons_seed=26, number_of_tests=1000)
    print("############### Low pvalue (>0.05) means the distributions don't match ###############")
