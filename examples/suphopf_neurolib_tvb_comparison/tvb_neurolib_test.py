import numpy as np
import scipy.stats
from neurolib_model import NeurolibModel
from suphopf_config import SupHopfConfig
from tvb_model import TvbModel


def run_test(config):
    neurolib_result = NeurolibModel(config).run()
    tvb_result = np.reshape(TvbModel(config).run(), neurolib_result.shape)
    np.testing.assert_allclose(neurolib_result, tvb_result, atol=1e-6)


def run_noise_comparison(initial_conditons_seed, number_of_tests):
    # @patrik toto nas v podstate nezaujima pre tvoj use case,
    # beztak sa neda rozumne porovnavat noise v TVB a Neurolibe
    pass


def connectivity_test(number_of_tests):
    for i in range(number_of_tests):
        config = SupHopfConfig(initial_conditions_seed=i)
        print(f"\rTest {i+1:04d}", end="\r")
        config.init_config_for_connectivity()
        run_test(config)
    print("\nCOMPLETE")


def delay_test():
    for dt in [0.001, 0.01, 0.1]:
        for speed in (round(x) for x in range(1, 25, 3)):
            for c_s in [0.01, 0.5, 1.0]:
                for i in range(10):
                    config_delay = SupHopfConfig(initial_conditions_seed=i)
                    config_delay.dt = dt
                    config_delay.speed = speed
                    config_delay.coupling_strength = c_s
                    print(f"Test {i+1:02d} with dt={config_delay.dt:.1e}, speed={config_delay.speed:02d}", end="\r")
                    config_delay.init_config_for_delays()
                    run_test(config_delay)
    print("\nCOMPLETE")


if __name__ == "__main__":

    print("############### Connectivity test ###############")
    connectivity_test(1000)
    print("############### Neurolib and TVB results are close enough! ###############")

    print("############### Delay test ###############")
    delay_test()
    print("############### Neurolib and TVB results are close enough! ###############")

    # print("############### Noise test (artificial, TVB compared to TVB) ###############")
    # run_noise_comparison(initial_conditons_seed=26, number_of_tests=1000)
    # print("############### Low pvalue means the distributions don't match ###############")
