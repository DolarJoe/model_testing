import numpy as np
import scipy.stats
from config import Config
from model import ExampleSimulatorWrapper


def run_test(config):
    resultA = ExampleSimulatorWrapper(config).run()
    # Use your selected models
    resultB = ExampleSimulatorWrapper(config).run()
    # Adjust tolerance as necessary.
    # Even equivalent models will produce differences larger than the default rtol
    np.testing.assert_allclose(resultA, resultB)  # , rtol=1e-6)


def dfun_test():
    # As there are several options for this test, it is up to the user to select one
    # An example implementation for MPR can be found in dfun tutorial Jupyter Notebook

    # @patrik toto nemusis implementovat
    pass


def noise_test(initial_conditons_seed, number_of_tests):

    for dt in [0.001, 0.01, 0.1]:
        resultsA = []
        resultsB = []

        config = Config(initial_conditions_seed=initial_conditons_seed)
        config.dt = dt
        config.init_cond_for_noise()

        # Run models with noise
        config.noise_seed = 0
        config.noise = 10.0
        resultsA = ExampleSimulatorWrapper(config).run().flatten()
        config.noise_seed = 0 + number_of_tests
        resultsB = ExampleSimulatorWrapper(config).run().flatten()

        # Statistical test
        ks_stat, ks_p = scipy.stats.ks_2samp(resultsB, resultsA)
        print(f"KS Test dt={dt} → Statistic: {ks_stat:.4f}, p-value: {ks_p:.4e}")


def connectivity_test(number_of_tests):
    for i in range(number_of_tests):
        config = Config(initial_conditions_seed=i)
        print(f"\rTest {i+1:04d}", end="\r")
        config.init_config_for_connectivity()
        run_test(config)
    print("\nCOMPLETE")


def delay_test():
    for dt in [0.001, 0.01, 0.1]:
        for speed in (round(x) for x in range(1, 25, 3)):
            for c_s in [0.01, 0.5, 1.0]:
                for i in range(10):
                    config_delay = Config(initial_conditions_seed=i)
                    config_delay.dt = dt
                    config_delay.speed = speed
                    config_delay.coupling_strength = c_s
                    print(f"Test {i+1:02d} with dt={config_delay.dt:.1e}, speed={config_delay.speed:02d}", end="\r")
                    config_delay.init_config_for_delays()
                    run_test(config_delay)
    print("\nCOMPLETE")


if __name__ == "__main__":

    dfun_test()

    print("############### Connectivity test ###############")
    connectivity_test(1000)
    print("############### Neurolib and TVB results are close enough! ###############")

    print("############### Delay test ###############")
    delay_test()
    print("############### Neurolib and TVB results are close enough! ###############")

    print("############### Noise test ###############")
    noise_test(initial_conditons_seed=26, number_of_tests=1000)
    print("############### Low pvalue means the distributions don't match ###############")
