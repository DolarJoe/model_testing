from config import Config
from neurolib_model import NeurolibModel
from tvb_model import TvbModel
import numpy as np


def run_test(config):
    neurolib_result = NeurolibModel(config).run()
    tvb_result = np.reshape(TvbModel(config).run()[0][1].T, neurolib_result.shape)
    np.testing.assert_allclose(neurolib_result, tvb_result)


if __name__ == "__main__":

    config = Config(random_seed=46)
    config.init_config_for_connectivity()
    print("############### Connectivity test ###############")
    run_test(config)
    print("############### Neurolib and TVB results are close enough! ###############")

    # config_delay = Config(random_seed=46)
    # config_delay.init_config_for_delays()
    # print("############### Delay test ###############")
    # run_test(config_delay)
    # print("############### Neurolib and TVB results are close enough! ###############")
