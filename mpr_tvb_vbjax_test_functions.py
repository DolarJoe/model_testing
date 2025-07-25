import os
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from tvb.simulator.lab import *
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from vbjax import MPRTheta, mpr_dfun

from mpr_tvb_vbjax_default_parameters import coupling_, default_values


def create_meshgrid_linspace(test_case: pd.Series) -> List[npt.NDArray]:
    return np.meshgrid(
        np.linspace(
            test_case.r_low,
            test_case.r_high,
            int(test_case.linspace_size),
        ),
        np.linspace(
            test_case.V_low,
            test_case.V_high,
            int(test_case.linspace_size),
        ),
    )


def load_or_generate_data_for_testcase(test_case: pd.Series) -> List[npt.NDArray]:
    file_path = f"test_data/random_points{test_case.r_low}_{test_case.r_high}_{test_case.V_low}_{test_case.V_high}.npz"

    if not os.path.exists(file_path):
        rng = np.random.default_rng(seed=7583146)

        r_random = rng.uniform(low=test_case.r_low, high=test_case.r_high, size=int(test_case.linspace_size))
        V_random = rng.uniform(low=test_case.V_low, high=test_case.V_high, size=int(test_case.linspace_size))
        save_data_for_testcase(test_case, r_random, V_random)

    with np.load(file_path) as data:
        return [data["r_random"], data["V_random"]]


def save_data_for_testcase(test_case: pd.Series, r_random: npt.NDArray, V_random: npt.NDArray):
    filename = f"random_points{test_case.r_low}_{test_case.r_high}_{test_case.V_low}_{test_case.V_high}.npz"
    np.savez_compressed(f"test_data/{filename}", r_random=r_random, V_random=V_random)


def run_tvb_implementation(test_case=default_values.iloc[0], test_data: List[npt.NDArray] | None = None):

    if not test_data:
        test_data = create_meshgrid_linspace(test_case)

    parameters_only = test_case[["tau", "I", "Delta", "J", "eta", "cr", "cv"]]

    param_dict = {k: np.array([v]) for k, v in parameters_only.items()}

    return MontbrioPazoRoxin(**param_dict, Gamma=np.r_[0.0]).dfun(
        test_data,
        coupling=coupling_,
    )


def run_vbjax_implementation(test_case=default_values.iloc[0], test_data: List[npt.NDArray] | None = None):
    if not test_data:
        test_data = create_meshgrid_linspace(test_case)
    parameters_only = test_case[["tau", "I", "Delta", "J", "eta", "cr", "cv"]]
    params_as_mprtheta = MPRTheta(**parameters_only.to_dict())
    return mpr_dfun(
        test_data,
        c=coupling_,
        p=params_as_mprtheta,
    )


def run_test(test_case: pd.Series):
    test_data = load_or_generate_data_for_testcase(test_case)

    return (
        run_tvb_implementation(test_case, test_data),
        run_vbjax_implementation(test_case, test_data),
    )
