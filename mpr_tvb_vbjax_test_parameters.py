import itertools

import numpy as np
import pandas as pd


import numpy as np
import itertools

# Coupling
coupling_ = np.array([[0], [0]])

default_values = pd.DataFrame(
    {
        "tau": 1.0,
        "I": 0.0,
        "Delta": 1.0,
        "J": 15.0,
        "eta": -5.0,
        "cr": 1.0,
        "cv": 0.0,
        "r_low": 0.0,
        "r_high": 2.0,
        "V_low": -2.0,
        "V_high": 1.5,
        "linspace_size": 100,
    },
    index=[0],
)


# _test_cases = [
#     {"eta": [-10, -5, -0.001]},
#     {"tau": [0.001, 5, 8, 12, 14.9]},
#     {"I": [-10, 10]},
# ]


_test_cases = [
    dict(zip(["tau", "I", "Delta", "J", "eta", "cr", "cv"], values))
    for values in itertools.product(
        np.linspace(0.001, 15.0, 3),  # tau
        np.linspace(-10.0, 10.0, 3),  # I
        np.linspace(0.0, 10.0, 3),  # Delta
        np.linspace(-25.0, 25.0, 3),  # J
        np.linspace(-10.0, 10.0, 3),  # eta
        np.linspace(0.0, 1.0, 3),  # cr
        np.linspace(0.0, 1.0, 3),  # cv
    )
]


def _expand_test_cases(base_cases):
    """Expands each test case dict that contains lists into all combinations."""
    expanded = []
    for case in base_cases:
        keys, values = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in case.items()])
        for combo in itertools.product(*values):
            expanded.append(dict(zip(keys, combo)))

    default_values_dict = default_values.iloc[0].to_dict()
    problematic_test_cases = []

    for test_case in expanded:
        if not set(test_case.keys()).issubset(default_values_dict.keys()):
            problematic_test_cases.append(test_case)
    if problematic_test_cases:
        raise ValueError(
            f"The values in these test cases don't match the default parametes: {problematic_test_cases}"
        )

    full_tests = [default_values.iloc[0].to_dict() | case for case in expanded]
    df = pd.DataFrame(full_tests)
    df.to_csv("test_data/test_data.csv", index=False)
    return df


test_parameters = _expand_test_cases(_test_cases)
