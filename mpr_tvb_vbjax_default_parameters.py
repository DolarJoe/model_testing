import numpy as np
import pandas as pd


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
