import numpy as np
import pytest
from mpr_tvb_vbjax_test_functions import run_test

from mpr_tvb_vbjax_test_parameters import test_parameters


@pytest.mark.parametrize("row", [row for _, row in test_parameters.iterrows()])
def test_one_case(row):
    assert np.allclose(*run_test(row))
