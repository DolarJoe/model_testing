import json
import os

import numpy as np
import pytest
from mpr_tvb_vbjax_test_functions import run_test

from mpr_tvb_vbjax_test_parameters import test_parameters

indices, rows = zip(*test_parameters.iterrows())
results = []


@pytest.fixture(scope="session")
def is_single_test(pytestconfig):
    # All command line args after 'pytest'
    args = pytestconfig.args  # list of CLI args, e.g. ['test.py::test_one_case[6]']
    # If exactly one argument that contains a nodeid with param, consider it a single test run
    single_test = len(args) == 1 and ("::" in args[0])
    return single_test


@pytest.mark.parametrize("row", rows, ids=indices)
def test_one_case(row, is_single_test, request):
    test_results = run_test(row)
    try:
        assert np.allclose(*test_results)
        results.append("1")
    except AssertionError as identifier:
        results.append("0")
        if not is_single_test:
            raise identifier
        call_id = request.node.callspec.id
        with open(f"results/failed_test_id_{call_id}.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "test_case": row.to_dict(),
                    "test_results": {
                        "tvb": test_results[0].tolist(),
                        "vbjax": test_results[1].tolist(),
                    },
                },
                file,
            )
        raise identifier


@pytest.fixture(scope="session", autouse=True)
def write_results(is_single_test):
    yield 0
    if is_single_test:
        return
    with open(f"results/all_test_results.csv", "w", encoding="utf-8") as file:
        file.write("test_results" + os.linesep)
        print(results)
        file.writelines(os.linesep.join(results))
