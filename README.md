# Test suite for tvb and vbjax implementations of MPR model

## How to run

Run the full test suite with the following command
``` bash
pytest mpr_tvb_vbjax_test.py
```

Or only a single (presumably failing) test with a specific id via this command, replacing `[id]` with for example `[42]`

``` bash 
pytest mpr_tvb_vbjax_test.py::test_one_case[id]
```
When running only 1 test, the framework will save the test parameters as well as actual results from the models into `results/failed_test_id_[id].json`

All of the test parameters can be viewed in `test_data/test_data.csv` and the test ids roughly match line numbers (off by one error)

## TODO

When running full suite, save passing/failing test results so that will map to `test_data.csv`