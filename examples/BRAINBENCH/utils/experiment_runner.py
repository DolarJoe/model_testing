"""Main experiment execution logic."""

import itertools
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from base_driver import BaseDriver
from models.experiment_config import ExperimentConfig, GlobalSettings, RunConfig
from utils.constants import (
    CONNECTIVITY_RANDOM,
    CONNECTIVITY_TVB,
    DRIVER_SUFFIX,
    FILENAME_PATTERN,
)
from utils.matrices import generate_matrices, load_matrices_from_tvb
from utils.result_collector import ResultCollector


class ExperimentRunner:
    """Handles the execution of brain simulation experiments."""

    def __init__(self) -> None:
        self.experiment_config: ExperimentConfig | None = None
        self.result_collector: ResultCollector = ResultCollector()

    def run(self, experiment_config: ExperimentConfig) -> None:
        try:
            experiment_config.validate()
        except ValueError as e:
            print(f"❌ Invalid experiment configuration: {e}")
            return

        self.result_collector.set_result_dir(Path(experiment_config.result_dir))
        self.experiment_config = experiment_config
        self.connectivity_matrix, self.delay_matrix = self._prepare_network_matrices(experiment_config.global_settings)

        self._print_experiment_info(self.experiment_config.global_settings, self.connectivity_matrix)

        self._execute_runs()

    def _prepare_network_matrices(self, global_settings: GlobalSettings) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare connectivity and delay matrices based on configuration."""
        print(f"\nConnectivity: {global_settings.connectivity}")

        if global_settings.connectivity == CONNECTIVITY_RANDOM:
            if global_settings.n_nodes is None:
                raise ValueError("n_nodes is required for random connectivity")
            return generate_matrices(global_settings.n_nodes)
        elif global_settings.connectivity == CONNECTIVITY_TVB:
            return load_matrices_from_tvb()
        else:
            raise ValueError(f"Unknown connectivity type: {global_settings.connectivity}")

    def _print_experiment_info(self, global_settings: GlobalSettings, connectivity_matrix: np.ndarray) -> None:
        """Print experiment information."""
        print(f"Network: {connectivity_matrix.shape[0]} nodes")
        print(f"Duration: {global_settings.duration}ms")
        print(f"Time step (dt): {global_settings.dt}ms")

    def _execute_runs(self) -> None:
        """Execute a complete experiment based on configuration."""
        if self.experiment_config is None:
            raise ValueError("Experiment configuration is not set.")

        start_time = time.perf_counter()  # Measure whole experiment execution time

        # Sort runs to execute ground truth runs first
        # for further comparison of sim results
        for run_config in sorted(self.experiment_config.runs, key=lambda r: r.ground_truth, reverse=True):
            try:
                self._execute_run(
                    self.experiment_config.global_settings,
                    run_config,
                    self.connectivity_matrix,
                    self.delay_matrix,
                )
            except Exception as e:
                print(f"\t❌ Error during simulation: {e}")
                print("\n--- FULL TRACEBACK START ---")
                traceback.print_exc()
                print("--- FULL TRACEBACK END ---")
        print(f"\n🎉 All runs completed in {time.perf_counter() - start_time:.2f}s.")

    def _execute_run(
        self,
        global_settings: GlobalSettings,
        run_config: RunConfig,
        connectivity_matrix: np.ndarray,
        delay_matrix: np.ndarray,
    ) -> None:
        """Execute a single experiment run."""
        sim_name = run_config.driver.__name__.replace(DRIVER_SUFFIX, "")
        print(f"\n🚀 Running [{run_config.id}] {sim_name} with {run_config.model}...")

        if run_config.ground_truth:
            print("\tGround truth run.")

        if run_config.batch_run:
            print("\tBatch run.")
            self._execute_batch_run(global_settings, run_config, connectivity_matrix, delay_matrix)
        else:
            self._execute_parameter_sweep_run(global_settings, run_config, connectivity_matrix, delay_matrix)

    def _execute_batch_run(
        self,
        global_settings: GlobalSettings,
        run_config: RunConfig,
        connectivity_matrix: np.ndarray,
        delay_matrix: np.ndarray,
    ) -> None:
        """Execute a batch run (single simulation with all parameters)."""
        timestamps: List[Tuple[str, str]] = []
        driver = run_config.driver()

        # self.system_monitor.start()
        self._add_timestamp(timestamps, run_config.id, "Loading model")
        driver.load_model(run_config.model, run_config.params)
        self._add_timestamp(timestamps, run_config.id, "Setting up network")
        driver.setup_network(connectivity_matrix, delay_matrix)
        self._add_timestamp(timestamps, run_config.id, "Running simulation")
        sim_time = driver.run_simulation(run_config.batch_run, global_settings.duration, global_settings.dt)
        self._add_timestamp(timestamps, run_config.id, "Simulation completed")

        print("\t1/1 | ", end="")
        print(f"{run_config.params} | ", end="")
        print(f"{sim_time:.2f}s")

        self._process_run_data(driver, global_settings, run_config, timestamps)

    def _execute_parameter_sweep_run(
        self,
        global_settings: GlobalSettings,
        run_config: RunConfig,
        connectivity_matrix: np.ndarray,
        delay_matrix: np.ndarray,
    ) -> None:
        """Execute parameter sweep (multiple simulations with params combinations)."""
        timestamps: List[Tuple[str, str]] = []
        driver = run_config.driver()
        param_combinations = self._get_params_combinations(run_config.params)

        sim_time = 0.0  # Measuring simulation time for one parameter combination
        for i, param_combination in enumerate(param_combinations):
            self._add_timestamp(
                timestamps,
                run_config.id,
                f"{i + 1}/{len(param_combinations)} Loading model",
            )
            driver.load_model(run_config.model, param_combination.copy())
            self._add_timestamp(
                timestamps,
                run_config.id,
                f"{i + 1}/{len(param_combinations)} Setting up network",
            )
            driver.setup_network(connectivity_matrix, delay_matrix)
            self._add_timestamp(
                timestamps,
                run_config.id,
                f"{i + 1}/{len(param_combinations)} Running simulation",
            )
            sim_time = driver.run_simulation(run_config.batch_run, global_settings.duration, global_settings.dt)
            self._add_timestamp(
                timestamps,
                run_config.id,
                f"{i + 1}/{len(param_combinations)} Simulation completed",
            )

            print(f"\t{i + 1}/{len(param_combinations)} | ", end="")
            print(f"{param_combination} | ", end="")
            print(f"{sim_time:.2f}s")

        self._process_run_data(driver, global_settings, run_config, timestamps)

    def _add_timestamp(self, timestamps: List[Tuple[str, str]], run_id: int, step_desc: str) -> None:
        """Add a timestamp for a specific step in the simulation."""
        timestamps.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{run_id} {step_desc}"))

    def _get_params_combinations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all combinations of params from a dict of parameter lists."""
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations

    def _process_run_data(
        self,
        driver: BaseDriver,
        global_settings: GlobalSettings,
        run_config: RunConfig,
        timestamps: List[Tuple[str, str]],
    ) -> None:
        run_data = self.result_collector.create_run_data(
            global_settings,
            run_config,
            timestamps,
        )
        self.result_collector.save_run_data(run_data)

        if self.result_collector.result_dir is None:
            raise ValueError("Result directory not set in ResultCollector.")
        driver.save_simulation_plot(
            self.result_collector.result_dir
            / FILENAME_PATTERN.format(
                id=run_config.id,
                simulator=run_config.driver.__name__.replace(DRIVER_SUFFIX, ""),
                model=run_config.model,
                content="activity_plot",
                format="png",
            )
        )
