from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BaseDriver(ABC):
    """Abstract Base Class for all brain simulator drivers.
    Ensures a consistent interface for benchmarking different implementations.
    """

    MODELS: Dict[str, Any] = {}

    def __init__(self) -> None:
        self.seed = 42
        self.model = None
        self.results = None
        self.model_name: str | None = None

    @abstractmethod
    def load_model(self, model_name: str, params: Dict[str, Any]) -> None:
        """Initialize the specific neural model or oscillator.
        :param model_name: Name of the model (e.g., 'Hopf', 'Kuramoto')
        :param params: Dictionary of parameters for the model
        """
        pass

    @abstractmethod
    def setup_network(
        self, connectivity_matrix: np.ndarray, delay_matrix: np.ndarray
    ) -> None:
        """Set up the brain network structure using a weights matrix.
        :param connectivity_matrix: Square matrix of connection weights
        """
        pass

    @abstractmethod
    def run_simulation(self, batch_run: bool, duration: float, dt: float) -> float:
        """Execute the simulation and return the execution time.
        :param batch_run: Whether to run a batch simulation (multiple parameter sets)
        :param duration: Simulation time in milliseconds
        :param dt: Integration time step
        :return: Execution time (wall-clock) in seconds
        """
        pass

    @abstractmethod
    def save_simulation_plot(self, path: Path) -> None:
        """Save a plot of the simulation results to the specified path."""
        pass

    @abstractmethod
    def get_results(self) -> Any:
        """Return the simulation output (e.g., time series data)."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return the current model parameters."""
        pass
