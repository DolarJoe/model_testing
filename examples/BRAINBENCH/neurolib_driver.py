import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from base_driver import BaseDriver
from neurolib.models.hopf import HopfModel
from utils.constants import MODEL_HOPF


class NeurolibDriver(BaseDriver):
    MODELS = {
        # "jansen-rit" Not implemented yet
        # "montbrio" Not implemented yet
        MODEL_HOPF: HopfModel,
    }

    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {}

    def load_model(self, model_name: str, params: Dict[str, Any]) -> None:
        if model_name not in self.MODELS:
            raise ValueError(f"Model '{model_name}' not recognized in NeurolibDriver.")
        self.model_name = model_name
        self.params = {k: float(np.asarray(v).flat[0]) if isinstance(v, (np.ndarray, list)) else v for k, v in params.items()}

    def setup_network(self, connectivity_matrix: np.ndarray, delay_matrix: np.ndarray) -> None:
        if self.model_name is None:
            raise ValueError("Model name must be set before setting up network.")

        self.model = self.MODELS[self.model_name](Cmat=connectivity_matrix, Dmat=delay_matrix, seed=self.seed)
        if self.model is None:
            raise ValueError(f"Failed to initialize model '{self.model_name}'.")
        self.model.params.update(self.params)

    def run_simulation(self, batch_run: bool, duration: float, dt: float) -> float:
        if self.model is None:
            raise ValueError("Model must be loaded before running simulation.")

        self.model.params["dt"] = dt
        self.model.params["duration"] = duration

        start_time = time.perf_counter()
        self.model.run()
        return time.perf_counter() - start_time

    def get_results(self) -> Any:
        if self.model is None:
            raise ValueError("Model must be loaded before getting results.")
        return self.model.outputs

    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model must be loaded before getting parameters.")

        normalised_params = {}
        for key, value in self.model.params.items():
            if isinstance(value, np.ndarray):
                normalised_params[key] = value.tolist()
            else:
                normalised_params[key] = value
        return normalised_params

    def save_simulation_plot(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model must be loaded before saving plot.")

        plt.figure(figsize=(12, 4))
        plt.plot(self.model.t, self.model.x.T)
        plt.title(f"Brain Activity: {self.model_name}")
        plt.xlabel("Time [ms]")
        plt.ylabel("Activity")

        plt.savefig(path)
        plt.close()
