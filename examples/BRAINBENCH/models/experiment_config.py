"""Configuration models for experiments."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from base_driver import BaseDriver


@dataclass
class GlobalSettings:
    """Global experiment settings."""

    connectivity: str
    duration: float
    dt: float
    n_nodes: Optional[int] = None  # Required only for random connectivity


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""

    id: int
    driver: Type[BaseDriver]
    model: str
    params: Dict[str, Any]
    batch_run: bool = False
    ground_truth: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    result_dir: str
    global_settings: GlobalSettings
    runs: List[RunConfig]

    def validate(self) -> None:
        """Validate the experiment configuration."""
        from utils.constants import CONNECTIVITY_RANDOM

        if not os.path.exists(self.result_dir):
            raise ValueError(f"Result directory '{self.result_dir}' does not exist")

        if not self.runs:
            raise ValueError("'runs' list cannot be empty")

        if len(self.runs) != len({run.id for run in self.runs}):
            raise ValueError("Each run must have a unique 'id'")

        if self.global_settings.connectivity == CONNECTIVITY_RANDOM and self.global_settings.n_nodes is None:
            raise ValueError("Missing 'n_nodes' for random connectivity in global_settings")

        has_ground_truth = any(run.ground_truth for run in self.runs)
        if not has_ground_truth:
            raise ValueError("At least one run must have 'ground_truth' set to True")
