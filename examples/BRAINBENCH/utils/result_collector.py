"""Result handling for experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models.experiment_config import GlobalSettings, RunConfig
from utils.constants import (
    DRIVER_SUFFIX,
    FILENAME_PATTERN,
)


class ResultCollector:
    """Handles creation and saving of experiment results."""

    def __init__(self) -> None:
        self.result_dir: Path | None = None

    def set_result_dir(self, result_dir: Path) -> None:
        self.result_dir = result_dir

    def create_run_data(
        self,
        global_settings: GlobalSettings,
        run_config: RunConfig,
        timestamps: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Create standardized result data dictionary."""
        return {
            "id": run_config.id,
            "simulator": run_config.driver.__name__.replace(DRIVER_SUFFIX, ""),
            "model": run_config.model,
            "duration": global_settings.duration,
            "dt": global_settings.dt,
            "ground_truth": run_config.ground_truth,
            "batch_run": run_config.batch_run,
            "timestamps": timestamps,
        }

    def save_run_data(self, run_data: Dict[str, Any]) -> None:
        """Save result data to JSON file with timestamp."""
        if self.result_dir is None:
            raise ValueError("Result directory not set in ResultCollector.")
        with open(
            self.result_dir
            / FILENAME_PATTERN.format(
                id=run_data["id"],
                simulator=run_data["simulator"],
                model=run_data["model"],
                content="run_data",
                format="json",
            ),
            "w",
        ) as f:
            json.dump(run_data, f, indent=4)
