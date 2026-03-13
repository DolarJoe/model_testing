"""Constants used throughout the brain benchmarking system."""

from typing import Final

# Connectivity types
CONNECTIVITY_TVB: Final[str] = "tvb"
CONNECTIVITY_RANDOM: Final[str] = "random"

# Model types
MODEL_JANSEN_RIT: Final[str] = "jansen-rit"
MODEL_MONTBRIO: Final[str] = "montbrio"
MODEL_HOPF: Final[str] = "hopf"

# File naming patterns
TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
FILENAME_PATTERN: Final[str] = "{id}_{simulator}_{model}_{content}.{format}"

# Directory names
UTILS_DIR: Final[str] = "utils"
MODELS_DIR: Final[str] = "models"
DRIVERS_DIR: Final[str] = "drivers"
RESULTS_DIR: Final[str] = "results"

# Driver naming convention
DRIVER_SUFFIX: Final[str] = "Driver"

# Memory monitoring
MEMORY_UNIT_MB: Final[float] = 1024 * 1024  # bytes to MB conversion
MEMORY_MONITOR_INTERVAL: Final[float] = 0.01  # seconds
