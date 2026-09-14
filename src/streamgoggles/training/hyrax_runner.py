"""hyrax integration (sketch for later).

Registers dataset and model with hyrax, runs hyrax training pipeline.

Rationale: hyrax is the orchestration layer for production training
(distributed, async loading, advanced logging). Kept optional and separate
from plain_runner.py so early development can proceed without hyrax overhead.

Stage 1: Stub (structure only, raises NotImplementedError).
Stage 2+: Implement after plain_runner.py is validated.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamgoggles.datasets.stream_map_dataset import StreamMapDataset
    from streamgoggles.models.unet import UNet
    from streamgoggles.storage import ModelStore, SimulationStore

logger = logging.getLogger(__name__)


def register_dataset_with_hyrax(dataset: "StreamMapDataset", config: dict) -> str:
    """Register dataset with hyrax registry.

    Parameters:
        dataset: StreamMapDataset instance.
        config: Configuration dict.

    Returns:
        hyrax registry key (string).

    Raises:
        NotImplementedError (stub; implement Stage 2+).
    """
    raise NotImplementedError


def register_model_with_hyrax(model: "UNet", config: dict) -> str:
    """Register model with hyrax registry.

    Parameters:
        model: UNet instance.
        config: Configuration dict.

    Returns:
        hyrax registry key (string).

    Raises:
        NotImplementedError (stub; implement Stage 2+).
    """
    raise NotImplementedError


def train_with_hyrax(
    config_toml_path: str, model_store: "ModelStore", dataset_store: "SimulationStore"
) -> dict:
    """Load hyrax config and run training pipeline.

    Parameters:
        config_toml_path: Path to hyrax runtime config (TOML).
        model_store: ModelStore for checkpointing.
        dataset_store: SimulationStore for persisted samples.

    Returns:
        dict with training results and final checkpoint path.

    Raises:
        NotImplementedError (stub; implement Stage 2+).
    """
    raise NotImplementedError
