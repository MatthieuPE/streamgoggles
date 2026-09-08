"""Parameter-addressed storage with canonical hashing and manifest queries.

Three stores:
- SimulationStore: labeled training/eval samples (map, label, valid_mask, metadata)
- BackgroundMapStore: cached background raw and finalized maps per distance
- ModelStore: trained model checkpoints and config snapshots

All use canonical parameter-dict hashing for addressability and determinism.

Rationale: Isolate storage logic (key generation, path resolution, I/O) from 
application logic. Enables flexible querying (e.g., "all samples with width in 
[0.1, 0.3]") and efficient caching.
"""

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ParameterStore:
    """Base class for parameter-keyed storage.
    
    Subclasses implement storage of different object types (samples, maps, models)
    all addressable by a canonical parameter dict hash.
    """
    
    def __init__(self, root: str | Path):
        """Initialize store with root directory.
        
        Parameters:
            root: Root directory for this store (will be created if absent).
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.parquet"
    
    def key(self, params: dict) -> str:
        """Generate canonical SHA-1 hash of params (order-independent, deterministic).
        
        Parameters:
            params: dict of parameter values.
        
        Returns:
            40-character hex SHA-1 hash.
        
        Rationale: Parameters may be nested or in any order; canonical JSON 
        serialization ensures same params always hash to same key.
        """
        raise NotImplementedError
    
    def path(self, params: dict) -> Path:
        """Resolve parameter dict to storage path.
        
        Parameters:
            params: dict of parameter values.
        
        Returns:
            Path to stored object (e.g., root / key[:4] / key[4:].npz).
        
        Rationale: Shallow subdirectory tree (first 4 hash chars) avoids 
        filesystem issues with flat directories.
        """
        raise NotImplementedError
    
    def exists(self, params: dict) -> bool:
        """Check if params are already stored.
        
        Parameters:
            params: dict of parameter values.
        
        Returns:
            True if this params dict has a stored object.
        """
        raise NotImplementedError
    
    def save(self, obj: Any, params: dict, metadata: dict | None = None) -> Path:
        """Save object to disk keyed by params.
        
        Parameters:
            obj: Object to save (numpy array, Sample, model state_dict, etc.).
            params: Parameter dict (used for key generation).
            metadata: optional dict of additional metadata to store.
        
        Returns:
            Path to saved object.
        
        Raises:
            ValueError if obj type is not supported by this store.
        """
        raise NotImplementedError
    
    def load(self, params: dict) -> Any:
        """Load and return object from disk.
        
        Parameters:
            params: Parameter dict (must match a previously saved object).
        
        Returns:
            Loaded object.
        
        Raises:
            FileNotFoundError if params not stored.
        """
        raise NotImplementedError
    
    def query(self, **ranges) -> pd.DataFrame:
        """Query manifest by parameter ranges.
        
        Parameters:
            **ranges: Keyword arguments specifying ranges, e.g., 
                width=(0.1, 0.3), age=12, surface_brightness=[20, 25, 30].
        
        Returns:
            DataFrame of matching entries from manifest.parquet.
        
        Rationale: Enables efficient selection of subsets without loading 
        all objects (e.g., "which samples have width ∈ [0.1, 0.3]?").
        """
        raise NotImplementedError


class SimulationStore(ParameterStore):
    """Storage for labeled training and evaluation samples.
    
    Each sample is a Sample dataclass (map_stack, label_stack, valid_mask, params, metadata),
    persisted as npz with metadata in a parquet manifest.
    
    Attributes (inherited):
        root: typically data/simulations/
        manifest_path: data/simulations/manifest.parquet
    
    Rationale: SimulationStore.get_or_generate() is the single access point 
    for the dataset, returning either a cached sample (eval mode) or generating 
    on-the-fly (training mode).
    """
    
    def save_sample(self, sample: "Sample", params: dict) -> Path:
        """Save a Sample (map, label, valid_mask, metadata) to disk.
        
        Parameters:
            sample: Sample instance with map_stack, label_stack, valid_mask, metadata.
            params: Parameter dict (used for addressing).
        
        Returns:
            Path to saved .npz file.
        
        Raises:
            ValueError if sample is malformed.
        """
        raise NotImplementedError
    
    def load_sample(self, params: dict) -> "Sample":
        """Load a Sample from disk.
        
        Parameters:
            params: Parameter dict matching a saved Sample.
        
        Returns:
            Sample instance.
        
        Raises:
            FileNotFoundError if params not stored.
        """
        raise NotImplementedError
    
    def get_or_generate(
        self, 
        params: dict, 
        generator_fn, 
        persist: bool = False
    ) -> "Sample":
        """Get sample from cache if persisted, else generate on-the-fly.
        
        Parameters:
            params: Parameter dict for sample.
            generator_fn: Callable(params) -> Sample if not in cache.
            persist: If True, save generated sample to cache.
        
        Returns:
            Sample (from cache or generated).
        
        Rationale: Unifies on-the-fly (training) and cached (eval) mode access.
        """
        raise NotImplementedError


class BackgroundMapStore(ParameterStore):
    """Storage for cached background maps (raw and optionally finalized).
    
    Key is hash of (source config, survey, release, study_region, cuts, clipping, 
    filter config, pixelization, distance_modulus).
    
    Each store entry contains:
        raw_map_full: HEALPix count map, shape (npix,)
        valid_mask_full: HEALPix bool mask, shape (npix,)
        finalized_map_full (optional): finalized map if background subtraction is enabled
    
    Attributes (inherited):
        root: typically data/background_maps/
    
    Rationale: Background maps are expensive to compute and used by many samples.
    Caching them per (source, survey, cuts, filter, distance) tuple enables efficient
    per-sample injection: reuse cached background, compute stream locally, combine.
    """
    
    def save_background(
        self,
        raw_map: np.ndarray,
        valid_mask: np.ndarray,
        finalized_map: np.ndarray | None,
        params: dict
    ) -> Path:
        """Save background maps to disk.
        
        Parameters:
            raw_map: Raw HEALPix count map, shape (npix,).
            valid_mask: HEALPix bool mask, shape (npix,).
            finalized_map: Optional finalized map (if smoothing/background-subtract enabled),
                shape (npix,).
            params: Parameter dict (used for addressing).
        
        Returns:
            Path to saved .npz file.
        """
        raise NotImplementedError
    
    def load_background(self, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Load background maps from disk.
        
        Parameters:
            params: Parameter dict.
        
        Returns:
            Tuple (raw_map, valid_mask, finalized_map or None).
        
        Raises:
            FileNotFoundError if params not stored.
        """
        raise NotImplementedError


class ModelStore(ParameterStore):
    """Storage for trained model checkpoints and metadata.
    
    Key is hash of (model config, label policy, training config, dataset config).
    
    Each store entry contains:
        state_dict: model weights (torch.nn.Module.state_dict() format)
        config_snapshot: full config dict at training time
        training_metrics: dict of final losses, epochs, etc.
    
    Attributes (inherited):
        root: typically data/models/
    
    Rationale: Checkpoint logic (state_dict I/O, config versioning) is isolated
    from training code. ModelStore.load_model() rebuilds the model object from
    a config snapshot and loads weights.
    """
    
    def save_model(
        self,
        model,
        config: dict,
        metrics: dict,
        params: dict
    ) -> Path:
        """Save model checkpoint to disk.
        
        Parameters:
            model: torch.nn.Module instance.
            config: Configuration dict (full config snapshot).
            metrics: Training metrics dict (losses, epochs, etc.).
            params: Parameter dict (used for addressing).
        
        Returns:
            Path to saved checkpoint.
        
        Raises:
            ValueError if model is not a torch.nn.Module.
        """
        raise NotImplementedError
    
    def load_model(self, model_class, params: dict):
        """Load model from checkpoint.
        
        Parameters:
            model_class: torch.nn.Module class (or factory function) to instantiate.
            params: Parameter dict matching a saved checkpoint.
        
        Returns:
            model_class instance with weights loaded from checkpoint.
        
        Raises:
            FileNotFoundError if params not stored.
        """
        raise NotImplementedError
