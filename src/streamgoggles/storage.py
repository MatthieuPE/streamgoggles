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

import hashlib
import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from streamgoggles.sample import Sample

logger = logging.getLogger(__name__)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/Path values into plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _flatten_for_manifest(params: dict) -> dict:
    """Flatten a params dict into manifest columns.

    Plain scalars become their own column (so `query()` can range/equality
    filter on them directly, e.g. `query(width=(0.1, 0.3))`). Nested
    dicts/lists are JSON-encoded into a single column — still exact-matchable,
    just not range-queryable, which is fine since none of the three stores'
    params (§5 of the build prompt) need range queries on nested values.
    """
    row = {}
    for k, v in params.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            row[k] = v
        else:
            row[k] = json.dumps(_to_jsonable(v), sort_keys=True)
    return row


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
        canonical = json.dumps(
            _to_jsonable(params), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def path(self, params: dict) -> Path:
        """Resolve parameter dict to storage path.

        Parameters:
            params: dict of parameter values.

        Returns:
            Path to stored object (e.g., root / key[:4] / key[4:].npz).

        Rationale: Shallow subdirectory tree (first 4 hash chars) avoids
        filesystem issues with flat directories. Returned path has no
        extension — callers append the suffix (or treat it as a directory)
        appropriate to what they're storing.
        """
        key = self.key(params)
        return self.root / key[:4] / key[4:]

    def exists(self, params: dict) -> bool:
        """Check if params are already stored.

        Parameters:
            params: dict of parameter values.

        Returns:
            True if this params dict has a stored object.
        """
        manifest = self._load_manifest()
        if manifest.empty:
            return False
        return self.key(params) in set(manifest["id"])

    def save(self, obj: Any, params: dict, metadata: dict | None = None) -> Path:
        """Save object to disk keyed by params.

        Parameters:
            obj: Object to save (numpy array, Sample, model state_dict, etc.).
            params: Parameter dict (used for key generation).
            metadata: optional dict of additional metadata to store.

        Returns:
            Path to saved object.

        Rationale: Generic pickle-based fallback for arbitrary picklable
        objects. Stores with a more specific on-disk format (SimulationStore,
        BackgroundMapStore: npz; ModelStore: torch state_dict + JSON) use
        their own typed save_*/load_* methods instead, but still go through
        `key`/`path`/`_upsert_manifest_row` here.
        """
        path = self.path(params).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        self._upsert_manifest_row(params, metadata)
        return path

    def load(self, params: dict) -> Any:
        """Load and return object from disk.

        Parameters:
            params: Parameter dict (must match a previously saved object).

        Returns:
            Loaded object.

        Raises:
            FileNotFoundError if params not stored.
        """
        path = self.path(params).with_suffix(".pkl")
        if not path.exists():
            raise FileNotFoundError(
                f"No object stored for params={params!r} (expected {path})"
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def query(self, **ranges) -> pd.DataFrame:
        """Query manifest by parameter ranges.

        Parameters:
            **ranges: Keyword arguments specifying ranges, e.g.,
                width=(0.1, 0.3), age=12, surface_brightness=[20, 25, 30].
                A 2-tuple is an inclusive range, a list/set is membership,
                anything else is equality.

        Returns:
            DataFrame of matching entries from manifest.parquet.

        Rationale: Enables efficient selection of subsets without loading
        all objects (e.g., "which samples have width ∈ [0.1, 0.3]?").
        """
        manifest = self._load_manifest()
        if manifest.empty:
            return manifest

        mask = pd.Series(True, index=manifest.index)
        for column, spec in ranges.items():
            if column not in manifest.columns:
                raise KeyError(
                    f"{column!r} is not a manifest column. Available: {list(manifest.columns)}"
                )
            if isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                mask &= manifest[column].between(lo, hi)
            elif isinstance(spec, (list, set, frozenset)):
                mask &= manifest[column].isin(spec)
            else:
                mask &= manifest[column] == spec
        return manifest[mask].reset_index(drop=True)

    def _load_manifest(self) -> pd.DataFrame:
        """Read manifest.parquet, or an empty DataFrame if it doesn't exist yet."""
        if not self.manifest_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.manifest_path)

    def _upsert_manifest_row(self, params: dict, metadata: dict | None) -> None:
        """Append (or replace) this params' manifest row, keyed by its hash.

        Re-saving the same params overwrites its row rather than accumulating
        duplicates — the key is a deterministic function of params, so there
        is exactly one manifest entry per distinct params dict.
        """
        key = self.key(params)
        row = {
            "id": key,
            "created_at": datetime.now(UTC).isoformat(),
            **_flatten_for_manifest(params),
            "metadata": json.dumps(_to_jsonable(metadata))
            if metadata is not None
            else None,
        }
        new_row = pd.DataFrame([row])

        manifest = self._load_manifest()
        if not manifest.empty:
            manifest = pd.concat(
                [manifest[manifest["id"] != key], new_row], ignore_index=True
            )
        else:
            manifest = new_row
        manifest.to_parquet(self.manifest_path, index=False)


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

    def save_sample(self, sample: Sample, params: dict) -> Path:
        """Save a Sample (map, label, valid_mask, metadata) to disk.

        Parameters:
            sample: Sample instance with map_stack, label_stack, valid_mask, metadata.
            params: Parameter dict (used for addressing).

        Returns:
            Path to saved .npz file.
        """
        path = self.path(params).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            map_stack=sample.map_stack,
            label_stack=sample.label_stack,
            valid_mask=sample.valid_mask,
            params_json=json.dumps(_to_jsonable(sample.params), sort_keys=True),
            metadata_json=(
                json.dumps(_to_jsonable(sample.metadata), sort_keys=True)
                if sample.metadata is not None
                else ""
            ),
        )
        self._upsert_manifest_row(params, sample.metadata)
        return path

    def load_sample(self, params: dict) -> Sample:
        """Load a Sample from disk.

        Parameters:
            params: Parameter dict matching a saved Sample.

        Returns:
            Sample instance.

        Raises:
            FileNotFoundError if params not stored.
        """
        path = self.path(params).with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(
                f"No sample stored for params={params!r} (expected {path})"
            )
        with np.load(path) as data:
            metadata_json = str(data["metadata_json"])
            return Sample(
                map_stack=data["map_stack"],
                label_stack=data["label_stack"],
                valid_mask=data["valid_mask"],
                params=json.loads(str(data["params_json"])),
                metadata=json.loads(metadata_json) if metadata_json else None,
            )

    def get_or_generate(
        self, params: dict, generator_fn, persist: bool = False
    ) -> Sample:
        """Get sample from cache if persisted, else generate on-the-fly.

        Parameters:
            params: Parameter dict for sample.
            generator_fn: Callable(params) -> Sample if not in cache.
            persist: If True, save generated sample to cache.

        Returns:
            Sample (from cache or generated).

        Rationale: Unifies on-the-fly (training) and cached (eval) mode access.
        """
        if persist and self.exists(params):
            return self.load_sample(params)

        sample = generator_fn(params)

        if persist:
            self.save_sample(sample, params)

        return sample


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
        params: dict,
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
        path = self.path(params).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"raw_map": raw_map, "valid_mask": valid_mask}
        if finalized_map is not None:
            arrays["finalized_map"] = finalized_map
        np.savez_compressed(path, **arrays)
        self._upsert_manifest_row(params, metadata=None)
        return path

    def load_background(
        self, params: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Load background maps from disk.

        Parameters:
            params: Parameter dict.

        Returns:
            Tuple (raw_map, valid_mask, finalized_map or None).

        Raises:
            FileNotFoundError if params not stored.
        """
        path = self.path(params).with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(
                f"No background map stored for params={params!r} (expected {path})"
            )
        with np.load(path) as data:
            raw_map = data["raw_map"]
            valid_mask = data["valid_mask"]
            finalized_map = data.get("finalized_map", None)
        return raw_map, valid_mask, finalized_map


class ModelStore(ParameterStore):
    """Storage for trained model checkpoints and metadata.

    Key is hash of (model config, label policy, training config, dataset config).

    Each store entry is a directory containing:
        state_dict.pt: model weights (torch.nn.Module.state_dict() format)
        config.json: full config snapshot at training time (also the kwargs
            used to reconstruct the model: `model_class(**config)`)
        metrics.json: dict of final losses, epochs, etc.

    Attributes (inherited):
        root: typically data/models/

    Rationale: Checkpoint logic (state_dict I/O, config versioning) is isolated
    from training code. ModelStore.load_model() rebuilds the model object from
    a config snapshot and loads weights.
    """

    def save_model(self, model, config: dict, metrics: dict, params: dict) -> Path:
        """Save model checkpoint to disk.

        Parameters:
            model: torch.nn.Module instance.
            config: Configuration dict (full config snapshot; also the kwargs
                `load_model` will pass to `model_class(**config)`).
            metrics: Training metrics dict (losses, epochs, etc.).
            params: Parameter dict (used for addressing).

        Returns:
            Path to saved checkpoint directory.

        Raises:
            TypeError if model is not a torch.nn.Module.
        """
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be a torch.nn.Module, got {type(model)!r}")

        model_dir = self.path(params)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_dir / "state_dict.pt")
        with open(model_dir / "config.json", "w") as f:
            json.dump(_to_jsonable(config), f, indent=2, sort_keys=True)
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(_to_jsonable(metrics), f, indent=2, sort_keys=True)

        self._upsert_manifest_row(params, metadata=None)
        return model_dir

    def load_model(self, model_class, params: dict):
        """Load model from checkpoint.

        Parameters:
            model_class: torch.nn.Module class (or factory function) to instantiate.
                Called as `model_class(**config)` using the config snapshot saved
                alongside the weights.
            params: Parameter dict matching a saved checkpoint.

        Returns:
            model_class instance with weights loaded from checkpoint.

        Raises:
            FileNotFoundError if params not stored.
        """
        import torch

        model_dir = self.path(params)
        state_dict_path = model_dir / "state_dict.pt"
        config_path = model_dir / "config.json"
        if not state_dict_path.exists():
            raise FileNotFoundError(
                f"No model stored for params={params!r} (expected {state_dict_path})"
            )

        with open(config_path) as f:
            config = json.load(f)

        model = model_class(**config)
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model
