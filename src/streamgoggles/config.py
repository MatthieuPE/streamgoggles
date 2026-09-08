"""Configuration loading and parameter spec management.

Handles:
- Fixed vs. free parameter specifications
- YAML config loading (streams, background, matched filter)
- Distance modulus list resolution
- Immutable config manipulation (fix/free methods)
- Eval grid enumeration
"""

import dataclasses
import itertools
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Parameter can be fixed or free (sampled)."""

    FIXED = "fixed"
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    DISCRETE = "discrete"


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """Single parameter specification: fixed scalar or free (range/discrete set).

    Rationale: Encapsulate parameter types so config loading can validate and
    convert freely. Frozen for immutability.
    """

    name: str
    dist_type: DistributionType
    value: Any = None  # for FIXED
    min_val: float | None = None  # for UNIFORM, LOG_UNIFORM
    max_val: float | None = None
    log: bool = False  # if True, UNIFORM means log-uniform
    values: list | None = None  # for DISCRETE

    def is_fixed(self) -> bool:
        """Return True if this parameter is fixed."""
        return self.dist_type == DistributionType.FIXED

    def sample(self, rng: np.random.Generator) -> float | int | str:
        """Draw a sample from this parameter spec.

        Returns:
            Scalar value (float, int, or string) sampled according to spec.

        Raises:
            ValueError if spec is fixed or ill-formed.
        """
        if self.dist_type == DistributionType.FIXED:
            return self.value
        if self.dist_type == DistributionType.UNIFORM:
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"{self.name!r}: UNIFORM spec requires min_val/max_val"
                )
            return float(rng.uniform(self.min_val, self.max_val))
        if self.dist_type == DistributionType.LOG_UNIFORM:
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"{self.name!r}: LOG_UNIFORM spec requires min_val/max_val"
                )
            if self.min_val <= 0 or self.max_val <= 0:
                raise ValueError(f"{self.name!r}: LOG_UNIFORM requires positive bounds")
            log_val = rng.uniform(np.log(self.min_val), np.log(self.max_val))
            return float(np.exp(log_val))
        if self.dist_type == DistributionType.DISCRETE:
            if not self.values:
                raise ValueError(
                    f"{self.name!r}: DISCRETE spec requires a non-empty values list"
                )
            idx = rng.integers(0, len(self.values))
            return self.values[idx]
        raise ValueError(f"{self.name!r}: unknown dist_type {self.dist_type!r}")


def _parse_param_spec(name: str, raw: Any) -> ParameterSpec:
    """Parse one YAML value into a ParameterSpec.

    Accepts the forms documented in the prompt: a bare scalar (fixed), a
    ``{min, max}``/``{min, max, log: true}`` dict (uniform/log-uniform), or a
    ``{values: [...]}`` dict (discrete). The shape alone determines the type;
    there is no explicit "type" key in the YAML.
    """
    if not isinstance(raw, dict):
        return ParameterSpec(name=name, dist_type=DistributionType.FIXED, value=raw)

    if "values" in raw:
        return ParameterSpec(
            name=name, dist_type=DistributionType.DISCRETE, values=list(raw["values"])
        )

    if "min" in raw and "max" in raw:
        log = bool(raw.get("log", False))
        dist_type = DistributionType.LOG_UNIFORM if log else DistributionType.UNIFORM
        return ParameterSpec(
            name=name,
            dist_type=dist_type,
            min_val=float(raw["min"]),
            max_val=float(raw["max"]),
            log=log,
        )

    raise ValueError(
        f"Cannot parse parameter spec for {name!r}: {raw!r} "
        "(expected a scalar, {min, max[, log]}, or {values: [...]})"
    )


def _parse_richness_spec(raw: dict) -> tuple[str, ParameterSpec]:
    """Parse the ``richness`` block, which nests a spec under one representation key.

    E.g. ``{surface_brightness: 30.0}`` (fixed) or
    ``{mass: {min: 1.0e3, max: 1.0e5, log: true}}`` (free). Exactly one of
    ``surface_brightness``/``mass``/``nstars`` must be present.
    """
    valid_kinds = ("surface_brightness", "mass", "nstars")
    kinds_present = [k for k in valid_kinds if k in raw]
    if len(kinds_present) != 1:
        raise ValueError(
            f"richness must specify exactly one of {valid_kinds}, got keys {list(raw)}"
        )
    kind = kinds_present[0]
    return kind, _parse_param_spec("richness", raw[kind])


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    """Stream/injection parameter specifications.

    Attributes:
        params: dict mapping param_name -> ParameterSpec
        background_fraction: fraction of samples with no stream (default 0.0)
        label_policy: "binary", "density", or "soft_distance"
        persist: whether to persist samples to disk
        richness_kind: which representation ("surface_brightness", "mass", or
            "nstars") the "richness" entry in `params` is expressed in.
    """

    params: dict[str, ParameterSpec]
    background_fraction: float = 0.0
    label_policy: str = "density"
    persist: bool = False
    richness_kind: str = "nstars"

    _RESERVED_KEYS = frozenset({"background_fraction", "label", "persist", "richness"})

    @classmethod
    def load(cls, path: str | Path) -> "StreamConfig":
        """Load stream config from YAML file.

        Parameters:
            path: Path to YAML config.

        Returns:
            StreamConfig instance.
        """
        with open(Path(path)) as f:
            raw = yaml.safe_load(f) or {}

        label = raw.get("label", {})
        label_policy = (
            label.get("policy", "density") if isinstance(label, dict) else str(label)
        )

        params = {
            name: _parse_param_spec(name, value)
            for name, value in raw.items()
            if name not in cls._RESERVED_KEYS
        }

        richness_kind = "nstars"
        if "richness" in raw:
            richness_kind, richness_spec = _parse_richness_spec(raw["richness"])
            params["richness"] = richness_spec

        return cls(
            params=params,
            background_fraction=float(raw.get("background_fraction", 0.0)),
            label_policy=label_policy,
            persist=bool(raw.get("persist", False)),
            richness_kind=richness_kind,
        )

    def free_parameters(self) -> list[str]:
        """Return list of parameter names that are free (sampled).

        Returns:
            List of free parameter names; empty if all fixed.
        """
        return [name for name, spec in self.params.items() if not spec.is_fixed()]

    def fix(self, name: str, value: Any) -> "StreamConfig":
        """Return new config with parameter fixed to value (immutable).

        Parameters:
            name: Parameter name to fix.
            value: Fixed value.

        Returns:
            New StreamConfig with this parameter fixed.
        """
        if name not in self.params:
            raise KeyError(f"Unknown parameter: {name!r}")
        new_params = dict(self.params)
        new_params[name] = ParameterSpec(
            name=name, dist_type=DistributionType.FIXED, value=value
        )
        return dataclasses.replace(self, params=new_params)

    def free(self, name: str, spec: dict) -> "StreamConfig":
        """Return new config with parameter free per spec (immutable).

        Parameters:
            name: Parameter name to free.
            spec: Distribution spec dict (e.g., {min, max} or {values}).

        Returns:
            New StreamConfig with this parameter free.
        """
        if not isinstance(spec, dict):
            raise TypeError(
                f"free() requires a distribution spec dict, got {spec!r}; use fix() for a scalar value."
            )
        new_params = dict(self.params)
        new_params[name] = _parse_param_spec(name, spec)
        return dataclasses.replace(self, params=new_params)


@dataclasses.dataclass(frozen=True)
class BackgroundConfig:
    """Background source and processing configuration.

    Attributes:
        source: "data_file", "light", or "injection"
        survey: survey name (e.g., "lsst")
        release: release name (e.g., "dp2", "yr1")
        source_config: source-specific kwargs
        dust_correction_enabled: whether to apply dust correction
        study_region: StudyRegion config dict (ignored for data_file)
        cuts: list of Cut dicts
        magnitude_clipping: dict of per-band {min, max} mag bounds
    """

    source: str
    survey: str = "lsst"
    release: str = "dp2"
    source_config: dict = dataclasses.field(default_factory=dict)
    dust_correction_enabled: bool = True
    study_region: dict | None = None
    cuts: list[dict] = dataclasses.field(default_factory=list)
    magnitude_clipping: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundConfig":
        """Load background config from YAML file.

        Parameters:
            path: Path to YAML config.

        Returns:
            BackgroundConfig instance.

        Notes:
            If `background.dust_correction.enabled` is not given explicitly,
            it defaults to True only for `source == "data_file"` — the real
            DP2 skim is not pre-dereddened, but both `light` and `injection`
            already account for extinction internally (see PLAN.md §6.1.6).
        """
        with open(Path(path)) as f:
            raw = yaml.safe_load(f) or {}

        bg = raw.get("background", {})
        source = bg["source"]
        source_config = bg.get(source, {})
        dust_correction = bg.get("dust_correction", {})
        dust_enabled = dust_correction.get("enabled", source == "data_file")

        return cls(
            source=source,
            survey=bg.get("survey", "lsst"),
            release=bg.get("release", "dp2"),
            source_config=source_config,
            dust_correction_enabled=bool(dust_enabled),
            study_region=raw.get("study_region"),
            cuts=raw.get("cuts", []),
            magnitude_clipping=raw.get("magnitude_clipping"),
        )


@dataclasses.dataclass(frozen=True)
class MatchedFilterConfig:
    """Matched filter and pixelization configuration.

    Attributes:
        reference_isochrone: dict with age (Gyr) and z (metallicity, mass fraction)
        bands: list of photometric bands (e.g., ["g", "r"])
        pixelization: PixelizationSpec dict with nside, image_size_pix, pixel_scale_deg, etc.
        distance_mode: "fixed" or "scan"
        distance_value: single distance modulus (if mode is "fixed")
        distance_range: dict with min, max, step (if mode is "scan")
        finalize_config: dict with enabled, smoothing_deg, background_subtract
    """

    reference_isochrone: dict
    bands: list[str]
    pixelization: dict
    distance_mode: str = "fixed"
    distance_value: float | None = None
    distance_range: dict | None = None
    finalize_config: dict = dataclasses.field(
        default_factory=lambda: {"enabled": False}
    )

    @classmethod
    def load(cls, path: str | Path) -> "MatchedFilterConfig":
        """Load matched filter config from YAML file.

        Parameters:
            path: Path to YAML config.

        Returns:
            MatchedFilterConfig instance.
        """
        with open(Path(path)) as f:
            raw = yaml.safe_load(f) or {}

        distance = raw.get("distance", {"mode": "fixed", "value": 17.5})
        distance_range = {
            k: distance[k] for k in ("min", "max", "step") if k in distance
        }

        return cls(
            reference_isochrone=raw["reference_isochrone"],
            bands=raw["bands"],
            pixelization=raw.get("pixelization", {}),
            distance_mode=distance.get("mode", "fixed"),
            distance_value=distance.get("value"),
            distance_range=distance_range or None,
            finalize_config=raw.get("finalize", {"enabled": False}),
        )

    def distance_moduli(self) -> list[float]:
        """Resolve distance configuration to list of trial distance moduli.

        Returns:
            List of trial distance moduli in ascending order.

        Rationale: All downstream code uses distance as a list (scan of length 1
        for fixed mode), enabling future scans without code duplication.
        """
        if self.distance_mode == "fixed":
            if self.distance_value is None:
                raise ValueError(
                    "distance_mode='fixed' requires distance_value to be set"
                )
            return [float(self.distance_value)]

        if self.distance_mode == "scan":
            if self.distance_range is None:
                raise ValueError(
                    "distance_mode='scan' requires distance_range with min/max/step"
                )
            dmin = float(self.distance_range["min"])
            dmax = float(self.distance_range["max"])
            step = float(self.distance_range["step"])
            if step <= 0:
                raise ValueError("distance_range.step must be positive")
            n = round((dmax - dmin) / step) + 1
            return [float(v) for v in np.linspace(dmin, dmin + (n - 1) * step, n)]

        raise ValueError(f"Unknown distance_mode: {self.distance_mode!r}")


@dataclasses.dataclass
class EvalGrid:
    """Enumerated evaluation set: Cartesian product of free parameter values.

    Attributes:
        points: list of dicts, each mapping param_name -> value
        seeds: list of int, one seed per point for reproducibility

    Rationale: Stores the fixed evaluation set separately from training, ensuring
    consistent evaluation across runs.
    """

    points: list[dict]
    seeds: list[int]

    def __post_init__(self):
        """Validate points and seeds have same length."""
        if len(self.points) != len(self.seeds):
            raise ValueError(
                f"EvalGrid: {len(self.points)} points but {len(self.seeds)} seeds"
            )


def _eval_grid_axis(
    spec: ParameterSpec, n_points_per_range: int, n_points_discrete: int | None
) -> list:
    """Return the candidate values one free parameter contributes to the grid."""
    if spec.dist_type == DistributionType.DISCRETE:
        values = list(spec.values)
        if n_points_discrete is not None and len(values) > n_points_discrete:
            idx = np.linspace(0, len(values) - 1, n_points_discrete)
            unique_idx = sorted({round(i) for i in idx})
            values = [values[i] for i in unique_idx]
        return values

    if spec.dist_type == DistributionType.UNIFORM:
        return [
            float(v)
            for v in np.linspace(spec.min_val, spec.max_val, n_points_per_range)
        ]

    if spec.dist_type == DistributionType.LOG_UNIFORM:
        log_grid = np.linspace(
            np.log(spec.min_val), np.log(spec.max_val), n_points_per_range
        )
        return [float(v) for v in np.exp(log_grid)]

    raise ValueError(f"Cannot build an eval grid axis for dist_type={spec.dist_type!r}")


def build_eval_grid(
    config: StreamConfig,
    n_points_per_range: int = 5,
    n_points_discrete: int | None = None,
    seed: int = 42,
) -> EvalGrid:
    """Build evaluation grid from free parameters (Cartesian product).

    Parameters:
        config: StreamConfig with parameter specs.
        n_points_per_range: number of equally-spaced points per uniform/log-uniform range.
        n_points_discrete: if specified, enumerate at most this many points per discrete set.
        seed: RNG seed for grid enumeration (ensures reproducibility).

    Returns:
        EvalGrid with enumerated points and fixed seeds.

    Rationale: Enumeration is deterministic and reproducible. Grid size grows as
    Cartesian product of per-parameter sizes, which is expected and documented.

    Raises:
        ValueError if all parameters are fixed (no grid to build).
    """
    free_names = config.free_parameters()
    if not free_names:
        raise ValueError(
            "No free parameters in config; nothing to build an eval grid from."
        )

    axis_values = {
        name: _eval_grid_axis(
            config.params[name], n_points_per_range, n_points_discrete
        )
        for name in free_names
    }

    combos = itertools.product(*(axis_values[name] for name in free_names))
    points = [dict(zip(free_names, combo)) for combo in combos]

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in points]

    return EvalGrid(points=points, seeds=seeds)
