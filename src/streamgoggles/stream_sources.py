"""Pluggable stream source implementations (streamobs, external sims, etc).

Rationale: Abstract "where stream stars come from" so the injector doesn't care
whether it's streamobs, pre-realized external sims, or future sources. Stream stars
are realized in stream frame (phi1, phi2, dist) with TRUE magnitudes and converted
to sky coordinates on-the-fly.
"""

import logging
from typing import Protocol

import numpy as np
import pandas as pd
from streamobs.columns import true_col
from streamobs.model import StreamModel

logger = logging.getLogger(__name__)


class StreamSource(Protocol):
    """Protocol for generating stream star catalogs.

    All implementations realize stream in stream frame (phi1, phi2, dist),
    returning TRUE (unreddened) magnitudes and an is_stream flag.

    Rationale: Protocol enables pluggable sources; downstream injection
    applies survey selection and coordinate transformation to all sources
    identically.
    """

    def realize(self, params: dict, rng: np.random.Generator) -> pd.DataFrame:
        """Realize one stream sample.

        `params` is a stream parameter dict; see the module/class docstrings
        of implementations (e.g. `StreamObsSource.realize`) for the exact
        keys expected, roughly: morphology ("uniform" or "spline"), width
        and length (degrees), distance_modulus, age (Gyr), z (metallicity
        mass fraction), nstars, plus other params as needed per morphology
        or source.

        Parameters:
            params: Stream parameter dict (see above).
            rng: np.random.Generator instance for reproducibility.

        Returns:
            DataFrame with exactly these columns -- the minimal information
            downstream code needs, nothing else (e.g. no velocity or mass
            columns, even if the underlying generator happens to produce
            them): ``phi1``/``phi2`` (stream frame coordinates, degrees),
            ``dist`` (TRUE distance modulus, mag -- despite the name, this
            is a distance modulus, not a physical distance in kpc/pc;
            streamobs's own column name, kept as-is for consistency with
            streamobs's convention), the two ``<survey>_<band>_true``
            columns (true magnitudes in streamobs convention, before survey
            errors, for whichever two bands were requested), and
            ``is_stream`` (bool, always True).

        Raises:
            ValueError if params are invalid or inconsistent.
        """


class StreamObsSource(StreamSource):
    """Generate stream via streamobs.model.StreamModel.

    Wraps streamobs's stream generation, using the isochrone and survey
    parameters passed through params.

    Rationale: streamobs is the canonical stream generation library.
    Wrapping it here allows reuse without importing streamobs everywhere,
    and handles conversion from user-friendly richness specs (surface
    brightness, mass, nstars) to streamobs's nstars parameter.

    Implementation note (verified against streamobs/model.py): both
    morphologies use the plain `StreamModel` class, NOT
    `SplineStreamModel` — `SplineStreamModel.__init__` unconditionally
    injects a `stream_name` kwarg into every sub-config, meant for the
    *file*-backed interpolation classes (`FileCubicSplineInterpolation`,
    `FileLinearDensityCubicSplineInterpolation`); the *inline* ones this
    class uses (`CubicSplineInterpolation`,
    `LinearDensityCubicSplineInterpolation`) don't accept that kwarg at all
    and raise `TypeError` if given it. Passing the same section types
    directly to `StreamModel`'s plain `density`/`track` sections sidesteps
    this entirely and was confirmed to work end-to-end.
    """

    def realize(self, params: dict, rng: np.random.Generator) -> pd.DataFrame:
        """Realize stream via streamobs.

        Parameters:
            params: dict of stream parameters -- see the key reference below.
            rng: np.random.Generator instance for reproducibility.

        ``params`` keys:

        - ``morphology``: "uniform" | "spline" (decision 5 -- no other values).
        - ``nstars``: true number of stars to generate (already converted
          from richness upstream -- see inject_utils.py; never computed here).
        - ``distance_modulus``: true distance modulus of the stream (mag).
        - ``age``: population age (Gyr).
        - ``z``: population metallicity (mass fraction).
        - ``survey``, ``release``: optional, default "lsst"/"dp2" -- the
          isochrone/column namespace for the true-magnitude columns
          (``<survey>_<band>_true``; release is dropped from that name by
          streamobs's own `true_col` convention).
        - ``band_1``, ``band_2``: optional, default "g"/"r".
        - For ``morphology="uniform"``:

          - ``width``: cross-track Gaussian sigma (degrees).
          - ``length``: on-sky track length (degrees) -> phi1 sampled
            uniformly in [-length/2, length/2].

        - For ``morphology="spline"``:

          - ``control_points``: list of >=2 dicts ``{"phi1": ..., "phi2":
            ..., "width": ... (optional)}`` defining the track center line
            (phi2 vs phi1) and, if given per-point, the local cross-track
            width. Sorted by phi1 internally; need not be pre-sorted.
          - ``width``: fallback constant width (degrees) used for any
            control point that doesn't specify its own "width".
          - Density along the track is uniform (flat intensity) by
            construction -- non-uniform density profiles are not exposed
            here (not required by decision 5).

        Returns:
            DataFrame with exactly phi1, phi2, dist, is_stream, and the two
            true-magnitude columns for band_1/band_2 (see StreamSource.realize's
            docstring) — mu1/mu2/rv (no velocity model configured -> always
            NaN) and the internal-only initial `mass` column are dropped.

        Raises:
            ValueError if morphology is not "uniform"/"spline", if a
                spline has fewer than 2 control points, if a control
                point is missing "width" with no params["width"] fallback,
                or if streamobs unexpectedly didn't produce the requested
                true-magnitude columns.
        """
        morphology = params["morphology"]
        nstars = int(params["nstars"])
        distance_modulus = float(params["distance_modulus"])
        age = params["age"]
        z = params["z"]
        survey = params.get("survey", "lsst")
        release = params.get("release", "dp2")
        band_1 = params.get("band_1", "g")
        band_2 = params.get("band_2", "r")

        isochrone_cfg = {
            "name": "Marigo2017",
            "survey": survey,
            "release": release,
            "age": age,
            "z": z,
            "band_1": band_1,
            "band_2": band_2,
        }
        distance_modulus_cfg = {
            "center": {"type": "Constant", "value": distance_modulus},
            "spread": {"type": "Constant", "value": 0.0},
        }

        if morphology == "uniform":
            density_cfg, track_cfg = self._uniform_density_track_config(params)
        elif morphology == "spline":
            density_cfg, track_cfg = self._spline_density_track_config(params)
        else:
            raise ValueError(
                f"Unknown morphology {morphology!r}; expected 'uniform' or 'spline' (decision 5)"
            )

        config = {
            "density": density_cfg,
            "track": track_cfg,
            "distance_modulus": distance_modulus_cfg,
            "isochrone": isochrone_cfg,
        }

        model = StreamModel(config)
        df = model.sample(nstars, rng=rng)
        df["is_stream"] = True

        # Keep only what downstream code actually needs (Protocol docstring):
        # phi1/phi2/dist, is_stream, and the two TRUE magnitude columns for the
        # requested bands. streamobs's `sample()` also returns mu1/mu2/rv
        # (always NaN here — no velocity model is configured) and the initial
        # stellar `mass` (only needed internally to sample magnitudes) — both
        # are noise at this stage, not signal, so they're dropped rather than
        # carried through the pipeline unused.
        mag_cols = [true_col(band_1, survey), true_col(band_2, survey)]
        keep_cols = ["phi1", "phi2", "dist", "is_stream", *mag_cols]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"streamobs.model.StreamModel did not produce expected columns "
                f"{missing}; got {sorted(df.columns)}"
            )
        return df[keep_cols]

    @staticmethod
    def _uniform_density_track_config(params: dict) -> tuple[dict, dict]:
        """Build (density, track) config sections for a straight stream."""
        width = float(params["width"])
        length = float(params["length"])
        density_cfg = {"type": "Uniform", "xmin": -length / 2.0, "xmax": length / 2.0}
        track_cfg = {
            "center": {"type": "Constant", "value": 0.0},
            "spread": {"type": "Constant", "value": width},
            "sampler": "Gaussian",
        }
        return density_cfg, track_cfg

    @staticmethod
    def _spline_density_track_config(params: dict) -> tuple[dict, dict]:
        """Build (density, track) config sections for a spline stream from
        `params["control_points"]`."""
        control_points = params["control_points"]
        if len(control_points) < 2:
            raise ValueError(
                f"spline morphology needs >=2 control_points, got {len(control_points)}"
            )

        phi1_nodes = np.array([p["phi1"] for p in control_points], dtype=float)
        phi2_nodes = np.array([p["phi2"] for p in control_points], dtype=float)
        order = np.argsort(phi1_nodes)
        phi1_nodes = phi1_nodes[order]
        phi2_nodes = phi2_nodes[order]

        default_width = params.get("width")
        width_nodes = np.array(
            [p.get("width", default_width) for p in control_points], dtype=float
        )[order]
        if np.any(np.isnan(width_nodes)):
            raise ValueError(
                "spline control_points must each have a 'width', or params['width'] "
                "must be set as a fallback for points that omit it"
            )

        density_cfg = {
            "type": "LinearDensityCubicSplineInterpolation",
            "intensity_nodes": phi1_nodes,
            "intensity_node_values": np.ones_like(
                phi1_nodes
            ),  # flat density along track
            "spread_nodes": phi1_nodes,
            "spread_node_values": width_nodes,
        }
        track_cfg = {
            "center": {
                "type": "CubicSplineInterpolation",
                "nodes": phi1_nodes,
                "node_values": phi2_nodes,
            },
            "spread": {
                "type": "CubicSplineInterpolation",
                "nodes": phi1_nodes,
                "node_values": width_nodes,
            },
            "sampler": "Gaussian",
        }
        return density_cfg, track_cfg


class ExternalSimSource(StreamSource):
    """Load pre-realized stream catalogs from data/external_sims/stream_{id}/.

    Minimal required columns: phi1, phi2.
    Optional columns: true magnitudes per band (streamobs convention), age, z (metallicity Z).

    If magnitudes/age/z are missing, fallback: call streamobs isochrone
    sampling using params['age'], params['z'].

    Rationale: Enables use of external stream simulations (e.g., N-body outputs)
    without rewriting them to streamobs format. Designed to slot in seamlessly.

    File schema (PLAN.md §6.1.7, decided 2026-09-08, per user's "avoid file
    proliferation" preference): ONE file per realization, not a directory —
    `data/external_sims/stream_{number}.parquet`. Required columns: `phi1`,
    `phi2`. Optional: true-magnitude columns (`<survey>_<band>_true`) and
    `age`/`z` as constant-valued columns (repeated per row) rather than a
    separate sidecar metadata file.

    Stage 1: Stub (raise NotImplementedError).
    Stage 2+: Full implementation after StreamObsSource is validated.
    """

    def __init__(self):
        """Initialize external sim source."""

    def realize(self, params: dict, rng: np.random.Generator) -> pd.DataFrame:
        """Load pre-realized stream from disk or fallback to streamobs sampling.

        Parameters:
            params: dict with:
                stream_id: "stream_NNN" or similar
                age (optional): Gyr (used if magnitudes missing in realization)
                z (optional): metallicity Z (used if missing in realization)

        Returns:
            DataFrame with phi1, phi2, and (either loaded or sampled) magnitudes.

        Raises:
            FileNotFoundError if stream_id directory not found.
            NotImplementedError (Stage 1 stub).
        """
        raise NotImplementedError(
            "ExternalSimSource is a second-stage feature (decision 14) — not "
            "implemented yet. See the class docstring for the resolved file "
            "schema (data/external_sims/stream_{number}.parquet)."
        )
