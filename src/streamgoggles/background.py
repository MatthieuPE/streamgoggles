"""Background catalog preparation and caching.

Pipeline:
1. Load via BackgroundSource (selects from data_file / light / injection).
2. Apply user cuts (SNR, extendedness, etc.), logging rejection counts.
3. Apply magnitude clipping per band.
4. For each named matched filter, for each trial distance modulus:
   a. select() stars via that filter.
   b. make_raw_map() on HEALPix grid.
   c. Cache raw map + valid_mask via BackgroundMapStore.
   d. If finalization enabled: finalize_full() and cache.
5. Expose Background.footprint as HEALPix mask of valid pixels.

Multiple named matched filters (2026-09-09 pivot): a sample's input is no
longer one map per trial distance, but one per (filter, distance) pair --
typically a real isochrone filter plus one or more deliberately "bad" decoy
filters (matched_filter.ShiftedColorBoxFilter), so a network sees both what
a real overdensity looks like and what generic background contamination
looks like at the same trial distance. Each filter is applied to the
background exactly once here (still cached, still checked before the
catalog itself is loaded), same as the single-filter case before it.

Cut/apply_cuts()/apply_magnitude_clipping() live in data_preparation.py, not
here -- they're applied identically to the background catalog (here) and to
the injected stream catalog (injector.py, decision 13), so they belong
somewhere both can import from rather than in either pipeline-specific module.

All caching goes through BackgroundMapStore; every sample reuses cached results.

Rationale: Background is expensive to compute and reused by all samples.
Caching at the right level (per filter, per distance, per background config)
amortizes cost and enables efficient per-sample injection (background fixed,
only stream varies).
"""

import dataclasses
import logging

import numpy as np
import pandas as pd

from streamgoggles.background_sources import BackgroundSource, StudyRegion
from streamgoggles.data_preparation import (
    Cut,
    apply_cuts,
    apply_magnitude_clipping,
)
from streamgoggles.matched_filter import (
    MatchedFilter,
    PixelizationSpec,
    finalize_full,
    make_raw_map,
)
from streamgoggles.storage import BackgroundMapStore

logger = logging.getLogger(__name__)


def build_raw_background_maps(
    catalog: pd.DataFrame,
    matched_filters: dict[str, MatchedFilter],
    bands: list[str],
    distance_moduli: list[float],
    pix: PixelizationSpec,
) -> dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Build raw HEALPix maps for each (filter, distance modulus) pair.

    Parameters:
        catalog: Background catalog (already cut + clipped).
        matched_filters: dict mapping filter name -> MatchedFilter instance
            (e.g. {"good": StreamobsSplineFilter(...), "decoy":
            ShiftedColorBoxFilter(...)}).
        bands: List of photometric bands.
        distance_moduli: List of trial distance moduli.
        pix: PixelizationSpec.

    Returns:
        dict mapping filter_name -> {distance_modulus -> (raw_map_full,
        valid_mask_full)}. raw_map_full: HEALPix counts (npix,).
        valid_mask_full: HEALPix bool mask (npix,); same for every filter
        and distance (determined by catalog coverage, not by selection).

    Rationale: One raw map per (filter, distance) pair (since selection
    depends on both), but valid_mask is shared (determined by survey
    coverage, not selection).
    """
    result = {}
    for filter_name, matched_filter in matched_filters.items():
        per_distance = {}
        for dm in distance_moduli:
            selected = matched_filter.select(catalog, bands, dm)
            raw_map, valid_mask = make_raw_map(catalog, selected, pix)
            per_distance[float(dm)] = (raw_map, valid_mask)
        result[filter_name] = per_distance
    return result


@dataclasses.dataclass
class Background:
    """Loaded and cached background data.

    Attributes:
        catalog: Full cleaned catalog (after cuts + clipping), or None if
            every requested (filter, distance modulus) pair was already
            cached (so nothing needed loading -- see `load_or_cache`).
        raw_map_full_dict: dict mapping filter_name -> {distance_modulus ->
            raw_map_full (npix,), float}.
        valid_mask_full: HEALPix bool mask (npix,); same for every filter
            and distance.
        finalized_map_full_dict: dict mapping filter_name -> {distance_modulus
            -> finalized_map_full or None (None when finalization is
            disabled)}.
        footprint: HEALPix bool mask (npix,); True for valid pixels (data coverage).
            Same array as valid_mask_full, exposed under its own name for
            callers (windows.py) that only care about coverage, not distance.

    Rationale: Bundles all background data (catalog + cached maps) in one place.
    Maps are (filter, distance)-dependent but valid_mask is shared (determined
    by survey coverage).
    """

    catalog: pd.DataFrame | None
    raw_map_full_dict: dict[str, dict[float, np.ndarray]]
    valid_mask_full: np.ndarray
    finalized_map_full_dict: dict[str, dict[float, np.ndarray | None]] = (
        dataclasses.field(default_factory=dict)
    )
    footprint: np.ndarray | None = None

    @classmethod
    def load_or_cache(
        cls,
        source: BackgroundSource,
        source_cfg: dict,
        study_region: StudyRegion | None,
        cuts: list[Cut],
        clipping: dict | None,
        matched_filters: dict[str, MatchedFilter],
        bands: list[str],
        distance_moduli: list[float],
        finalize_cfg: dict | None,
        store: BackgroundMapStore,
        pix: PixelizationSpec,
        survey: str = "lsst",
        release: str = "dp2",
        filter_configs: dict[str, dict] | None = None,
    ) -> "Background":
        """Load background via source, apply processing, cache maps.

        Parameters:
            source: BackgroundSource instance (data_file, light, or injection).
            source_cfg: Source-specific config (e.g., path for data_file).
            study_region: StudyRegion (used only by light/injection sources).
            cuts: List of Cut rules.
            clipping: Magnitude clipping dict (e.g., {'g': {min, max}, 'r': {min, max}}).
            matched_filters: dict mapping filter name -> MatchedFilter instance.
                Each is applied to the background independently and cached
                independently (2026-09-09 pivot from a single filter).
            bands: List of bands for selection.
            distance_moduli: List of trial distance moduli.
            finalize_cfg: Finalization config (enabled, smoothing, background_subtract).
            store: BackgroundMapStore for caching.
            pix: PixelizationSpec -- only `nside`/`nest` affect a full-sky raw
                map, so only those two fields enter the cache key (not
                window-specific fields like center_ra/rotation_deg, which
                don't apply to a full-sky background map).
            survey, release: Passed to `source.load()` and used to build the
                `<survey>_<release>` column namespace for cuts/clipping/select.
            filter_configs: dict mapping filter name -> serializable dict
                identifying that filter's configuration (e.g. its reference
                isochrone + bands), used only for the cache key -- a
                MatchedFilter object itself isn't reliably
                hashable/serializable across implementations. A filter with
                no entry here simply has `None` folded into its cache key.

        Returns:
            Background instance with catalog + cached maps.

        Raises:
            FileNotFoundError if source path/resource is missing.

        Rationale: The cache key depends only on config (never on the loaded
        catalog itself), so cache existence is checked for every requested
        (filter, distance modulus) pair BEFORE loading anything. When every
        pair is already cached, `source.load()` (which can be expensive -- a
        full synthetic generation, or a large real catalog) is skipped
        entirely; `catalog` is then `None` (see its attribute docstring).
        This is what makes "cache once, reuse per-sample" (module docstring)
        actually efficient rather than just avoiding the map-building work.
        Cache granularity stays per (filter, distance) pair (not per-filter
        all-or-nothing), so adding one new filter/distance to an existing
        config doesn't recompute anything already cached.
        """
        namespace = f"{survey}_{release}" if release else survey
        filter_configs = filter_configs or {}
        base_key_common = {
            "source_type": type(source).__name__,
            "source_cfg": source_cfg,
            "survey": survey,
            "release": release,
            "study_region": dataclasses.asdict(study_region)
            if study_region is not None
            else None,
            "cuts": [dataclasses.asdict(c) for c in cuts],
            "clipping": clipping,
            "nside": pix.nside,
            "nest": pix.nest,
            "finalize_config": finalize_cfg,
        }

        def _base_key_for(filter_name: str) -> dict:
            return {
                **base_key_common,
                "filter_name": filter_name,
                "filter_config": filter_configs.get(filter_name),
            }

        raw_map_full_dict: dict[str, dict[float, np.ndarray]] = {}
        finalized_map_full_dict: dict[str, dict[float, np.ndarray | None]] = {}
        valid_mask_full = None
        to_compute: dict[str, list[float]] = {}

        for filter_name in matched_filters:
            base_key = _base_key_for(filter_name)
            raw_map_full_dict[filter_name] = {}
            finalized_map_full_dict[filter_name] = {}
            missing = []
            for dm in distance_moduli:
                key = {**base_key, "distance_modulus": float(dm)}
                if store.exists(key):
                    raw_map, valid_mask, finalized_map = store.load_background(key)
                    raw_map_full_dict[filter_name][float(dm)] = raw_map
                    finalized_map_full_dict[filter_name][float(dm)] = finalized_map
                    if valid_mask_full is None:
                        valid_mask_full = valid_mask
                else:
                    missing.append(dm)
            if missing:
                to_compute[filter_name] = missing

        catalog = None
        if to_compute:
            catalog = source.load(survey, release, study_region, source_cfg)
            catalog = apply_cuts(catalog, cuts, namespace=namespace)
            if clipping:
                catalog = apply_magnitude_clipping(
                    catalog, clipping, namespace=namespace
                )

            finalize_enabled = (
                bool(finalize_cfg.get("enabled", False)) if finalize_cfg else False
            )
            for filter_name, missing_dms in to_compute.items():
                base_key = _base_key_for(filter_name)
                computed = build_raw_background_maps(
                    catalog,
                    {filter_name: matched_filters[filter_name]},
                    bands,
                    missing_dms,
                    pix,
                )[filter_name]
                for dm, (raw_map, valid_mask) in computed.items():
                    finalized_map = (
                        finalize_full(raw_map, valid_mask, finalize_cfg)
                        if finalize_enabled
                        else None
                    )
                    raw_map_full_dict[filter_name][dm] = raw_map
                    finalized_map_full_dict[filter_name][dm] = finalized_map
                    if valid_mask_full is None:
                        valid_mask_full = valid_mask
                    key = {**base_key, "distance_modulus": dm}
                    store.save_background(raw_map, valid_mask, finalized_map, key)

        return cls(
            catalog=catalog,
            raw_map_full_dict=raw_map_full_dict,
            valid_mask_full=valid_mask_full,
            finalized_map_full_dict=finalized_map_full_dict,
            footprint=valid_mask_full,
        )
