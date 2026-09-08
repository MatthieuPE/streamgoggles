"""Background catalog preparation and caching.

Pipeline:
1. Load via BackgroundSource (selects from data_file / light / catalogue).
2. Apply user cuts (SNR, extendedness, etc.), logging rejection counts.
3. Apply magnitude clipping per band.
4. For each trial distance modulus:
   a. select() stars via matched filter.
   b. make_raw_map() on HEALPix grid.
   c. Cache raw map + valid_mask via BackgroundMapStore.
   d. If finalization enabled: finalize_full() and cache (or build cached FinalizedBackground).
5. Expose Background.footprint as HEALPix mask of valid pixels.

All caching goes through BackgroundMapStore; every sample reuses cached results.

Rationale: Background is expensive to compute and reused by all samples.
Caching at the right level (per distance, per background config) amortizes cost
and enables efficient per-sample injection (background fixed, only stream varies).
"""

import dataclasses
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Cut:
    """One quality cut rule.
    
    Attributes:
        quantity: column name or special name (e.g., "snr", "mag", "extendedness").
        band: photometric band (e.g., "g", "r"); may be None if quantity doesn't apply.
        op: comparison operator as string (">"、"<"、"=="、"!="、">="、"<=").
        value: comparison value.
        
    Alternative: callable-based cut.
        callable: "module:function_name" string pointing to external cut function.
    
    Rationale: Encapsulate a single cut rule; chain multiple rules in order.
    """
    quantity: str | None = None
    band: str | None = None
    op: str | None = None
    value: float | int | None = None
    callable: str | None = None  # "module:function"
    
    def apply(self, df: pd.DataFrame) -> np.ndarray:
        """Apply this cut to a DataFrame.
        
        Parameters:
            df: Input DataFrame.
        
        Returns:
            Boolean array, True for rows passing the cut.
        
        Raises:
            ValueError if cut is malformed or column missing.
        """
        raise NotImplementedError


def apply_cuts(
    df: pd.DataFrame,
    cuts: list[Cut],
    verbose: bool = True
) -> pd.DataFrame:
    """Apply all cuts in order; log rejection counts.
    
    Parameters:
        df: Input DataFrame.
        cuts: List of Cut instances.
        verbose: If True, log cumulative rejection counts.
    
    Returns:
        Filtered DataFrame (rows passing all cuts).
    
    Rationale: Sequential application allows cuts to depend on previous cuts
    and makes logging clear ("After SNR cut: 50K → 45K rows; extendedness: 45K → 40K").
    """
    raise NotImplementedError


def apply_magnitude_clipping(
    df: pd.DataFrame,
    clipping: dict
) -> pd.DataFrame:
    """Apply per-band magnitude clipping.
    
    Parameters:
        df: DataFrame with magnitudes in streamobs convention (e.g., 'lsst_g_obs', 'lsst_r_obs').
        clipping: dict mapping band -> {min, max} mag bounds, e.g.,
            {'g': {min: 16, max: 26.5}, 'r': {min: 16, max: 26}}.
    
    Returns:
        Clipped DataFrame.
    
    Rationale: Separate from cuts for clarity; clipping is uniform per band,
    not a quality assessment.
    """
    raise NotImplementedError


@dataclasses.dataclass
class Background:
    """Loaded and cached background data.
    
    Attributes:
        catalog: Full cleaned catalog (after cuts + clipping).
        raw_map_full_dict: dict mapping distance_modulus -> raw_map_full (npix,), float.
        valid_mask_full: HEALPix bool mask (npix,); same for all distances.
        finalized_map_full_dict: dict mapping distance_modulus -> finalized_map_full or None.
        footprint: HEALPix bool mask (npix,); True for valid coverage.
    
    Rationale: Bundles all background data (catalog + cached maps) in one place.
    Maps are distance-dependent but valid_mask is shared (determined by survey coverage).
    """
    catalog: pd.DataFrame
    raw_map_full_dict: dict[float, np.ndarray]
    valid_mask_full: np.ndarray
    finalized_map_full_dict: dict[float, np.ndarray | None] = dataclasses.field(
        default_factory=dict
    )
    footprint: np.ndarray | None = None
    
    @classmethod
    def load_or_cache(
        cls,
        source: "BackgroundSource",
        source_cfg: dict,
        study_region: "StudyRegion | None",
        cuts: list[Cut],
        clipping: dict | None,
        matched_filter: "MatchedFilter",
        bands: list[str],
        distance_moduli: list[float],
        finalize_cfg: dict | None,
        store: "BackgroundMapStore"
    ) -> "Background":
        """Load background via source, apply processing, cache maps.
        
        Parameters:
            source: BackgroundSource instance (data_file, light, or catalogue).
            source_cfg: Source-specific config (e.g., path for data_file).
            study_region: StudyRegion (used only by light/catalogue).
            cuts: List of Cut rules.
            clipping: Magnitude clipping dict (e.g., {'g': {min, max}, 'r': {min, max}}).
            matched_filter: MatchedFilter instance.
            bands: List of bands for selection.
            distance_moduli: List of trial distance moduli.
            finalize_cfg: Finalization config (enabled, smoothing, background_subtract).
            store: BackgroundMapStore for caching.
        
        Returns:
            Background instance with catalog + cached maps.
        
        Raises:
            FileNotFoundError if source path missing or data can't be loaded.
        """
        raise NotImplementedError


def build_raw_background_maps(
    catalog: pd.DataFrame,
    matched_filter: "MatchedFilter",
    bands: list[str],
    distance_moduli: list[float],
    pix: "PixelizationSpec"
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Build raw HEALPix maps for each distance modulus.
    
    Parameters:
        catalog: Background catalog (already cut + clipped).
        matched_filter: MatchedFilter instance.
        bands: List of photometric bands.
        distance_moduli: List of trial distance moduli.
        pix: PixelizationSpec.
    
    Returns:
        dict mapping distance_modulus -> (raw_map_full, valid_mask_full).
        raw_map_full: HEALPix counts (npix,).
        valid_mask_full: HEALPix bool mask (npix,); same for all distances.
    
    Rationale: One raw map per distance (since selection depends on distance modulus),
    but valid_mask is shared (determined by survey coverage, not distance).
    """
    raise NotImplementedError
