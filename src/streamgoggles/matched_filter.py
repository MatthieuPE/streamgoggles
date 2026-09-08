"""Matched filter interface, HEALPix pixelization, and gnomonic projection.

Core operations:
- select(): apply isochrone color-magnitude selection via streamobs (unweighted)
- make_raw_map(): pixelize selected stars to HEALPix counts
- project(): HEALPix → 2D gnomonic image with healpy interpolation
- combine_full_maps(): add background + stream raw maps
- finalize_full(): apply smoothing + polynomial background subtraction
- crop_window(): crop finalized full map to one window

All operations use explicit valid_mask throughout (decision 22).

Rationale: Isolate pixelization, projection, and background-model logic from 
pipeline flow. Confirm additivity explicitly (background + stream raw maps).
Keep model code (smoothing/fitting functions) wrapped but reusable.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Protocol

import healpy as hp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PixelizationSpec:
    """HEALPix + gnomonic projection parameters.
    
    Attributes:
        nside: HEALPix NSIDE parameter (power of 2; default 128).
        center_ra: Window center RA (degrees).
        center_dec: Window center Dec (degrees).
        rotation_deg: Position angle / tilt of window [0, 360) (degrees).
        image_size_pix: (height, width) in pixels.
        pixel_scale_deg: pixel scale (degrees/pixel).
        projection: projection type ("gnomonic" only for now).
        interpolate: whether to use healpy interpolation when projecting.
        nest: HEALPix ordering scheme (False = RING, True = NESTED).
    
    Rationale: Bundles all pixelization/projection parameters in one place
    for easy passing through the pipeline.
    """
    nside: int = 128
    center_ra: float = 0.0
    center_dec: float = 0.0
    rotation_deg: float = 0.0
    image_size_pix: tuple[int, int] = (256, 256)
    pixel_scale_deg: float = 0.05
    projection: str = "gnomonic"
    interpolate: bool = True
    nest: bool = False
    
    def __post_init__(self):
        """Validate parameters."""
        if not hp.isnsideok(self.nside):
            raise ValueError(f"Invalid nside={self.nside}. Must be a power of 2.")


class MatchedFilter(Protocol):
    """Protocol for isochrone-based matched filtering.
    
    Implementations wrap different backends (streamobs, external, etc.).
    The key operation is select(): a hard boolean cut in color-magnitude space.
    
    Rationale: Protocol enables pluggable filter implementations without
    coupling to any particular backend (streamobs, ugali, etc.).
    """
    
    def select(
        self,
        catalog: pd.DataFrame,
        bands: list[str],
        distance_modulus: float
    ) -> np.ndarray:
        """Apply isochrone color-magnitude selection.
        
        Parameters:
            catalog: DataFrame with magnitudes/errors in streamobs convention
                (e.g., 'lsst_g_obs', 'lsst_r_obs', 'lsst_g_err', 'lsst_r_err').
            bands: List of bands to use in selection (e.g., ['g', 'r']).
            distance_modulus: Trial distance modulus (magnitude units).
        
        Returns:
            Boolean array, shape (len(catalog),); True for passed stars.
        
        Rationale: Unweighted selection (hard cut) ensures raw-map additivity.
        """


class StreamobsSplineFilter(MatchedFilter):
    """Wrap streamobs's spline-based matched filter.
    
    The matched filter is the one implemented inside streamobs using 
    spline-based isochrone selection, NOT a hand-written cut function.
    
    Attributes:
        iso_config: dict with age (Gyr) and z (metallicity, mass fraction)
            for the reference isochrone.
    
    Rationale: streamobs's spline-based filter is the canonical implementation.
    Wrapping it here allows reuse without importing streamobs throughout
    the pipeline.
    """
    
    def __init__(self, iso_config: dict):
        """Initialize filter with reference isochrone.
        
        Parameters:
            iso_config: dict with keys age, z (or deserialize from config).
        """
        raise NotImplementedError
    
    def select(
        self,
        catalog: pd.DataFrame,
        bands: list[str],
        distance_modulus: float
    ) -> np.ndarray:
        """Apply streamobs spline-based selection.
        
        Wraps streamobs's selection function; does not change numerics.
        
        Parameters:
            catalog: DataFrame (must be in streamobs convention).
            bands: List of photometric bands.
            distance_modulus: Trial distance modulus.
        
        Returns:
            Boolean selection mask.
        """
        raise NotImplementedError


class PlaceholderFilter(MatchedFilter):
    """Trivial color-magnitude box selection for testing without streamobs.
    
    Implements a simple box cut in (g-r, r) space; useful for unit tests
    that need a filter but don't want streamobs dependencies.
    
    Rationale: Enables full pipeline testing without real isochrone data.
    """
    
    def __init__(self):
        """Initialize placeholder filter."""
        pass
    
    def select(
        self,
        catalog: pd.DataFrame,
        bands: list[str],
        distance_modulus: float
    ) -> np.ndarray:
        """Apply trivial color-magnitude box cut.
        
        Parameters:
            catalog: DataFrame.
            bands: List of bands (must include 'g' and 'r').
            distance_modulus: Trial distance modulus (used to shift absolute magnitudes).
        
        Returns:
            Boolean mask (all True for now; customize as needed for tests).
        """
        raise NotImplementedError


def make_raw_map(
    catalog: pd.DataFrame,
    selected: np.ndarray,
    pix: PixelizationSpec,
    ra_col: str = "ra",
    dec_col: str = "dec"
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw HEALPix count map and valid_mask from selected stars.
    
    Parameters:
        catalog: DataFrame with 'ra' and 'dec' columns (degrees).
        selected: Boolean array, shape (len(catalog),), True for stars to count.
        pix: PixelizationSpec with nside.
        ra_col, dec_col: Column names for RA/Dec (default 'ra', 'dec').
    
    Returns:
        Tuple (raw_map, valid_mask):
            raw_map: HEALPix count map, shape (npix,), float. Pixels with counts
                are 0.0 or positive; invalid pixels (outside footprint) are NaN.
            valid_mask: HEALPix bool mask, shape (npix,). True where data exists
                (survey coverage), False where invalid (outside footprint).
    
    Rationale: Explicit valid_mask (decision 22) distinguishes "zero counts" from
    "no data". Non-NaN but zero counts in raw_map indicate valid pixels with no
    stars; NaN would indicate invalid pixels (outside footprint). This separation
    is crucial for finalization (smoothing, polynomial fit) to exclude only invalid
    pixels, not all-zero pixels.
    """
    raise NotImplementedError


def project(
    raw_map: np.ndarray,
    valid_mask: np.ndarray,
    pix: PixelizationSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Project HEALPix map to 2D gnomonic image.
    
    Performs HEALPix → 2D gnomonic (tangent-plane) projection centered
    on the window, with optional healpy interpolation. Handles rotation
    (position angle) as part of the tangent-plane projection.
    
    Parameters:
        raw_map: HEALPix map, shape (npix,), float.
        valid_mask: HEALPix bool mask, shape (npix,).
        pix: PixelizationSpec with center, rotation, image_size_pix, pixel_scale_deg.
    
    Returns:
        Tuple (image, image_valid_mask):
            image: 2D projected image, shape pix.image_size_pix, float.
                Pixels outside valid area are filled with 0.0 (but marked invalid).
            image_valid_mask: 2D bool mask, shape pix.image_size_pix.
                True where valid (inside footprint), False outside.
    
    Rationale: Gnomonic projection handles projection effects (RA stretching at
    high Dec) correctly. Rotation is applied as part of tangent-plane projection,
    not post-hoc image rotation (which would need separate interpolation).
    Interpolation (if enabled) propagates both data and validity through projection.
    """
    raise NotImplementedError


def combine_full_maps(
    background_raw_full: np.ndarray,
    stream_raw_full: np.ndarray,
    valid_mask_full: np.ndarray
) -> np.ndarray:
    """Sum background and stream raw HEALPix maps.
    
    Both maps are full-sky HEALPix (npix,); stream_raw_full is mostly zeros
    outside the stream's footprint. Result is combined full-sky map.
    
    Parameters:
        background_raw_full: HEALPix count map (npix,), float.
        stream_raw_full: HEALPix count map (npix,), float.
        valid_mask_full: HEALPix bool mask (npix,).
    
    Returns:
        combined_raw_full: Sum of background and stream, shape (npix,), float.
            Valid pixels are sums; invalid pixels are NaN (from valid_mask).
    
    Rationale: Additive combination is key to caching the finalized background
    once and reusing it across samples (§4.3b shortcut):
    finalize_full(background + stream) == finalize_full(background) + finalize_full(stream)
    if finalize_full is linear (fixed-kernel smoothing + fixed-basis polyfit).
    """
    raise NotImplementedError


def finalize_full(
    combined_raw_full: np.ndarray,
    valid_mask_full: np.ndarray,
    cfg: dict | None = None
) -> np.ndarray:
    """Apply smoothing and polynomial background subtraction to full map.
    
    Operates ONLY on valid_mask==True pixels. Invalid pixels (outside footprint)
    are excluded from fitting and remain NaN in output.
    
    When cfg is None or cfg['enabled'] is False, returns combined_raw_full unchanged
    (identity operation).
    
    Parameters:
        combined_raw_full: HEALPix count map (npix,), float. Already contains
            background + stream combined.
        valid_mask_full: HEALPix bool mask (npix,).
        cfg: dict with keys:
            enabled (bool): if False, skip finalization (default False).
            smoothing_deg (float): Gaussian smoothing width in degrees (0 = no smoothing).
            background_subtract (bool): if True, fit and subtract polynomial background.
    
    Returns:
        finalized_map_full: Smoothed and (optionally) background-subtracted map,
            shape (npix,), float. Invalid pixels remain NaN.
    
    Rationale: This wraps the user's existing smoothing and background-subtraction
    functions. It is designed to be LINEAR (fixed-kernel convolution, fixed-basis
    polynomial fit), so the shortcut is safe: finalize(bg+stream) == finalize(bg) + finalize(stream).
    Before the shortcut is enabled, a dedicated test (§7/§4.3b) must verify
    finalize(bg+stream) == finalize(bg) + finalize_local(stream).
    Must use mask-aware convolution (astropy.convolution, not scipy) to avoid
    NaN contamination from neighbors.
    
    Raises:
        ValueError if cfg specifies unsupported operations.
    """
    raise NotImplementedError


def crop_window(
    finalized_full: np.ndarray,
    valid_mask_full: np.ndarray,
    window: "Window",
    pix: PixelizationSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Project and crop finalized full map to one window.
    
    Takes an already-finalized (smoothed/background-subtracted) full-sky map
    and crops it to one window via projection.
    
    Parameters:
        finalized_full: Full-sky HEALPix map (npix,), float.
        valid_mask_full: Full-sky HEALPix bool mask (npix,).
        window: Window instance with center, rotation, size.
        pix: PixelizationSpec for projection.
    
    Returns:
        Tuple (windowed_image, windowed_valid_mask):
            windowed_image: 2D image, shape pix.image_size_pix, float.
            windowed_valid_mask: 2D bool mask, same shape.
    
    Rationale: Purely a crop/projection operation; no fitting happens here.
    Calls project() internally.
    """
    raise NotImplementedError
