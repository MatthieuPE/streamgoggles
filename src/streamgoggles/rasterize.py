"""Target label rasterization: convert stream members to target maps.

Three label policies (decision 6, §4.6):
1. binary: 1 where stream members fall (optionally dilated), 0 elsewhere.
2. density: raw star-count map or normalized, emphasizes high-concentration regions.
3. soft_distance: per-channel soft weight by |dm_c - dm_true| (stub, scan mode).

Each policy produces (n_dist, ny, nx) label stack matching map_stack.
Window cropping happens here; partial streams beyond minimum overlap are included as-is.

Rationale: Isolate label generation from injection logic. Configure via policy.
Support all policies even if only one (density) is used in first milestone.
"""

import enum
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelPolicy(enum.Enum):
    """Label generation policy."""
    BINARY = "binary"
    DENSITY = "density"
    SOFT_DISTANCE = "soft_distance"


def rasterize(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    stream_width_deg: float,
    params: dict,
    policy: str,
    window: "Window",
    pix: "PixelizationSpec"
) -> np.ndarray:
    """Rasterize stream members to 2D label map stack.
    
    Converts stream member positions to per-channel target labels matching
    the shape and projection of the map_stack.
    
    Parameters:
        stream_members_ra, stream_members_dec: Arrays of stream star positions (degrees).
        stream_width_deg: Stream width (degrees), used for dilation in binary mode.
        params: Stream parameter dict (may include distance_modulus for soft_distance).
        policy: Label policy ("binary", "density", or "soft_distance").
        window: Window for cropping.
        pix: PixelizationSpec for projection.
    
    Returns:
        label_stack: (n_dist, ny, nx) float32 array.
        
        binary policy: 0/1 mask. 1 where stream members rasterized (optionally
            dilated to stream width). 0 elsewhere inside valid area.
        
        density policy: per-pixel star count or normalized. Emphasizes high-density
            regions (stream core). Optionally smoothed with same kernel as filter.
        
        soft_distance policy (stub): per-channel weight f(|dm_c - dm_true|),
            Gaussian decay with configurable tolerance_mag. Zeros out channels
            far from true distance modulus.
    
    Raises:
        ValueError if policy is not supported or params missing required keys.
        NotImplementedError for soft_distance (stub; implement Stage 2+).
    
    Rationale: Labels are clipped to the window (partial streams beyond the
    ≥5° floor still produce a partial label, which is CORRECT — no special-casing).
    Output shape matches map_stack for direct per-pixel loss computation.
    """
    raise NotImplementedError


def rasterize_binary(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    stream_width_deg: float,
    window: "Window",
    pix: "PixelizationSpec",
    dilate_to_width: bool = False
) -> np.ndarray:
    """Binary label: 1 where stream members fall, 0 elsewhere.
    
    Parameters:
        stream_members_ra, stream_members_dec: Stream star positions (degrees).
        stream_width_deg: Stream width (degrees).
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        dilate_to_width: If True, dilate mask to stream_width_deg around members.
    
    Returns:
        (n_dist, ny, nx) binary label (float32, values 0 or 1).
    
    Raises:
        ValueError if inputs are incompatible.
    """
    raise NotImplementedError


def rasterize_density(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    window: "Window",
    pix: "PixelizationSpec",
    normalization: str = "max",
    smooth_sigma_deg: float | None = None
) -> np.ndarray:
    """Density label: raw star count, optionally normalized and smoothed.
    
    Parameters:
        stream_members_ra, stream_members_dec: Stream star positions (degrees).
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        normalization: "max" (max=1), "sum" (sum=1), or "none" (no normalization).
        smooth_sigma_deg: Gaussian smoothing width (degrees); None disables.
    
    Returns:
        (n_dist, ny, nx) density label (float32, values in [0, 1] or [0, ∞) per norm).
    
    Raises:
        ValueError if normalization mode not recognized.
    
    Rationale: Emphasizes high-density stream core. Network learns to predict
    high values where density is high, low values where sparse.
    Normalization can be per-sample (sum=1) or per-pixel (max=1).
    Smoothing uses same kernel as matched filter for consistency.
    """
    raise NotImplementedError


def rasterize_soft_distance(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    params: dict,
    distance_moduli: list[float],
    window: "Window",
    pix: "PixelizationSpec",
    tolerance_mag: float = 0.3
) -> np.ndarray:
    """Soft-distance label: per-channel weight by |dm_c - dm_true|.
    
    Stub for later scan-mode support. Each channel c gets weight
    w_c = exp(-(dm_c - dm_true)^2 / (2 * sigma^2)), where sigma
    is inferred from tolerance_mag.
    
    Parameters:
        stream_members_ra, stream_members_dec: Stream star positions (degrees).
        params: Must include 'distance_modulus' (true distance).
        distance_moduli: List of trial distance moduli.
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        tolerance_mag: RMS distance tolerance (mag units) for Gaussian width.
    
    Returns:
        (n_dist, ny, nx) soft-distance label (float32, values soft-weighted).
    
    Raises:
        NotImplementedError (stub; implement after binary/density validated).
    """
    raise NotImplementedError
