"""Window sampling and placement for training samples.

A Window represents one sky field: center (RA, Dec), size, and rotation angle.
Windows are sampled randomly, with or without stream constraints.

Rationale: Decouple window geometry and sampling from stream injection.
Enables testing window constraints independently from stream generation.

Key constraint (decision 21): when a stream is present, rejection-sample windows
until ≥5° of stream length falls inside. Stream may still be cut off beyond that
floor—this is a constraint, not a guarantee of full inclusion.
"""

import dataclasses
import logging
from typing import Callable

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Window:
    """One sky window: center, size, and rotation.
    
    Attributes:
        center_ra: Window center RA (degrees).
        center_dec: Window center Dec (degrees).
        rotation_deg: Position angle / tilt [0, 360) (degrees).
        width_deg: Window width (degrees), typically derived from pixel scale.
        height_deg: Window height (degrees).
    
    Rationale: Bundles window geometry in one place for easy passing
    and consistent projection calculations.
    """
    center_ra: float
    center_dec: float
    rotation_deg: float = 0.0
    width_deg: float = 12.8
    height_deg: float = 12.8


def sample_random_window(
    background_footprint: np.ndarray,
    nside: int,
    size_deg: float = 12.8,
    rng: np.random.Generator | None = None
) -> Window:
    """Sample a random window uniformly within the background footprint.
    
    Parameters:
        background_footprint: HEALPix bool mask (npix,), True for valid pixels.
        nside: HEALPix NSIDE parameter.
        size_deg: Window size (degrees).
        rng: np.random.Generator instance (if None, creates one with default seed).
    
    Returns:
        Window with center inside footprint and random tilt ∈ [0, 360).
    
    Rationale: Simple uniform sampling over the footprint; no stream constraints.
    Used for pure-background negatives (no stream injection).
    """
    raise NotImplementedError


def sample_stream_window(
    stream_ra: np.ndarray,
    stream_dec: np.ndarray,
    stream_width_deg: float,
    background_footprint: np.ndarray,
    nside: int,
    size_deg: float = 12.8,
    min_stream_length_deg: float = 5.0,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None
) -> Window:
    """Rejection-sample a window ensuring ≥min_stream_length_deg of stream inside.
    
    Parameters:
        stream_ra, stream_dec: Arrays of stream star positions (degrees).
        stream_width_deg: Stream width (degrees), used for on-sky length calculations.
        background_footprint: HEALPix bool mask (npix,), True for valid pixels.
        nside: HEALPix NSIDE parameter.
        size_deg: Window size (degrees).
        min_stream_length_deg: Minimum on-sky stream length inside window (degrees).
        max_attempts: Max rejection-sampling attempts before raising.
        rng: np.random.Generator instance (if None, creates one).
    
    Returns:
        Window with center inside footprint, random tilt, and ≥min_stream_length_deg
        of stream inside it.
    
    Raises:
        RuntimeError if max_attempts exceeded without finding valid window.
    
    Rationale: Ensures every training sample with a stream includes a meaningful
    portion of the stream track. The constraint is a FLOOR: stream may still
    legitimately be cut off by the window edge beyond this minimum, and such
    partial-stream samples are INCLUDED in training (no special-casing).
    
    Note: Stream may be shorter than min_stream_length_deg; in that case,
    the entire stream must fit inside the window.
    """
    raise NotImplementedError


def tile_footprint(
    background_footprint: np.ndarray,
    nside: int,
    tile_size_deg: float = 12.8,
    tilt_deg: float = 0.0
) -> list[Window]:
    """Generate regular grid of non-overlapping tiles covering footprint.
    
    Tiles the background footprint with a regular grid (fixed tilt).
    This is for full-survey inference (Phase 2-adjacent), NOT for labeled
    sample generation (which uses random windows per decision 21).
    
    Parameters:
        background_footprint: HEALPix bool mask (npix,), True for valid pixels.
        nside: HEALPix NSIDE parameter.
        tile_size_deg: Tile size (degrees).
        tilt_deg: Fixed position angle for all tiles (degrees).
    
    Returns:
        List of Window objects tiling the footprint.
    
    Rationale: Simple utility for full-field inference. Kept separate and
    clearly labeled to avoid confusion with the per-sample random window
    logic used for labeled training.
    """
    raise NotImplementedError
