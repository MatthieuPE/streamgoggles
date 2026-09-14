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

import healpy as hp
import numpy as np

from streamgoggles.matched_filter import world_to_tangent_plane

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

    def __post_init__(self):
        """Validate size and normalize rotation into [0, 360)."""
        if self.width_deg <= 0 or self.height_deg <= 0:
            raise ValueError(
                f"width_deg/height_deg must be positive, got "
                f"{self.width_deg}/{self.height_deg}"
            )
        self.rotation_deg = float(self.rotation_deg) % 360.0

    def contains(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """Boolean mask: which (ra, dec) points fall inside this window.

        Parameters:
            ra_deg, dec_deg: Sky positions (degrees), same shape.

        Returns:
            Boolean array, same shape as the input.

        Rationale: Reuses `world_to_tangent_plane` (matched_filter.py's exact
        inverse of the deprojection `project()`/`crop_window()` use), so
        "inside the window" means the same thing here as it does when the
        window is actually projected/cropped.
        """
        xi_deg, eta_deg = world_to_tangent_plane(
            ra_deg, dec_deg, self.center_ra, self.center_dec, self.rotation_deg
        )
        return (np.abs(xi_deg) <= self.width_deg / 2.0) & (
            np.abs(eta_deg) <= self.height_deg / 2.0
        )


def sample_random_window(
    background_footprint: np.ndarray,
    nside: int,
    size_deg: float = 12.8,
    rng: np.random.Generator | None = None,
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

    Note: the center is snapped to a uniformly-chosen valid HEALPix pixel's
    own center (no additional sub-pixel jitter) — HEALPix pixels are
    equal-area, so this is an unbiased sample of the footprint at the
    footprint's own resolution, which is far finer than a typical window.
    """
    rng = rng if rng is not None else np.random.default_rng()

    valid_pixels = np.flatnonzero(background_footprint)
    if valid_pixels.size == 0:
        raise ValueError("background_footprint has no valid pixels to sample from")

    pixel = rng.choice(valid_pixels)
    center_ra, center_dec = hp.pix2ang(nside, int(pixel), lonlat=True)
    rotation_deg = float(rng.uniform(0.0, 360.0))

    return Window(
        center_ra=float(center_ra),
        center_dec=float(center_dec),
        rotation_deg=rotation_deg,
        width_deg=size_deg,
        height_deg=size_deg,
    )


def _stream_arc_length_coord(
    stream_ra: np.ndarray, stream_dec: np.ndarray
) -> np.ndarray:
    """Approximate arc-length coordinate along the stream, in degrees.

    Projects the stars onto a local tangent plane centered on the stream's
    own centroid, then onto their principal axis (via SVD) — the axis of
    greatest spread, which for a reasonably straight/smoothly-curving stream
    approximates its track direction. Used only to measure "how much of the
    stream's on-sky length falls inside a candidate window," not as a
    physical coordinate.
    """
    center_ra = float(np.mean(stream_ra))
    center_dec = float(np.mean(stream_dec))
    xi, eta = world_to_tangent_plane(stream_ra, stream_dec, center_ra, center_dec, 0.0)
    coords = np.column_stack([xi, eta])
    coords = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    principal_axis = vt[0]
    return coords @ principal_axis


def sample_stream_window(
    stream_ra: np.ndarray,
    stream_dec: np.ndarray,
    stream_width_deg: float,
    background_footprint: np.ndarray,
    nside: int,
    size_deg: float = 12.8,
    min_stream_length_deg: float = 5.0,
    max_attempts: int = 100,
    rng: np.random.Generator | None = None,
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
        ValueError if stream_ra/stream_dec are empty.
        RuntimeError if max_attempts exceeded without finding valid window.

    Rationale: Ensures every training sample with a stream includes a meaningful
    portion of the stream track. The constraint is a FLOOR: stream may still
    legitimately be cut off by the window edge beyond this minimum, and such
    partial-stream samples are INCLUDED in training (no special-casing).

    Note: Stream may be shorter than min_stream_length_deg; in that case,
    the entire stream must fit inside the window.

    Candidate (center, tilt) pairs are drawn by jittering around a randomly
    chosen stream star, within roughly one window size — a random center
    drawn from the whole footprint would rarely land near the (typically
    much smaller) stream, making rejection sampling very inefficient. This
    is purely a sampling-efficiency choice, not a semantic constraint (the
    prompt's own §4.4b calls this out as an implementation detail): the
    accepted window is still just "a random center and a random tilt"
    satisfying the overlap floor below.
    """
    rng = rng if rng is not None else np.random.default_rng()

    stream_ra = np.asarray(stream_ra, dtype=float)
    stream_dec = np.asarray(stream_dec, dtype=float)
    if stream_ra.size == 0:
        raise ValueError("stream_ra/stream_dec must be non-empty")

    arc_length_coord = _stream_arc_length_coord(stream_ra, stream_dec)
    total_length_deg = (
        float(arc_length_coord.max() - arc_length_coord.min())
        if stream_ra.size > 1
        else 0.0
    )
    required_length_deg = min(min_stream_length_deg, total_length_deg)

    for _ in range(max_attempts):
        anchor_idx = rng.integers(0, stream_ra.size)
        anchor_ra = stream_ra[anchor_idx]
        anchor_dec = stream_dec[anchor_idx]

        cos_dec = max(np.cos(np.radians(anchor_dec)), 1e-6)
        d_ra = rng.uniform(-size_deg, size_deg) / cos_dec
        d_dec = rng.uniform(-size_deg, size_deg)
        center_ra = float((anchor_ra + d_ra) % 360.0)
        center_dec = float(np.clip(anchor_dec + d_dec, -90.0, 90.0))
        rotation_deg = float(rng.uniform(0.0, 360.0))

        candidate = Window(
            center_ra=center_ra,
            center_dec=center_dec,
            rotation_deg=rotation_deg,
            width_deg=size_deg,
            height_deg=size_deg,
        )

        inside = candidate.contains(stream_ra, stream_dec)
        if not inside.any():
            continue

        covered_length_deg = float(
            arc_length_coord[inside].max() - arc_length_coord[inside].min()
        )
        if covered_length_deg < required_length_deg - 1e-9:
            continue

        pixel = hp.ang2pix(
            nside, candidate.center_ra, candidate.center_dec, lonlat=True
        )
        if not background_footprint[pixel]:
            continue

        return candidate

    raise RuntimeError(
        f"sample_stream_window: no valid window found after {max_attempts} attempts "
        f"(required >= {required_length_deg:.3f} deg of stream inside a "
        f"{size_deg} deg window)."
    )


def tile_footprint(
    background_footprint: np.ndarray,
    nside: int,
    tile_size_deg: float = 12.8,
    tilt_deg: float = 0.0,
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

    Known limitation: the RA grid is built from a simple min/max range per
    declination band, so a footprint that wraps across RA=0/360 will not be
    tiled correctly — not handled here, matching the same documented
    limitation as the study-region centroid (PLAN.md).
    """
    valid_pixels = np.flatnonzero(background_footprint)
    if valid_pixels.size == 0:
        return []

    ra, dec = hp.pix2ang(nside, valid_pixels, lonlat=True)
    dec_min, dec_max = float(dec.min()), float(dec.max())

    windows = []
    dec_center = dec_min
    while dec_center <= dec_max + 1e-9:
        band_mask = np.abs(dec - dec_center) <= tile_size_deg / 2.0
        if band_mask.any():
            cos_dec = max(np.cos(np.radians(dec_center)), 1e-6)
            ra_step = tile_size_deg / cos_dec
            ra_band = ra[band_mask]
            ra_min, ra_max = float(ra_band.min()), float(ra_band.max())

            ra_center = ra_min
            while ra_center <= ra_max + 1e-9:
                pixel = hp.ang2pix(nside, ra_center % 360.0, dec_center, lonlat=True)
                if background_footprint[pixel]:
                    windows.append(
                        Window(
                            center_ra=float(ra_center % 360.0),
                            center_dec=dec_center,
                            rotation_deg=tilt_deg,
                            width_deg=tile_size_deg,
                            height_deg=tile_size_deg,
                        )
                    )
                ra_center += ra_step

        dec_center += tile_size_deg

    return windows
