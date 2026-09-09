"""Target label rasterization: convert stream members to target maps.

Three label policies (decision 6, §4.6):
1. binary: 1 where stream members fall (optionally dilated), 0 elsewhere.
2. density: raw star-count map or normalized, emphasizes high-concentration regions.
3. soft_distance: per-channel soft weight by |dm_c - dm_true| (stub, scan mode).

binary/density don't depend on distance modulus at all, so rasterize()
produces a single (ny, nx) label image; the caller (injector.py) broadcasts
it identically across every map_stack channel. Only soft_distance genuinely
varies per channel (Stage 2+ stub, distance_moduli-aware by construction).

Rasterization reuses matched_filter.py's own HEALPix->gnomonic pipeline
(make_raw_map + project), NOT a hand-rolled tangent-plane binning, so label
pixels line up exactly with map_stack's pixels. Unlike a data map, a
member-only raw map's own valid_mask means "this HEALPix pixel contains a
stream star" (matched_filter.make_raw_map's own docstring: not meant to be
used downstream) -- a label must be defined everywhere in the window
regardless, so rasterization always projects with an all-valid mask instead.

Window cropping happens here; partial streams beyond minimum overlap are included as-is.

Rationale: Isolate label generation from injection logic. Configure via policy.
Support all policies even if only one (density) is used in first milestone.
"""

import dataclasses
import enum
import logging

import healpy as hp
import numpy as np
import pandas as pd

from streamgoggles.matched_filter import PixelizationSpec, make_raw_map, project
from streamgoggles.windows import Window

logger = logging.getLogger(__name__)


def _dilate_bool(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow a 2D boolean mask by `radius_px` pixels, 4-connectivity
    (iterative OR-shift) -- equivalent to
    `scipy.ndimage.binary_dilation(mask, iterations=radius_px)` with its
    default structuring element (verified to match exactly), but without
    importing scipy.ndimage: its native extension has been observed to crash
    a later `import torch` in the same process in this environment (a
    conflicting-OpenMP-runtime segfault, reproduced running this module's
    tests ahead of tests/test_storage.py's torch-based fixtures), so this
    tiny operation isn't worth pulling that dependency in for.
    """
    out = mask.copy()
    for _ in range(radius_px):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


class LabelPolicy(enum.Enum):
    """Label generation policy."""

    BINARY = "binary"
    DENSITY = "density"
    SOFT_DISTANCE = "soft_distance"


def _project_stream_members(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    window: Window,
    pix: PixelizationSpec,
) -> np.ndarray:
    """Rasterize member positions to a HEALPix count map, then project to
    the window -- same nside/projection as map_stack, all-valid mask (see
    module docstring), and nearest-neighbor (not interpolated) so a
    member's count stays in the one pixel it actually fell in, uncontaminated
    by bilinear blending, before any explicit dilation/smoothing is applied.
    """
    catalog = pd.DataFrame(
        {
            "ra": np.asarray(stream_members_ra, dtype=float),
            "dec": np.asarray(stream_members_dec, dtype=float),
        }
    )
    selected = np.ones(len(catalog), dtype=bool)
    raw_map, _member_valid_mask = make_raw_map(catalog, selected, pix)

    window_pix = dataclasses.replace(
        pix,
        center_ra=window.center_ra,
        center_dec=window.center_dec,
        rotation_deg=window.rotation_deg,
        interpolate=False,
    )
    all_valid = np.ones_like(raw_map, dtype=bool)
    image, _image_valid = project(raw_map, all_valid, window_pix)
    return image


def rasterize(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    stream_width_deg: float,
    params: dict,
    policy: str,
    window: Window,
    pix: PixelizationSpec,
    dilate_to_width: bool = False,
    normalization: str = "max",
    smooth_sigma_deg: float | None = None,
) -> np.ndarray:
    """Rasterize stream members to a 2D label image.

    Converts stream member positions to a target label matching the pixel
    grid map_stack's own channels are projected onto (the caller broadcasts
    this single image across every distance-modulus channel for binary/density
    -- see module docstring).

    Parameters:
        stream_members_ra, stream_members_dec: Arrays of stream star positions (degrees).
        stream_width_deg: Stream width (degrees), used for dilation in binary mode.
        params: Stream parameter dict (must include 'distance_modulus' for
            soft_distance; unused for binary/density).
        policy: Label policy ("binary", "density", or "soft_distance").
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        dilate_to_width: binary policy only -- see rasterize_binary.
        normalization: density policy only -- see rasterize_density.
        smooth_sigma_deg: density policy only -- see rasterize_density.

    Returns:
        label: (ny, nx) float32 array.

        binary policy: 0/1 mask. 1 where stream members rasterized (optionally
            dilated to stream width). 0 elsewhere.

        density policy: per-pixel star count, optionally smoothed and
            normalized. Emphasizes high-density regions (stream core).

    Raises:
        ValueError if policy is not "binary"/"density"/"soft_distance".
        NotImplementedError for soft_distance (stub; implement Stage 2+ --
            it needs distance_moduli, which this function's signature has no
            room for since binary/density don't take it at all; a genuinely
            per-channel policy needs rasterize_soft_distance directly).

    Rationale: Labels are clipped to the window (partial streams beyond the
    ≥5° floor still produce a partial label, which is CORRECT — no special-casing).
    Output pixel grid matches map_stack for direct per-pixel loss computation.
    """
    if policy == LabelPolicy.BINARY.value:
        return rasterize_binary(
            stream_members_ra,
            stream_members_dec,
            stream_width_deg,
            window,
            pix,
            dilate_to_width=dilate_to_width,
        )
    if policy == LabelPolicy.DENSITY.value:
        return rasterize_density(
            stream_members_ra,
            stream_members_dec,
            window,
            pix,
            normalization=normalization,
            smooth_sigma_deg=smooth_sigma_deg,
        )
    if policy == LabelPolicy.SOFT_DISTANCE.value:
        raise NotImplementedError(
            "soft_distance label policy is a Stage 2+ feature (decision 6) -- "
            "not implemented yet. Use rasterize_soft_distance directly once it "
            "is (it needs distance_moduli, which rasterize()'s signature has "
            "no room for)."
        )
    raise ValueError(
        f"Unknown label policy {policy!r}; expected 'binary'/'density'/'soft_distance'"
    )


def rasterize_binary(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    stream_width_deg: float,
    window: Window,
    pix: PixelizationSpec,
    dilate_to_width: bool = False,
) -> np.ndarray:
    """Binary label: 1 where stream members fall, 0 elsewhere.

    Parameters:
        stream_members_ra, stream_members_dec: Stream star positions (degrees).
        stream_width_deg: Stream width (degrees).
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        dilate_to_width: If True, dilate the mask by stream_width_deg / 2
            (isotropic, in pixel units -- a small-window gnomonic
            approximation, decision: no cos(dec)/projection-distortion
            correction, matching the same "just a configurable patch"
            simplicity as StudyRegion.is_inside).

    Returns:
        (ny, nx) binary label (float32, values 0 or 1).
    """
    image = _project_stream_members(stream_members_ra, stream_members_dec, window, pix)
    binary = image > 0.0

    if dilate_to_width and stream_width_deg > 0:
        radius_px = max(1, round((stream_width_deg / 2.0) / pix.pixel_scale_deg))
        binary = _dilate_bool(binary, radius_px)

    return binary.astype(np.float32)


def rasterize_density(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    window: Window,
    pix: PixelizationSpec,
    normalization: str = "max",
    smooth_sigma_deg: float | None = None,
) -> np.ndarray:
    """Density label: raw star count, optionally normalized and smoothed.

    Parameters:
        stream_members_ra, stream_members_dec: Stream star positions (degrees).
        window: Window for cropping.
        pix: PixelizationSpec for projection.
        normalization: "max" (max=1), "sum" (sum=1), or "none" (no normalization).
            A map with no members ("max"/"sum" of a zero image) is left as-is
            (all zeros) rather than dividing by zero.
        smooth_sigma_deg: Gaussian smoothing width (degrees); None/0 disables.
            Applied on the sphere (healpy.smoothing) before projection, same
            kernel convention as matched_filter.py's finalize_full, so a
            density label smoothed with the same sigma as the data map stays
            comparable to it.

    Returns:
        (ny, nx) density label (float32, values in [0, 1] or [0, inf) per norm).

    Raises:
        ValueError if normalization mode not recognized.

    Rationale: Emphasizes high-density stream core. Network learns to predict
    high values where density is high, low values where sparse.
    """
    if normalization not in ("max", "sum", "none"):
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected 'max'/'sum'/'none'"
        )

    catalog = pd.DataFrame(
        {
            "ra": np.asarray(stream_members_ra, dtype=float),
            "dec": np.asarray(stream_members_dec, dtype=float),
        }
    )
    selected = np.ones(len(catalog), dtype=bool)
    raw_map, _member_valid_mask = make_raw_map(catalog, selected, pix)

    if smooth_sigma_deg:
        raw_map = hp.smoothing(raw_map, sigma=np.radians(smooth_sigma_deg))
        raw_map = np.clip(raw_map, 0.0, None)  # smoothing can ring slightly negative

    window_pix = dataclasses.replace(
        pix,
        center_ra=window.center_ra,
        center_dec=window.center_dec,
        rotation_deg=window.rotation_deg,
        interpolate=False,
    )
    all_valid = np.ones_like(raw_map, dtype=bool)
    image, _image_valid = project(raw_map, all_valid, window_pix)

    if normalization == "max":
        peak = image.max()
        if peak > 0:
            image = image / peak
    elif normalization == "sum":
        total = image.sum()
        if total > 0:
            image = image / total

    return image.astype(np.float32)


def rasterize_soft_distance(
    stream_members_ra: np.ndarray,
    stream_members_dec: np.ndarray,
    params: dict,
    distance_moduli: list[float],
    window: Window,
    pix: PixelizationSpec,
    tolerance_mag: float = 0.3,
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
    raise NotImplementedError(
        "rasterize_soft_distance is a Stage 2+ feature (decision 6) -- not "
        "implemented yet; binary/density (rasterize_binary/rasterize_density) "
        "are implemented and validated first."
    )
