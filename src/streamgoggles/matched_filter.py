"""Matched filter interface, HEALPix pixelization, and gnomonic projection.

Core operations:
- select(): apply isochrone color-magnitude selection via streamobs (unweighted)
- make_raw_map(): pixelize selected stars to HEALPix counts
- project(): HEALPix → 2D gnomonic image with healpy interpolation
- combine_full_maps(): add background + stream raw maps
- finalize_full(): apply smoothing + polynomial background subtraction
- crop_window(): crop finalized full map to one window

All operations use explicit valid_mask throughout (decision 22).

Column convention: select() reads magnitude columns via
`streamobs.columns.obs_col(band, namespace)`, i.e. ``<namespace>_<band>_obs``
(e.g. "lsst_yr1_g_obs") when a namespace is set, or bare ``<band>_obs`` when
it isn't — this is streamobs's own real convention (confirmed against
streamobs/tests/test_match_filter.py, which calls
`is_in_match_filter(detected["lsst_yr4_g_obs"], detected["lsst_yr4_r_obs"], ...)`
directly), not an invented one. A stream catalog fresh out of StreamInjector
already has these namespaced columns; background.py's harmonize_columns()
(decision 13) must rename the real DP2 skim's columns into this exact same
`<namespace>_<band>_obs` shape so the same select() call works identically on
both without branching.

Rationale: Isolate pixelization, projection, and background-model logic from
pipeline flow. Confirm additivity explicitly (background + stream raw maps).
Keep model code (smoothing/fitting functions) wrapped but reusable.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING, Protocol

import healpy as hp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from streamgoggles.windows import Window

logger = logging.getLogger(__name__)

# Below this factor, pixel_scale_deg is considered to be oversampling the
# HEALPix map — project()/crop_window() would then be bilinearly
# interpolating detail the map doesn't actually contain.
_OVERSAMPLING_ERROR_FACTOR = 0.5


def native_pixel_scale_deg(nside: int) -> float:
    """Mean HEALPix pixel size at this nside, in degrees.

    This is the map's actual angular resolution — `project()`'s output
    `pixel_scale_deg` should be chosen close to this value, not much finer
    than it, or the projected image implies detail the HEALPix map was never
    sampled at.
    """
    return hp.nside2resol(nside, arcmin=True) / 60.0


@dataclasses.dataclass
class PixelizationSpec:
    """HEALPix + gnomonic projection parameters.

    Attributes:
        nside: HEALPix NSIDE parameter (power of 2; default 128).
        center_ra: Window center RA (degrees).
        center_dec: Window center Dec (degrees).
        rotation_deg: Position angle / tilt of window [0, 360) (degrees).
        image_size_pix: (height, width) in pixels.
        pixel_scale_deg: pixel scale (degrees/pixel). Optional — when omitted
            (None), it is auto-derived from `nside` via
            `native_pixel_scale_deg(nside)`. When given explicitly, it must
            not be much finer than that native resolution (see
            `__post_init__`) — pass a value only to deliberately use a
            *coarser*-than-native pixel size.
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
    pixel_scale_deg: float | None = None
    projection: str = "gnomonic"
    interpolate: bool = True
    nest: bool = False

    def __post_init__(self):
        """Validate parameters, and auto-derive pixel_scale_deg if omitted.

        Power-of-2 is only actually required for NESTED ordering (healpy's
        own `isnsideok` rule) — pass `nest` through so this doesn't reject
        valid RING-ordering nsides.
        """
        if not hp.isnsideok(self.nside, nest=self.nest):
            raise ValueError(
                f"Invalid nside={self.nside} for nest={self.nest}."
                + (" Must be a power of 2 when nest=True." if self.nest else "")
            )

        native = native_pixel_scale_deg(self.nside)
        if self.pixel_scale_deg is None:
            self.pixel_scale_deg = native
        elif self.pixel_scale_deg < _OVERSAMPLING_ERROR_FACTOR * native:
            raise ValueError(
                f"pixel_scale_deg={self.pixel_scale_deg:.4g} deg is more than "
                f"{1 / _OVERSAMPLING_ERROR_FACTOR:g}x finer than nside={self.nside}'s "
                f"native HEALPix resolution ({native:.4g} deg/pixel). The projected "
                "image would imply more angular detail than the map was ever "
                "sampled at. Omit pixel_scale_deg to auto-derive it from nside, "
                "or pass a coarser value, or increase nside."
            )


class MatchedFilter(Protocol):
    """Protocol for isochrone-based matched filtering.

    Implementations wrap different backends (streamobs, external, etc.).
    The key operation is select(): a hard boolean cut in color-magnitude space.

    Rationale: Protocol enables pluggable filter implementations without
    coupling to any particular backend (streamobs, ugali, etc.).
    """

    def select(
        self, catalog: pd.DataFrame, bands: list[str], distance_modulus: float
    ) -> np.ndarray:
        """Apply isochrone color-magnitude selection.

        Parameters:
            catalog: DataFrame with magnitudes in streamobs's `<namespace>_<band>_obs`
                convention (module docstring) — e.g. 'lsst_yr1_g_obs'.
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

    Verified against streamobs/streamobs/match_filter.py (PLAN.md §6.1.1):
    ``build_match_filter`` fits two `scipy.interpolate.interp1d` boundary
    splines (blue/red edges) to the isochrone locus as a function of apparent
    magnitude, then samples them onto a polygon; ``is_in_match_filter`` tests
    catalog points against that polygon via `matplotlib.path.Path`. Confirmed
    unweighted (hard boolean cut).

    Attributes:
        iso_config: dict with age (Gyr) and z (metallicity, mass fraction)
            for the reference isochrone, plus optional survey/isochrone_model
            overrides and any other streamobs.match_filter.build_match_filter
            keyword argument (color_spread, error_multiplier, etc.).
        namespace: survey/release column namespace (e.g. "lsst_yr1") used to
            look up `<namespace>_<band>_obs` columns via
            `streamobs.columns.obs_col`. None uses bare `<band>_obs`.

    Rationale: streamobs's spline-based filter is the canonical implementation.
    Wrapping it here allows reuse without importing streamobs throughout
    the pipeline.
    """

    def __init__(self, iso_config: dict, namespace: str | None = None):
        """Initialize filter with reference isochrone.

        Parameters:
            iso_config: dict with keys age, z (or deserialize from config).
            namespace: survey/release column namespace (e.g. "lsst_yr1"),
                matching whichever `Survey`/`BackgroundSource` produced the
                catalogs this filter will be applied to.

        Raises:
            KeyError if age or z is missing.
        """
        if "age" not in iso_config or "z" not in iso_config:
            raise KeyError("iso_config must contain 'age' and 'z' (metallicity)")
        self.iso_config = dict(iso_config)
        self.namespace = namespace
        self._polygon_cache: dict[tuple, np.ndarray] = {}

    def _polygon(self, bands: list[str], distance_modulus: float) -> np.ndarray:
        """Build (or retrieve from cache) the matched-filter polygon for this
        (bands, distance_modulus) pair. Building involves isochrone sampling
        via ugali, so caching avoids repeating that work across calls at the
        same trial distance."""
        cache_key = (tuple(bands), float(distance_modulus))
        if cache_key not in self._polygon_cache:
            from streamobs.match_filter import build_match_filter

            kwargs = dict(self.iso_config)
            age = kwargs.pop("age")
            z = kwargs.pop("z")
            survey = kwargs.pop("survey", "lsst")
            isochrone_model = kwargs.pop("isochrone_model", "Marigo2017")
            self._polygon_cache[cache_key] = build_match_filter(
                distance_modulus=distance_modulus,
                age=age,
                metallicity=z,
                survey=survey,
                isochrone_model=isochrone_model,
                band_1=bands[0],
                band_2=bands[1],
                **kwargs,
            )
        return self._polygon_cache[cache_key]

    def select(
        self, catalog: pd.DataFrame, bands: list[str], distance_modulus: float
    ) -> np.ndarray:
        """Apply streamobs spline-based selection.

        Wraps streamobs's selection function; does not change numerics.

        Parameters:
            catalog: DataFrame with `<namespace>_<band>_obs` columns matching
                `self.namespace`.
            bands: List of photometric bands.
            distance_modulus: Trial distance modulus.

        Returns:
            Boolean selection mask.
        """
        from streamobs.columns import obs_col
        from streamobs.match_filter import is_in_match_filter

        polygon = self._polygon(bands, distance_modulus)
        mag_1 = catalog[obs_col(bands[0], self.namespace)]
        mag_2 = catalog[obs_col(bands[1], self.namespace)]
        return is_in_match_filter(mag_1, mag_2, polygon_vertices=polygon)


def make_raw_map(
    catalog: pd.DataFrame,
    selected: np.ndarray,
    pix: PixelizationSpec,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw HEALPix count map and valid_mask from selected stars.

    Parameters:
        catalog: DataFrame with 'ra' and 'dec' columns (degrees).
        selected: Boolean array, shape (len(catalog),), True for stars to count.
        pix: PixelizationSpec with nside.
        ra_col, dec_col: Column names for RA/Dec (default 'ra', 'dec').

    Returns:
        Tuple (raw_map, valid_mask):
            raw_map: HEALPix count map, shape (npix,), float. 0.0 both where
                there are valid pixels with no selected stars AND where the
                pixel is invalid (decision 22: 0 is the fill value everywhere;
                only valid_mask distinguishes the two cases).
            valid_mask: HEALPix bool mask, shape (npix,). True where the
                catalog has ANY row (selected or not) in that pixel — i.e.
                where the underlying catalog actually has coverage. False
                elsewhere (outside the catalog's footprint).

    Rationale: Explicit valid_mask (decision 22) distinguishes "zero counts"
    from "no data". A pixel is valid iff the catalog has coverage there at
    all (any row), independent of whether any row passed selection — so
    "0 selected stars in a covered pixel" and "no survey coverage" stay
    distinguishable. For a dense background catalog this traces the real
    survey footprint; for a stream-only catalog it does not (a stream only
    touches a handful of pixels) — per the build prompt (§4.5 step 5),
    stream stars never define validity, only the background's own valid_mask
    (computed once, cached) does, so a stream's own valid_mask output here is
    not meant to be used downstream.
    """
    npix = hp.nside2npix(pix.nside)
    selected = np.asarray(selected, dtype=bool)

    if len(catalog) == 0:
        return np.zeros(npix, dtype=float), np.zeros(npix, dtype=bool)

    all_pixels = hp.ang2pix(
        pix.nside,
        catalog[ra_col].to_numpy(dtype=float),
        catalog[dec_col].to_numpy(dtype=float),
        nest=pix.nest,
        lonlat=True,
    )

    valid_mask = np.zeros(npix, dtype=bool)
    valid_mask[np.unique(all_pixels)] = True

    counts = np.bincount(all_pixels[selected], minlength=npix).astype(float)
    raw_map = counts[:npix]

    return raw_map, valid_mask


def _tangent_plane_radec(pix: PixelizationSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (ra_deg, dec_deg), each shape pix.image_size_pix, for the sky
    position of every pixel in this window.

    Standard gnomonic (TAN) deprojection (Calabretta & Greisen 2002) of a
    regular tangent-plane grid centered on (center_ra, center_dec), with the
    window's own rotation_deg applied to the tangent-plane offsets before
    deprojecting — i.e. rotation is part of the tangent-plane projection
    itself, not a post-hoc image rotation (decision 15).
    """
    ny, nx = pix.image_size_pix
    row_offsets = (np.arange(ny) - (ny - 1) / 2.0) * pix.pixel_scale_deg
    col_offsets = (np.arange(nx) - (nx - 1) / 2.0) * pix.pixel_scale_deg
    eta_deg, xi_deg = np.meshgrid(row_offsets, col_offsets, indexing="ij")

    xi = np.radians(xi_deg)
    eta = np.radians(eta_deg)

    theta = np.radians(pix.rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xi_rot = xi * cos_t - eta * sin_t
    eta_rot = xi * sin_t + eta * cos_t

    ra0 = np.radians(pix.center_ra)
    dec0 = np.radians(pix.center_dec)

    rho = np.sqrt(xi_rot**2 + eta_rot**2)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    safe_rho = np.where(rho > 0, rho, 1.0)

    dec = np.where(
        rho > 0,
        np.arcsin(cos_c * np.sin(dec0) + (eta_rot * sin_c * np.cos(dec0)) / safe_rho),
        dec0,
    )
    ra = ra0 + np.where(
        rho > 0,
        np.arctan2(
            xi_rot * sin_c,
            rho * np.cos(dec0) * cos_c - eta_rot * np.sin(dec0) * sin_c,
        ),
        0.0,
    )

    return np.degrees(ra) % 360.0, np.degrees(dec)


def world_to_tangent_plane(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward gnomonic (TAN) projection: sky (ra, dec) -> local tangent-plane
    offsets (xi_deg, eta_deg), in degrees.

    This is the exact mathematical inverse of `_tangent_plane_radec`'s
    deprojection (same rotation convention: rotation_deg is applied to the
    tangent-plane offsets, not as a post-hoc sky rotation) — round-tripping
    a `_tangent_plane_radec` grid through this function recovers the
    original row/col offsets. Used to test whether a sky point falls inside
    a `Window`: a point is inside a `size_deg` x `size_deg` window iff
    `abs(xi_deg) <= size_deg / 2` and `abs(eta_deg) <= size_deg / 2`.

    Parameters:
        ra_deg, dec_deg: Sky positions (degrees), any shape.
        center_ra_deg, center_dec_deg: Tangent point / window center (degrees).
        rotation_deg: Window position angle (degrees), same convention as
            `PixelizationSpec.rotation_deg` / `Window.rotation_deg`.

    Returns:
        Tuple (xi_deg, eta_deg), same shape as the input.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra0 = np.radians(center_ra_deg)
    dec0 = np.radians(center_dec_deg)

    d_ra = ra - ra0
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(d_ra)
    xi_rot = np.cos(dec) * np.sin(d_ra) / cos_c
    eta_rot = (
        np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(d_ra)
    ) / cos_c

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Inverse of _tangent_plane_radec's rotation (xi_rot = xi*cos - eta*sin,
    # eta_rot = xi*sin + eta*cos) -> rotate back by -theta.
    xi = xi_rot * cos_t + eta_rot * sin_t
    eta = -xi_rot * sin_t + eta_rot * cos_t

    return np.degrees(xi), np.degrees(eta)


def project(
    raw_map: np.ndarray, valid_mask: np.ndarray, pix: PixelizationSpec
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
    When interpolating, an output pixel is only marked valid if ALL FOUR
    HEALPix neighbors contributing to its bilinear interpolation are
    themselves valid — otherwise interpolation would silently blend in the
    invalid-pixel fill value (0.0) as if it were real data (decision 22).
    """
    ny, nx = pix.image_size_pix
    ra_grid, dec_grid = _tangent_plane_radec(pix)

    if pix.interpolate:
        pixnums, weights = hp.get_interp_weights(
            pix.nside, ra_grid.ravel(), dec_grid.ravel(), nest=pix.nest, lonlat=True
        )
        values = np.sum(weights * raw_map[pixnums], axis=0)
        all_valid = np.all(valid_mask[pixnums], axis=0)
    else:
        pixnums = hp.ang2pix(
            pix.nside, ra_grid.ravel(), dec_grid.ravel(), nest=pix.nest, lonlat=True
        )
        values = raw_map[pixnums]
        all_valid = valid_mask[pixnums]

    image = values.reshape(ny, nx).astype(float)
    image_valid_mask = all_valid.reshape(ny, nx)
    image[~image_valid_mask] = 0.0

    return image, image_valid_mask


def combine_full_maps(
    background_raw_full: np.ndarray,
    stream_raw_full: np.ndarray,
    valid_mask_full: np.ndarray,
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
            Valid pixels are sums; invalid pixels are forced to 0.0 (decision 22
            fill value) regardless of what the inputs held there.

    Rationale: Additive combination is key to caching the finalized background
    once and reusing it across samples (§4.3b shortcut):
    finalize_full(background + stream) == finalize_full(background) + finalize_full(stream)
    if finalize_full is linear (fixed-kernel smoothing + fixed-basis polyfit).
    """
    combined = background_raw_full + stream_raw_full
    return np.where(valid_mask_full, combined, 0.0)


def finalize_full(
    combined_raw_full: np.ndarray, valid_mask_full: np.ndarray, cfg: dict | None = None
) -> np.ndarray:
    """Apply smoothing and polynomial background subtraction to full map.

    Operates ONLY on valid_mask==True pixels. Invalid pixels (outside footprint)
    are excluded from fitting and remain at the 0.0 fill value in output.

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
            shape (npix,), float. Invalid pixels remain at the 0.0 fill value.

    Rationale: This wraps the user's existing smoothing and background-subtraction
    functions (data_preparation.process_data_polyfit2d/polyfit2d — confirmed
    linear: fixed-kernel healpy smoothing + fixed-basis polyfit, PLAN.md §6.2),
    so the §4.3b shortcut is mathematically safe once wired up. Before the
    shortcut is enabled, a dedicated test must verify
    finalize(bg+stream) == finalize(bg) + finalize_local(stream).

    Raises:
        NotImplementedError if cfg['enabled'] is True — adapting
        data_preparation.py's catalog-level functions to operate on a
        HEALPix map is a separate, not-yet-scoped integration task
        (PLAN.md §6.1.3); only the disabled (identity) path is implemented
        so far, which is all the first milestone needs (decision 18).
    """
    enabled = bool(cfg.get("enabled", False)) if cfg else False
    if not enabled:
        return combined_raw_full

    raise NotImplementedError(
        "finalize_full(enabled=True) is not implemented yet: adapting "
        "data_preparation.process_data_polyfit2d/polyfit2d from catalog-level "
        "to HEALPix-map-level is a separate integration task (PLAN.md §6.1.3), "
        "not needed for the first milestone (decision 18 default: disabled)."
    )


def crop_window(
    finalized_full: np.ndarray,
    valid_mask_full: np.ndarray,
    window: "Window",
    pix: PixelizationSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Project and crop finalized full map to one window.

    Takes an already-finalized (smoothed/background-subtracted) full-sky map
    and crops it to one window via projection.

    Parameters:
        finalized_full: Full-sky HEALPix map (npix,), float.
        valid_mask_full: Full-sky HEALPix bool mask (npix,).
        window: Window instance with center_ra, center_dec, rotation_deg.
        pix: PixelizationSpec for projection (its own center/rotation are
            overridden by the window's).

    Returns:
        Tuple (windowed_image, windowed_valid_mask):
            windowed_image: 2D image, shape pix.image_size_pix, float.
            windowed_valid_mask: 2D bool mask, same shape.

    Rationale: Purely a crop/projection operation; no fitting happens here.
    Calls project() internally.
    """
    window_pix = dataclasses.replace(
        pix,
        center_ra=window.center_ra,
        center_dec=window.center_dec,
        rotation_deg=window.rotation_deg,
    )
    return project(finalized_full, valid_mask_full, window_pix)
