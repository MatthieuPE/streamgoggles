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


class ShiftedColorBoxFilter(MatchedFilter):
    """A deliberately "bad" matched filter: a plain color-magnitude box,
    same magnitude range as a real isochrone filter at the same distance,
    but shifted off the isochrone locus in color.

    Rationale: a decoy negative control for training. Unlike
    `StreamobsSplineFilter`, this has no isochrone shape at all, so it
    passes generic background contamination through without preferentially
    selecting isochrone-consistent (i.e. real stream-like) stars. Pairing a
    "good" and a "bad" filter at the same trial distance teaches a network
    what background contamination looks like versus a real overdensity,
    since both channels see the same background but only the good one is
    biased toward real stream members.

    Attributes:
        reference_filter: The real filter this decoy is shifted relative
            to. Its `_polygon()` (cached there) supplies both the magnitude
            range and the color center to shift away from -- so the decoy
            always tracks whatever isochrone/distance/bands the reference
            filter is configured with, with no separate isochrone sampling
            of its own.
        color_shift: Offset (mag) added to the reference polygon's median
            color to get this box's center. Any nonzero value moves the box
            off the isochrone locus; sign is arbitrary (bluer vs redder).
        color_width: Full width (mag) of the box in color.
    """

    def __init__(
        self,
        reference_filter: StreamobsSplineFilter,
        color_shift: float,
        color_width: float = 0.3,
    ):
        """Initialize the decoy filter.

        Parameters:
            reference_filter: StreamobsSplineFilter to shift away from.
            color_shift: Color offset (mag) from the reference polygon's
                median color. Must be nonzero enough that the box doesn't
                still overlap the isochrone locus, but that's the caller's
                responsibility -- not validated here (depends on the
                isochrone's own color width, which varies by age/z/distance).
            color_width: Full width (mag) of the box in color (default 0.3).
        """
        self.reference_filter = reference_filter
        self.color_shift = color_shift
        self.color_width = color_width

    @property
    def namespace(self) -> str | None:
        """Column namespace, taken from `reference_filter` -- both filters
        must read the same catalog columns, so there is only one namespace
        to track, not two that could drift out of sync."""
        return self.reference_filter.namespace

    def select(
        self, catalog: pd.DataFrame, bands: list[str], distance_modulus: float
    ) -> np.ndarray:
        """Apply the shifted color-magnitude box cut.

        Parameters:
            catalog: DataFrame with `<namespace>_<band>_obs` columns matching
                `self.namespace` (== `self.reference_filter.namespace`).
            bands: List of photometric bands (color = bands[0] - bands[1],
                magnitude axis = bands[0] -- same convention as
                `streamobs.match_filter.build_match_filter`).
            distance_modulus: Trial distance modulus (only used to look up
                the reference filter's polygon at this distance; the box
                itself has no distance dependence beyond that).

        Returns:
            Boolean selection mask, shape (len(catalog),).
        """
        from streamobs.columns import obs_col

        polygon = self.reference_filter._polygon(bands, distance_modulus)
        mag_min, mag_max = polygon[:, 1].min(), polygon[:, 1].max()
        color_center = float(np.median(polygon[:, 0]))

        mag_1 = catalog[obs_col(bands[0], self.namespace)].to_numpy(dtype=float)
        mag_2 = catalog[obs_col(bands[1], self.namespace)].to_numpy(dtype=float)
        color = mag_1 - mag_2

        color_min = color_center + self.color_shift - self.color_width / 2.0
        color_max = color_center + self.color_shift + self.color_width / 2.0

        return (
            (mag_1 >= mag_min)
            & (mag_1 <= mag_max)
            & (color >= color_min)
            & (color <= color_max)
        )


class ColorBoxFilter(MatchedFilter):
    """A deliberately "bad" matched filter: a fixed color-magnitude box.

    Rationale: the same decoy negative control as `ShiftedColorBoxFilter`, but
    defined by absolute limits instead of relative to a reference filter's
    polygon. `ShiftedColorBoxFilter` moves with whatever isochrone, bands and
    trial distance its reference is configured with, so the decoy channel means
    something slightly different in every sample and every trial distance. Here
    the box is stated once, so the same region of colour-magnitude space is
    selected in every sample, at every trial distance, in every experiment --
    which is what makes the decoy channel comparable across runs.

    Being distance-independent is the point, not an omission: a real filter
    tracks the isochrone as the trial distance changes, and this one does not,
    which is exactly the contrast the decoy channel is there to teach.

    Attributes:
        color_range: (min, max) of ``bands[0] - bands[1]``, magnitudes.
        mag_range: (min, max) of the ``bands[0]`` magnitude.
        namespace: column namespace (``<survey>_<release>``) of the catalog
            columns to read, e.g. "lsst_yr1".
    """

    def __init__(
        self,
        color_range: tuple[float, float],
        mag_range: tuple[float, float],
        namespace: str | None = None,
    ):
        """Initialize the fixed decoy box.

        Parameters:
            color_range: (min, max) colour of the box, magnitudes. Choose it
                off the isochrone locus of the filters it accompanies -- not
                validated here, since where the locus sits depends on the
                isochrone and the distances in use.
            mag_range: (min, max) magnitude of the box in ``bands[0]``.
                Typically the magnitude range the catalog itself is cut to.
            namespace: column namespace to read magnitudes from. Usually the
                same as the "good" filter's.

        Raises:
            ValueError if either range is empty or reversed.
        """
        for name, (low, high) in (
            ("color_range", color_range),
            ("mag_range", mag_range),
        ):
            if not high > low:
                raise ValueError(
                    f"{name} must be (min, max) with max > min, got {(low, high)}"
                )
        self.color_range = (float(color_range[0]), float(color_range[1]))
        self.mag_range = (float(mag_range[0]), float(mag_range[1]))
        self._namespace = namespace

    @property
    def namespace(self) -> str | None:
        """Column namespace the magnitudes are read from."""
        return self._namespace

    def select(
        self, catalog: pd.DataFrame, bands: list[str], distance_modulus: float
    ) -> np.ndarray:
        """Apply the fixed color-magnitude box cut.

        Parameters:
            catalog: DataFrame with `<namespace>_<band>_obs` columns.
            bands: colour = ``bands[0] - bands[1]``, magnitude axis =
                ``bands[0]`` (same convention as the other filters).
            distance_modulus: ignored -- the box is fixed on the sky's own
                colour-magnitude plane, which is what makes it comparable
                across trial distances.

        Returns:
            Boolean selection mask, shape (len(catalog),).
        """
        from streamobs.columns import obs_col

        mag_1 = catalog[obs_col(bands[0], self.namespace)].to_numpy(dtype=float)
        mag_2 = catalog[obs_col(bands[1], self.namespace)].to_numpy(dtype=float)
        color = mag_1 - mag_2
        color_min, color_max = self.color_range
        mag_min, mag_max = self.mag_range
        return (
            (mag_1 >= mag_min)
            & (mag_1 <= mag_max)
            & (color >= color_min)
            & (color <= color_max)
        )


# The keys each filter type takes in a filters config, required and optional.
FILTER_TYPES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "isochrone": (frozenset({"reference_isochrone"}), frozenset()),
    "box": (frozenset({"color_range", "mag_range"}), frozenset()),
    "shifted_box": (
        frozenset({"reference", "color_shift"}),
        frozenset({"color_width"}),
    ),
}


def build_matched_filters(
    filters_cfg: dict[str, dict], namespace: str | None = None
) -> dict[str, MatchedFilter]:
    """Build the named matched filters from a plain configuration.

    This is the one place a filters configuration becomes filters, so choosing
    them is a matter of editing a dict (in a notebook) or the ``filters``
    section of ``config/matched_filter.yaml`` -- never code. ``filters_cfg``
    maps a channel name to one filter spec; its order is kept, and it is the
    order of the network's input channels at each trial distance. Three types:

    - ``{"type": "isochrone", "reference_isochrone": {"age": 12.5, "z": 0.0002}}``
      -- `StreamobsSplineFilter`, the real matched filter around an isochrone.
    - ``{"type": "box", "color_range": (1.2, 1.5), "mag_range": (18.0, 24.5)}``
      -- `ColorBoxFilter`, a decoy at absolute colour and magnitude limits: the
      same selection at every trial distance and in every run.
    - ``{"type": "shifted_box", "reference": "good", "color_shift": 0.5,
      "color_width": 0.3}`` -- `ShiftedColorBoxFilter`, the older decoy placed
      relative to an isochrone filter defined earlier in the same config.

    Parameters:
        filters_cfg: channel name -> filter spec, as above.
        namespace: survey/release column namespace (e.g. "lsst_yr1") of the
            catalogs the filters will be applied to.

    Returns:
        Channel name -> filter, in the order of ``filters_cfg``.

    Raises:
        ValueError for an unknown type, a missing or unexpected key, or a
        ``shifted_box`` whose reference is not an isochrone filter defined
        before it.
    """
    filters: dict[str, MatchedFilter] = {}
    for name, spec in filters_cfg.items():
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in FILTER_TYPES:
            raise ValueError(
                f"filter {name!r}: unknown type {kind!r}, expected one of "
                f"{sorted(FILTER_TYPES)}"
            )
        required, optional = FILTER_TYPES[kind]
        missing = required - spec.keys()
        unexpected = spec.keys() - required - optional
        if missing or unexpected:
            raise ValueError(
                f"filter {name!r} ({kind}): "
                + "; ".join(
                    part
                    for part in (
                        f"missing {sorted(missing)}" if missing else "",
                        f"unexpected {sorted(unexpected)}" if unexpected else "",
                    )
                    if part
                )
            )
        if kind == "isochrone":
            filters[name] = StreamobsSplineFilter(
                iso_config=dict(spec["reference_isochrone"]), namespace=namespace
            )
        elif kind == "box":
            filters[name] = ColorBoxFilter(
                color_range=tuple(spec["color_range"]),
                mag_range=tuple(spec["mag_range"]),
                namespace=namespace,
            )
        else:
            reference = filters.get(spec["reference"])
            if not isinstance(reference, StreamobsSplineFilter):
                raise ValueError(
                    f"filter {name!r} (shifted_box): reference {spec['reference']!r} "
                    "must be an isochrone filter defined earlier in the config"
                )
            filters[name] = ShiftedColorBoxFilter(
                reference_filter=reference,
                color_shift=spec["color_shift"],
                **(
                    {"color_width": spec["color_width"]}
                    if "color_width" in spec
                    else {}
                ),
            )
    return filters


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


def window_to_healpix_indices(
    window: "Window", pix: PixelizationSpec, nside: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map every HEALPix pixel this window covers to an image pixel.

    The inverse direction of `project`/`crop_window`: instead of asking
    "what sky position does each image pixel look at", this asks "which
    image pixel does each HEALPix pixel fall into". Iterating over HEALPix
    pixels rather than image pixels is what makes the result gap-free --
    walking image pixels instead would leave unfilled HEALPix pixels
    wherever the HEALPix grid is finer than the image grid.

    Sampling is nearest-neighbour, deliberately: this exists to put model
    predictions back on the sky, and interpolating probabilities across
    neighbouring pixels would blur a per-pixel number the whole
    `stream_detection` design intends to be read (and thresholded) as-is.

    Parameters:
        window: the Window the image was cropped to.
        pix: PixelizationSpec the image was produced with (its
            `image_size_pix`/`pixel_scale_deg` define the image grid; its
            own center/rotation are ignored in favour of `window`'s, exactly
            as `crop_window` does).
        nside: HEALPix resolution of the output map.

    Returns:
        Tuple `(healpix_indices, rows, cols, edge_margin_deg)`, all 1-D and
        the same length:

        - `healpix_indices`: HEALPix pixel ids covered by this window,
        - `rows`, `cols`: the image pixel each one samples,
        - `edge_margin_deg`: how far inside the window each one sits
          (0 at the border, largest at the center). `stitch_windows_to_healpix`
          uses it to choose between overlapping windows.
    """
    ny, nx = pix.image_size_pix
    scale = pix.pixel_scale_deg
    half_y = (ny - 1) / 2.0 * scale
    half_x = (nx - 1) / 2.0 * scale

    # A disc comfortably containing the window's corners; the tangent-plane
    # test below does the exact selection.
    radius_deg = float(np.hypot(half_x, half_y)) + scale
    center_vec = hp.ang2vec(window.center_ra, window.center_dec, lonlat=True)
    candidates = hp.query_disc(
        nside, center_vec, np.radians(radius_deg), inclusive=True
    )
    if candidates.size == 0:
        empty_i = np.empty(0, dtype=int)
        return empty_i, empty_i, empty_i, np.empty(0, dtype=float)

    ra, dec = hp.pix2ang(nside, candidates, lonlat=True)
    xi_deg, eta_deg = world_to_tangent_plane(
        ra, dec, window.center_ra, window.center_dec, window.rotation_deg
    )

    cols = np.rint(xi_deg / scale + (nx - 1) / 2.0).astype(int)
    rows = np.rint(eta_deg / scale + (ny - 1) / 2.0).astype(int)
    inside = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)

    candidates, rows, cols = candidates[inside], rows[inside], cols[inside]
    margin = np.minimum(
        half_x - np.abs(xi_deg[inside]), half_y - np.abs(eta_deg[inside])
    )
    return candidates, rows, cols, margin


def stitch_windows_to_healpix(
    images: list[np.ndarray],
    windows: list["Window"],
    pix: PixelizationSpec,
    nside: int,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Place per-window images back onto one full-sky HEALPix map.

    Overlapping windows are resolved by **keeping the value from the window
    in which each pixel sits furthest from an edge** -- the "overlap-tile"
    strategy from the original U-Net paper -- rather than averaging the
    overlaps.

    Averaging would be the obvious alternative and is the wrong one here:
    the model's output is a per-pixel probability meant to be thresholded by
    whoever reads it, and the mean of 0.9 and 0.1 is 0.5, which is not the
    same claim as a genuine 0.5. Choosing a single window per pixel keeps
    every value one the model actually produced. It also picks the *best*
    such value, since a pixel near a window's edge was predicted with less
    surrounding context than one at its center.

    Parameters:
        images: one 2-D image per window, each shaped `pix.image_size_pix`.
        windows: the windows those images were cropped to, same order.
        pix: PixelizationSpec defining the image grid.
        nside: HEALPix resolution of the output map.
        fill_value: value for HEALPix pixels no window covers.

    Returns:
        Tuple `(healpix_map, covered_mask)`: the stitched map (`npix`,) and
        a bool mask of which pixels any window actually covered -- pixels
        outside it hold `fill_value` and must not be read as data.

    Raises:
        ValueError if `images` and `windows` differ in length, or an image's
            shape doesn't match `pix.image_size_pix`.
    """
    if len(images) != len(windows):
        raise ValueError(
            f"{len(images)} images but {len(windows)} windows; they must correspond"
        )

    npix = hp.nside2npix(nside)
    out = np.full(npix, fill_value, dtype=float)
    best_margin = np.full(npix, -np.inf, dtype=float)
    covered = np.zeros(npix, dtype=bool)

    for image, window in zip(images, windows):
        image = np.asarray(image)
        if image.shape != tuple(pix.image_size_pix):
            raise ValueError(
                f"image shape {image.shape} != pix.image_size_pix "
                f"{tuple(pix.image_size_pix)}"
            )
        healpix_indices, rows, cols, margin = window_to_healpix_indices(
            window, pix, nside
        )
        if healpix_indices.size == 0:
            continue
        # Only overwrite where this window sees the pixel more centrally
        # than any window already did.
        wins = margin > best_margin[healpix_indices]
        chosen = healpix_indices[wins]
        out[chosen] = image[rows[wins], cols[wins]]
        best_margin[chosen] = margin[wins]
        covered[chosen] = True

    return out, covered
