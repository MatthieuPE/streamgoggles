"""Single-stream and population-stream injection into background.

Pipeline for one stream:

1. Resolve richness (nstars/mass/surface_brightness) to nstars (inject_utils
   conversions), then realize stream via stream_source (phi1, phi2, dist,
   true mags).
2. Place the realized stream at a random position + orientation inside the
   background's footprint, then rejection-sample a window with >=5 deg of
   stream inside it (decision 21).
3. Inject through survey selection (real streamobs.observed.StreamInjector):
   adds ra/dec (already present from step 2, so untouched), observed mags,
   errors, and the detection flag. Rows the survey never detected carry a
   "BAD_MAG" sentinel instead of a magnitude -- filtered out immediately
   after injection, before anything downstream touches those columns
   numerically.
4. Apply the SAME background cuts + magnitude clipping (data_preparation.py,
   decision 13) used on the background catalog.
5. For each (matched filter, trial distance) pair already cached on
   `background`: select + make_raw_map on the injected stream catalog.
6. Combine with the cached background map for that same (filter, distance),
   finalize (identity unless finalize_cfg enables it -- not implemented yet,
   matched_filter.py), crop to the window.
7. Label: the stream-only (background-excluded) raw counts for that same
   (filter, distance), cropped through the identical window/valid_mask as
   its map_stack channel (2026-09-09 pivot -- see label_policy="stream_count"
   below; the old rasterize.py-based labels are still available for
   label_policy in {"binary", "density", "soft_distance"}). 2026-09-15
   addition: label_policy="stream_detection" derives a per-channel binary
   detection target from the same stream-only raw counts instead of
   returning them directly -- see class docstring and PLAN.md section 6.12
   (the fix for Stage 1's count-amplitude recovery problem: retarget to a
   bounded detection label instead of trying to regress the raw count).
8. Return Sample.

Multiple named matched filters (2026-09-09 pivot): a sample's input is one
channel per (filter, distance) pair, not one per distance -- typically a
real isochrone filter plus one or more deliberately "bad" decoy filters
(matched_filter.ShiftedColorBoxFilter), so a network sees both what a real
overdensity looks like and what generic background contamination looks like
at the same trial distance. Channels are ordered distance-major,
filter-minor (`channel_index = dist_idx * n_filters + filter_idx`); the
exact mapping is recorded in Sample.metadata["channels"].

Pure-background samples (no stream) use inject_background_only() instead,
which skips steps 1, 3-5, 7 entirely -- window is placed randomly and there
is nothing to rasterize (an all-zero label_stack).

Population injection (stub, decision 19): sum raw stream maps, union labels.
Designed to compose the single-stream path without rewriting it.

Rationale: Isolate injection logic from background/window/filter. Enable
testing injection independently. Keep code simple: one stream at a time,
then compose.
"""

import dataclasses
import logging

import astropy.units as u
import gala.coordinates as gc
import healpy as hp
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from streamobs.columns import flag_col
from streamobs.observed import StreamInjector as ObsStreamInjector

from streamgoggles import inject_utils, rasterize
from streamgoggles.background import Background
from streamgoggles.data_preparation import Cut, apply_cuts, apply_magnitude_clipping
from streamgoggles.matched_filter import (
    MatchedFilter,
    PixelizationSpec,
    WindowProjection,
    combine_full_maps,
    crop_window,
    finalize_full,
    make_raw_map,
)
from streamgoggles.sample import Sample
from streamgoggles.stream_sources import StreamSource
from streamgoggles.windows import Window, sample_stream_window

logger = logging.getLogger(__name__)


def _isochrone_cfg_from_params(params: dict) -> dict:
    """Build the same isochrone-factory dict shape stream_sources.py uses
    (minus 'release', which ugali.isochrone.factory doesn't accept)."""
    return {
        "name": "Marigo2017",
        "survey": params.get("survey", "lsst"),
        "age": params["age"],
        "z": params["z"],
        "band_1": params.get("band_1", "g"),
        "band_2": params.get("band_2", "r"),
    }


def _stream_length_width_for_richness(params: dict) -> tuple[float, float]:
    """Best-effort (length, width) footprint estimate for the
    surface_brightness richness conversion, valid for both morphologies
    (decision 5): 'uniform' has both directly; 'spline' has no explicit
    'length', so it's derived from the control points' phi1 span."""
    width = params.get("width")
    if width is None:
        raise ValueError("richness_kind='surface_brightness' requires params['width']")
    if params["morphology"] == "uniform":
        length = params["length"]
    else:
        phi1_nodes = [p["phi1"] for p in params["control_points"]]
        length = max(phi1_nodes) - min(phi1_nodes)
    return float(length), float(width)


def resolve_richness_to_nstars(params: dict, richness_kind: str) -> dict:
    """Return a copy of `params` with 'richness' (in whichever unit
    `richness_kind` names) replaced by an integer 'nstars' (decision 16/§4.1).

    Parameters:
        params: Stream parameter dict. If it has no 'richness' key, returned
            unchanged (e.g. when nstars was already set directly).
        richness_kind: "nstars" (identity), "mass" (Msun, via
            inject_utils.convert_Mass_to_N -- ugali's own IsochroneModel.stellar_mass()
            convention), or "surface_brightness" (mag/arcsec^2, via
            inject_utils.convert_SurfaceBrightness_to_N -- brentq inversion).

    Returns:
        New params dict with 'nstars' set (and 'richness' removed); nstars is
        the TRUE star count, never reduced by selection (decision 16).

    Raises:
        ValueError for an unknown richness_kind, or for surface_brightness
            richness without the width (and, for uniform morphology, length)
            params it needs.
    """
    if "richness" not in params:
        return params

    richness = params["richness"]
    if richness_kind == "nstars":
        nstars = richness
    elif richness_kind == "mass":
        isochrone_params = {"isochrone": _isochrone_cfg_from_params(params)}
        nstars = inject_utils.convert_Mass_to_N(
            richness, isochrone_params=isochrone_params
        )
    elif richness_kind == "surface_brightness":
        length, width = _stream_length_width_for_richness(params)
        isochrone_params = {
            "isochrone": _isochrone_cfg_from_params(params),
            "distance_modulus": {"center": {"value": params["distance_modulus"]}},
        }
        nstars = inject_utils.convert_SurfaceBrightness_to_N(
            richness,
            isochrone_params=isochrone_params,
            stream_length=length,
            stream_width=width,
            band=params.get("band_1", "g"),
        )
    else:
        raise ValueError(
            f"Unknown richness_kind {richness_kind!r}; expected "
            "'nstars'/'mass'/'surface_brightness'"
        )

    new_params = {k: v for k, v in params.items() if k != "richness"}
    new_params["nstars"] = round(nstars)
    return new_params


def place_stream_in_footprint(
    stream_df: pd.DataFrame,
    footprint: np.ndarray,
    nside: int,
    rng: np.random.Generator,
    rotation_deg: float | None = None,
    center: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Place a realized stream (phi1/phi2 frame) at a random position and
    orientation somewhere inside `footprint`, adding 'ra'/'dec' columns.

    Parameters:
        stream_df: DataFrame with 'phi1'/'phi2' columns (stream_sources.py's
            realize() output).
        footprint: HEALPix bool mask (npix,), True for valid pixels
            (typically background.footprint).
        nside: HEALPix NSIDE matching `footprint`.
        rng: np.random.Generator instance.
        rotation_deg: Position angle of the phi1 axis on the sky (degrees).
            If None (default), drawn uniformly at random -- "orientation, if
            free, uniform within the footprint" (build prompt §4.4b). Pass
            an explicit value to use a sampled/fixed "orientation" parameter
            instead (e.g. config/streams/injection_grid.yaml's free
            `orientation` -- StreamInjector.inject_single_stream forwards
            `params["orientation"]` here when present). The placement
            *position* (which footprint pixel) is always random regardless
            (decision: "position: uniform_in_footprint" is never a tunable
            parameter, only a sentinel meaning "yes, place it").
        center: ``(ra, dec)`` of the stream frame's origin, degrees. For
            evaluation setups that need streams at chosen positions relative
            to each other (`StreamInjector.inject_streams_full_sky`); training
            always leaves it None, so the position stays random. Giving it
            skips the random draw of the position.

    Returns:
        Copy of `stream_df` with 'ra'/'dec' columns added (degrees, ICRS).
        The placement is recorded in ``.attrs["placement"]`` as
        ``{"center_ra", "center_dec", "rotation_deg"}`` (degrees): the stream
        frame's origin and the position angle of its phi1 axis, enough to
        recompute any sky position's (phi1, phi2), e.g. to measure flagged
        pixels in bands along the track
        (`evaluation.footprint.stream_frame_coordinates`).

    Raises:
        ValueError if footprint has no valid pixels.

    Rationale: streamobs's own StreamInjector.phi_to_radec()/_find_gc_frame()
    place a stream by rejection-sampling random great circles until enough
    points fall in a mask -- designed for the survey's own wide footprint.
    Our background footprint is a small StudyRegion-restricted patch
    (decision 24), for which that rejection search can fail outright
    (verified empirically: 0 successes in 1000 trials at survey scale). So
    placement here is direct instead: pick a random valid footprint pixel as
    the great-circle's phi1=phi2=0 origin (gala.coordinates.GreatCircleICRSFrame,
    same construction as background_sources._study_region_to_phi_box), with a
    position angle for the phi1 axis via SkyCoord.directional_offset_by().
    ra/dec set this way are then left alone by streamobs's own injection (it
    only fills ra/dec when absent).
    """
    valid_pixels = np.flatnonzero(footprint)
    if valid_pixels.size == 0:
        raise ValueError("footprint has no valid pixels to place a stream in")

    if center is None:
        pixel = rng.choice(valid_pixels)
        center_ra, center_dec = hp.pix2ang(nside, int(pixel), lonlat=True)
    else:
        center_ra, center_dec = float(center[0]), float(center[1])
    if rotation_deg is None:
        rotation_deg = float(rng.uniform(0.0, 360.0))
    else:
        rotation_deg = float(rotation_deg)

    center = SkyCoord(ra=float(center_ra) * u.deg, dec=float(center_dec) * u.deg)
    second = center.directional_offset_by(rotation_deg * u.deg, 1.0 * u.deg)
    gc_frame = gc.GreatCircleICRSFrame.from_endpoints(center, second, origin=center)

    stream_coord = SkyCoord(
        phi1=stream_df["phi1"].to_numpy(dtype=float) * u.deg,
        phi2=stream_df["phi2"].to_numpy(dtype=float) * u.deg,
        frame=gc_frame,
    )
    icrs = stream_coord.transform_to("icrs")

    out = stream_df.copy()
    out["ra"] = icrs.ra.deg
    out["dec"] = icrs.dec.deg
    out.attrs["placement"] = {
        "center_ra": float(center_ra),
        "center_dec": float(center_dec),
        "rotation_deg": rotation_deg,
    }
    return out


class StreamInjector:
    """Inject stream(s) into background, apply survey effects, build sample.

    Attributes:
        background: Background instance (catalog + cached maps), built with
            the SAME `matched_filters` dict (by name) as this injector.
        matched_filters: dict mapping filter name -> MatchedFilter instance
            (2026-09-09 pivot from a single filter). Applied to the stream
            catalog once per (filter, distance) pair, mirroring how
            `background` was cached.
        stream_source: StreamSource instance.
        cuts: List of Cut rules (applied to stream as well as background,
            via data_preparation.apply_cuts -- decision 13).
        clipping: Magnitude clipping dict, or None.
        pix: PixelizationSpec.
        survey, release: Passed to streamobs.observed.StreamInjector and
            used to build the `<survey>_<release>` column namespace --
            must match whatever `background`/`matched_filters` were built
            with, or cuts/select will look up the wrong columns.
        bands: Bands injected/selected on (e.g. ["g", "r"]).
        richness_kind: Unit of params['richness'] ("nstars"/"mass"/
            "surface_brightness" -- see resolve_richness_to_nstars).
        label_policy: "stream_count" (default, 2026-09-09 pivot): the label
            is the true stream-only (background-excluded) raw count for each
            (filter, distance) channel, computed directly here -- no
            rasterize.py involved. "stream_detection" (2026-09-15 addition,
            PLAN.md section 6.12): same per-channel stream-only raw count,
            but hard-thresholded to a binary {0.0, 1.0} detection target
            (`stream_raw > count_threshold`) instead of returned as a literal
            count -- pairs with UNet(head="sigmoid") and a bounded
            segmentation loss (models.losses.DiceLoss/BCEWithLogitsLoss/...)
            rather than a regression loss, sidestepping stream_count's
            amplitude-under-recovery problem (the actual goal is "is there a
            stream here", not the exact count). Otherwise ("binary"/
            "density"/"soft_distance"): dispatched to rasterize.rasterize()
            as before, a single distance-and-filter-independent 2D label
            broadcast across every channel.
        label_config: Extra rasterize.rasterize() kwargs, matching
            StreamConfig's YAML `label:` block (decision 9): "dilate_to_width"
            (binary), "normalization" (density), "smooth_sigma_deg" (density).
            Unused when label_policy is "stream_count" or "stream_detection".
        count_threshold: Only used when label_policy="stream_detection" --
            the per-pixel stream-only raw star count above which that pixel
            is labeled a detection (default 1.0 star). Chosen per (filter,
            distance) channel, same as the count itself -- a decoy filter's
            near-zero stream_raw at a pixel a real stream crosses stays
            below threshold, preserving the 2026-09-09 pivot's
            filter-dependence property.

            NOT a literal integer star count, despite the name: `stream_raw`
            is projected through crop_window's bilinear interpolation
            (matched_filter.py's project(), interpolate=True by default), so
            the value at any given output pixel is a smoothed blend of
            neighboring native HEALPix pixel counts, not the discrete count
            of stars landing in that exact pixel -- confirmed for real
            (>99% of nonzero values are non-integer to within 1e-3).
            `count_threshold` should be read as "interpolated local density,
            in star-count units," not literally "at least N stars here."

            Default 1.0 (2026-09-15, PLAN.md section 6.13): calibrated
            empirically against this project's real geometry (width=0.2,
            length=8.0 deg) at nside=512, sweeping both count_threshold and
            stream surface_brightness (30-35 mag/arcsec^2) and comparing the
            "good" (real isochrone) filter's positive-pixel count against
            the "decoy" (shifted color-box) filter's at the same threshold --
            see create_data.ipynb's "Calibrating count_threshold empirically"
            section (§4) for the exact code and the full result table (exact
            figures are somewhat sensitive to cuts_cfg/clipping_cfg, which
            differ slightly between create_data.ipynb and train_model.ipynb;
            the conclusion below was checked against both). Thresholds >=2
            already leave the faintest streams in this range
            (surface_brightness 35) completely undetected (an entirely
            empty label, not just a fainter tail trimmed); 1.0 is the
            lowest threshold where every richness point 30-35 stays
            non-empty. The real cost: at the bright end (SB 30-31), the
            decoy filter's positive-pixel count reaches roughly two-thirds
            of the "good" filter's at threshold=1.0 -- real contamination,
            not fully suppressed the way a higher threshold would, falling
            to a few percent by SB 33 and above (see PLAN.md section 6.13
            for the full threshold-vs-richness table and the reasoning for
            accepting this trade-off).
        finalize_cfg: Passed to matched_filter.finalize_full() when
            combining background + stream maps -- must match whatever
            `background` was cached with, or the combined map wouldn't be
            treated consistently.

    Rationale: Encapsulate injection logic; enable reuse with different
    backgrounds/filters/sources by swapping attributes.
    """

    def __init__(
        self,
        background: Background,
        matched_filters: dict[str, MatchedFilter],
        stream_source: StreamSource,
        cuts: list[Cut],
        clipping: dict | None,
        pix: PixelizationSpec,
        survey: str = "lsst",
        release: str = "dp2",
        bands: tuple[str, str] = ("g", "r"),
        richness_kind: str = "nstars",
        label_policy: str = "stream_count",
        label_config: dict | None = None,
        finalize_cfg: dict | None = None,
        count_threshold: float = 1.0,
        min_stream_length_deg: float = 5.0,
    ):
        """Initialize injector.

        Parameters:
            background: Background instance.
            matched_filters: dict mapping filter name -> MatchedFilter
                instance -- must use the same names as whatever
                `background` was built with (`Background.load_or_cache`'s
                own `matched_filters`), since channels are looked up by name.
            stream_source: StreamSource instance.
            cuts: List of Cut rules.
            clipping: Magnitude clipping dict, or None.
            pix: PixelizationSpec. `image_size_pix` is assumed square (Stage
                1 simplification -- window sampling only supports a single
                size_deg, matching windows.py's own API).
            survey, release: Column namespace components; must match
                `background`/`matched_filters`.
            bands: The two bands injected/selected on.
            richness_kind: Unit of params['richness'].
            label_policy: "stream_count" (default), "stream_detection", or a
                rasterize.py policy (see class docstring).
            label_config: Extra rasterize.rasterize() kwargs (see class docstring).
            finalize_cfg: Finalization config, matching whatever
                `background` was cached with.
            count_threshold: Only used when label_policy="stream_detection"
                (see class docstring).
        """
        self.background = background
        self.matched_filters = matched_filters
        self.filter_names = list(matched_filters)
        self.stream_source = stream_source
        self.cuts = cuts or []
        self.clipping = clipping or {}
        self.pix = pix
        self.survey = survey
        self.release = release
        self.bands = tuple(bands)
        self.richness_kind = richness_kind
        self.label_policy = label_policy
        self.label_config = label_config or {}
        self.finalize_cfg = finalize_cfg
        self.count_threshold = count_threshold
        # How much of the stream a training window must hold (decision 21's
        # 5 degrees by default); lower it when streams can be shorter than
        # that -- Tucana III is 4.8 degrees long.
        self.min_stream_length_deg = min_stream_length_deg
        self.namespace = f"{survey}_{release}" if release else survey
        self._obs_injector = ObsStreamInjector(survey, release=release)

    def _realize_and_inject(
        self,
        params: dict,
        rng: np.random.Generator,
        center: tuple[float, float] | None = None,
    ):
        """Realize a stream, place it, run it through the survey model, and
        apply this injector's cuts/clipping.

        Shared by `inject_single_stream` and `inject_stream_full_sky` so the
        two cannot drift apart. Draws from `rng` in a fixed order
        (realize -> place -> inject), which callers rely on for
        reproducibility.

        Returns:
            Tuple `(detected, resolved_params, placement)`: the surviving
            observed catalog, the parameter dict with richness resolved to
            `nstars` and bands defaulted, and where the stream was placed
            (`place_stream_in_footprint`'s ``attrs["placement"]``).
        """
        resolved_params = dict(params)
        resolved_params.setdefault("band_1", self.bands[0])
        resolved_params.setdefault("band_2", self.bands[1])
        # The stream's true magnitudes must be in the photometric system of the
        # survey that then observes it: left unset, the stream source falls
        # back to LSST, and a DES injector would observe LSST-band stars.
        resolved_params.setdefault("survey", self.survey)
        resolved_params.setdefault("release", self.release)
        resolved_params = resolve_richness_to_nstars(
            resolved_params, self.richness_kind
        )

        stream_df = self.stream_source.realize(resolved_params, rng)
        placed_df = place_stream_in_footprint(
            stream_df,
            self.background.footprint,
            self.pix.nside,
            rng,
            rotation_deg=resolved_params.get("orientation"),
            center=center,
        )
        placement = dict(placed_df.attrs["placement"])

        injected_df = self._obs_injector.inject(
            placed_df, bands=list(self.bands), rng=rng, verbose=False
        )
        detected = injected_df[injected_df[flag_col(self.namespace)]].reset_index(
            drop=True
        )

        detected = apply_cuts(detected, self.cuts, namespace=self.namespace)
        if self.clipping:
            detected = apply_magnitude_clipping(
                detected, self.clipping, namespace=self.namespace
            )
        return detected, resolved_params, placement

    def _build_full_sky_channels(self, detected):
        """Build every `(distance, filter)` channel as a FULL-SKY map.

        Distance-major, filter-minor channel order: every filter's map at one
        distance is contiguous. `Sample.metadata["channels"]` records this
        mapping explicitly so nothing downstream has to guess it.

        Returns:
            Tuple `(finalized_full, stream_raw_full, channels_meta)`, lists
            of equal length. `stream_raw_full` is the stream-only count map,
            deliberately **not** thresholded here: any detection threshold
            must be applied after cropping, since cropping interpolates.
        """
        distance_moduli = sorted(
            self.background.raw_map_full_dict[self.filter_names[0]]
        )
        finalized_full = []
        stream_raw_full = []
        channels_meta = []
        for dm in distance_moduli:
            for filter_name in self.filter_names:
                selected = self.matched_filters[filter_name].select(
                    detected, list(self.bands), dm
                )
                stream_raw, _stream_valid = make_raw_map(detected, selected, self.pix)
                combined = combine_full_maps(
                    self.background.raw_map_full_dict[filter_name][dm],
                    stream_raw,
                    self.background.valid_mask_full,
                )
                finalized = finalize_full(
                    combined, self.background.valid_mask_full, self.finalize_cfg
                )
                finalized_full.append(finalized)
                stream_raw_full.append(stream_raw)
                channels_meta.append({"filter": filter_name, "distance_modulus": dm})
        return finalized_full, stream_raw_full, channels_meta

    def _build_window_channels(self, detected, window):
        """Every `(distance, filter)` channel of ONE window, without full-sky maps.

        The same values as `_build_full_sky_channels` followed by `crop_window`
        on each channel, computed only where the window reads the sky:

        - the window's projection (for each window pixel, the HEALPix pixels
          around it and their weights, and whether they are valid) is computed
          once and shared by every channel, instead of once per channel and
          per label;
        - the stream's stars are placed on HEALPix pixels once, and each
          channel's selected stars are counted only on the pixels the window
          reads, then added to the cached background map there, instead of
          building two full-sky maps (about 3 million pixels each at
          nside 512) per channel;
        - a filter whose selection does not depend on the trial distance (the
          fixed decoy box) is evaluated once and reused at every distance.

        A test checks the result against the full-sky path, pixel for pixel.

        Returns:
            Tuple ``(maps, raw_labels, valid, channels_meta)``: the window's
            combined map per channel, its stream-only count map per channel
            (unthresholded), the window's validity mask, and the channel
            order, distance-major as `_build_full_sky_channels`.
        """
        projection = WindowProjection.for_window(
            window, self.pix, self.background.valid_mask_full
        )
        valid_at = self.background.valid_mask_full[projection.pixnums]
        if len(detected):
            star_pixels = hp.ang2pix(
                self.pix.nside,
                detected["ra"].to_numpy(dtype=float),
                detected["dec"].to_numpy(dtype=float),
                nest=self.pix.nest,
                lonlat=True,
            )
        else:
            star_pixels = np.empty(0, dtype=np.int64)
        distance_moduli = sorted(
            self.background.raw_map_full_dict[self.filter_names[0]]
        )
        maps, raw_labels, channels_meta = [], [], []
        distance_independent = {}
        for dm in distance_moduli:
            for filter_name in self.filter_names:
                matched_filter = self.matched_filters[filter_name]
                reuse = getattr(matched_filter, "distance_independent", False)
                if reuse and filter_name in distance_independent:
                    counts = distance_independent[filter_name]
                else:
                    selected = np.asarray(
                        matched_filter.select(detected, list(self.bands), dm),
                        dtype=bool,
                    )
                    counts = projection.counts(star_pixels[selected])
                    if reuse:
                        distance_independent[filter_name] = counts
                background = self.background.raw_map_full_dict[filter_name][dm][
                    projection.pixnums
                ]
                maps.append(
                    projection.image(np.where(valid_at, background + counts, 0.0))
                )
                raw_labels.append(projection.image(counts))
                channels_meta.append({"filter": filter_name, "distance_modulus": dm})
        return maps, raw_labels, projection.valid, channels_meta

    def inject_stream_full_sky(self, params: dict, rng: np.random.Generator) -> dict:
        """Inject a stream and return FULL-SKY channel maps, uncropped.

        `inject_single_stream` crops to one window, which is what training
        wants. This returns the same channels as whole-sky HEALPix maps
        instead, which is what *inference* wants: tile a region with
        `windows.tile_footprint`, crop each tile, run the model, and put the
        predictions back on the sky with
        `matched_filter.stitch_windows_to_healpix`. It is the simulated
        stand-in for a real survey map, where the map exists first and
        windows are chosen afterwards.

        No window is sampled here, so `rng` is consumed only by realization,
        placement and the survey model.

        Parameters:
            params: stream parameter dict, as `inject_single_stream`.
            rng: np.random.Generator.

        Returns:
            dict with:

            - ``map_full``: list of full-sky combined (background+stream)
              maps, one per channel;
            - ``stream_raw_full``: list of full-sky stream-only count maps,
              **unthresholded** -- apply `count_threshold` after cropping,
              since cropping interpolates;
            - ``valid_mask_full``: the background's full-sky validity mask;
            - ``channels``: the `{"filter", "distance_modulus"}` mapping, in
              channel order;
            - ``params``: the resolved parameter dict (richness -> nstars);
            - ``placement``: ``{"center_ra", "center_dec", "rotation_deg"}``,
              the stream frame on the sky.
        """
        detected, resolved_params, placement = self._realize_and_inject(params, rng)
        finalized_full, stream_raw_full, channels_meta = self._build_full_sky_channels(
            detected
        )
        return {
            "map_full": finalized_full,
            "stream_raw_full": stream_raw_full,
            "valid_mask_full": self.background.valid_mask_full,
            "channels": channels_meta,
            "params": resolved_params,
            "placement": placement,
        }

    def inject_streams_full_sky(
        self,
        params_list: list[dict],
        rng: np.random.Generator,
        centers: list[tuple[float, float] | None] | None = None,
    ) -> dict:
        """Inject several streams on one sky, return full-sky channel maps.

        For evaluation, not training: training injects one stream per window.
        It answers questions like "does a second stream nearby confuse the
        model?", which need streams at chosen positions relative to each other
        -- hence ``centers`` (and each stream's ``orientation`` parameter).
        The streams are realized and observed one after the other from
        ``rng``, then their detected stars are pooled before the maps are
        built, so the stream-only maps and the label cover all of them.

        Streams are processed in list order, each drawing from ``rng`` in the
        same way `inject_stream_full_sky` does. So with a fixed seed and a
        fixed center, the first stream is identical whether or not others
        follow it: a paired "alone" vs "with a neighbour" comparison.

        Parameters:
            params_list: one stream parameter dict per stream, as
                `inject_single_stream`.
            rng: np.random.Generator.
            centers: ``(ra, dec)`` per stream, or None (random position) for
                any of them; None for all random.

        Returns:
            dict with ``map_full``, ``stream_raw_full``, ``valid_mask_full``
            and ``channels`` as `inject_stream_full_sky`, and ``params`` /
            ``placement`` as lists, one entry per stream.
        """
        if centers is None:
            centers = [None] * len(params_list)
        detected, resolved, placements = [], [], []
        for params, center in zip(params_list, centers, strict=True):
            stars, resolved_params, placement = self._realize_and_inject(
                params, rng, center=center
            )
            detected.append(stars)
            resolved.append(resolved_params)
            placements.append(placement)
        finalized_full, stream_raw_full, channels_meta = self._build_full_sky_channels(
            pd.concat(detected, ignore_index=True)
        )
        return {
            "map_full": finalized_full,
            "stream_raw_full": stream_raw_full,
            "valid_mask_full": self.background.valid_mask_full,
            "channels": channels_meta,
            "params": resolved,
            "placement": placements,
        }

    def inject_single_stream(
        self,
        params: dict,
        rng: np.random.Generator,
        min_stream_length_deg: float | None = None,
        max_attempts: int = 100,
        max_placements: int = 10,
    ) -> Sample:
        """Realize one stream, inject it into the background, return a
        labeled sample.

        Pipeline: see module docstring. The window is sampled here (not
        passed in) because it depends on where the stream ends up, which
        isn't known until after placement (step 2) -- for a fixed window,
        use inject_background_only() (stream-independent placement).

        Parameters:
            params: Stream parameter dict (morphology, width, length,
                distance_modulus, age, z, richness or nstars, etc. -- see
                stream_sources.StreamSource.realize()). `band_1`/`band_2`
                default to `self.bands` if not given. `orientation`, if
                present, is forwarded to `place_stream_in_footprint` as the
                placement's position angle (config/streams/injection_grid.yaml's
                free `orientation` parameter) instead of drawing a random one;
                the placement *position* itself is always random regardless
                ("position: uniform_in_footprint" is a sentinel, never tunable).
            rng: Random number generator.
            min_stream_length_deg: Forwarded to windows.sample_stream_window;
                None uses the injector's own ``min_stream_length_deg``
                (5 deg unless configured).
            max_attempts: Forwarded to windows.sample_stream_window.
            max_placements: How many times to realize and place the stream
                again when a placement admits no valid window (the stream
                landed where no window centred in the footprint holds
                ``min_stream_length_deg`` of it, e.g. along the footprint
                edge) or no star of it was detected at all. Each retry
                continues the same ``rng``, so a sample is still
                reproducible, and a first placement that works is used as
                before.

        Returns:
            Sample(map_stack, label_stack, valid_mask, params, metadata).
            metadata includes the sampled window and the distance moduli
            list (matching map_stack's channel order).

        Raises:
            RuntimeError if no valid window is found in ``max_placements``
                placements (e.g. the footprint is too small or sparse for a
                >=min_stream_length_deg window).
            NotImplementedError if self.label_policy is "soft_distance"
                (rasterize.py, decision 6 -- still a stub).
        """
        if min_stream_length_deg is None:
            min_stream_length_deg = self.min_stream_length_deg
        size_deg = self.pix.image_size_pix[0] * self.pix.pixel_scale_deg
        failures = []
        for _ in range(max_placements):
            detected, resolved_params, _ = self._realize_and_inject(params, rng)
            if detected.empty:
                failures.append("no star of the stream was detected")
                continue
            try:
                window = sample_stream_window(
                    detected["ra"].to_numpy(dtype=float),
                    detected["dec"].to_numpy(dtype=float),
                    resolved_params.get("width", 0.0),
                    self.background.footprint,
                    self.pix.nside,
                    size_deg=size_deg,
                    min_stream_length_deg=min_stream_length_deg,
                    max_attempts=max_attempts,
                    rng=rng,
                )
                break
            except RuntimeError as error:
                failures.append(str(error))
        else:
            raise RuntimeError(
                f"inject_single_stream: no valid window in {max_placements} "
                f"placements of the stream; last failure: {failures[-1]}"
            )

        use_channelwise_label = self.label_policy in (
            "stream_count",
            "stream_detection",
        )
        finalize = bool(self.finalize_cfg and self.finalize_cfg.get("enabled"))
        if finalize:
            # Finalization works on the full sky (and is not implemented yet:
            # finalize_full raises), so it keeps the full-sky path.
            finalized_full, stream_raw_full, channels_meta = (
                self._build_full_sky_channels(detected)
            )
            map_channels, raw_labels, valid_mask = [], [], None
            for index in range(len(channels_meta)):
                windowed_map, windowed_valid = crop_window(
                    finalized_full[index],
                    self.background.valid_mask_full,
                    window,
                    self.pix,
                )
                map_channels.append(windowed_map)
                valid_mask = windowed_valid if valid_mask is None else valid_mask
                raw_labels.append(
                    crop_window(
                        stream_raw_full[index],
                        self.background.valid_mask_full,
                        window,
                        self.pix,
                    )[0]
                )
        else:
            map_channels, raw_labels, valid_mask, channels_meta = (
                self._build_window_channels(detected, window)
            )
        distance_moduli = sorted({c["distance_modulus"] for c in channels_meta})

        label_channels = []
        if use_channelwise_label:
            for windowed_label in raw_labels:
                # Threshold AFTER cropping, never before: cropping
                # interpolates, so thresholding first would produce a binary
                # map that interpolation then turned back into fractional
                # values -- a label that is no longer {0, 1} at all.
                if self.label_policy == "stream_detection":
                    windowed_label = (windowed_label > self.count_threshold).astype(
                        windowed_label.dtype
                    )
                label_channels.append(windowed_label)

        if use_channelwise_label:
            label_stack = np.stack(label_channels, axis=0)
        else:
            label_2d = rasterize.rasterize(
                stream_members_ra=detected["ra"].to_numpy(dtype=float),
                stream_members_dec=detected["dec"].to_numpy(dtype=float),
                stream_width_deg=resolved_params.get("width", 0.0),
                params=resolved_params,
                policy=self.label_policy,
                window=window,
                pix=self.pix,
                dilate_to_width=self.label_config.get("dilate_to_width", False),
                normalization=self.label_config.get("normalization", "max"),
                smooth_sigma_deg=self.label_config.get("smooth_sigma_deg"),
            )
            label_stack = np.broadcast_to(
                label_2d, (len(map_channels), *label_2d.shape)
            )

        return Sample(
            map_stack=np.stack(map_channels, axis=0).astype(np.float32),
            label_stack=np.asarray(label_stack, dtype=np.float32),
            valid_mask=valid_mask,
            params=resolved_params,
            metadata={
                "window": dataclasses.asdict(window),
                "distance_moduli": distance_moduli,
                "channels": channels_meta,
            },
        )

    def inject_population(
        self, params_list: list[dict], rng: np.random.Generator
    ) -> Sample:
        """Inject multiple streams into one window (stub, decision 19).

        Combines single-stream injection results:
        - Sum raw stream maps (additive; valid because unweighted).
        - Union labels (both sets of rasterized members).
        - Return combined Sample.

        Parameters:
            params_list: List of parameter dicts (one per stream).
            rng: Random number generator.

        Returns:
            Sample with summed maps and unioned labels.

        Raises:
            NotImplementedError (Stage 1 stub; implement in Stage 2b after
                inject_single_stream is validated, once rasterize.py exists).
        """
        raise NotImplementedError(
            "inject_population is a second-stage feature (decision 19) -- "
            "not implemented yet."
        )


def inject_background_only(
    background: Background, window: Window, pix: PixelizationSpec
) -> Sample:
    """Generate a pure-background (no stream) sample.

    Parameters:
        background: Background instance.
        window: Window to crop to.
        pix: PixelizationSpec.

    Returns:
        Sample with map_stack from background (its own cached finalized map
        if available, else the raw map -- both already match whatever
        finalize_cfg background was cached with), label_stack all zeros,
        params={} (no stream parameters apply).

    Rationale: Teaches network not to hallucinate streams in every field.
    Used when background_fraction > 0 in config. Unlike inject_single_stream,
    this needs no stream realization/injection/rasterization at all, so it
    has no dependency on rasterize.py. Produces the same (filter, distance)
    channel count/order as inject_single_stream (distance-major,
    filter-minor, from `background.raw_map_full_dict`'s own filter names and
    insertion order -- the same order `Background.load_or_cache` built it
    in), so background-only and stream samples stay shape-compatible.
    """
    filter_names = list(background.raw_map_full_dict)
    distance_moduli = sorted(background.raw_map_full_dict[filter_names[0]])
    map_channels = []
    channels_meta = []
    valid_mask = None
    for dm in distance_moduli:
        for filter_name in filter_names:
            chosen = background.finalized_map_full_dict.get(filter_name, {}).get(dm)
            if chosen is None:
                chosen = background.raw_map_full_dict[filter_name][dm]
            windowed_map, windowed_valid = crop_window(
                chosen, background.valid_mask_full, window, pix
            )
            map_channels.append(windowed_map)
            if valid_mask is None:
                valid_mask = windowed_valid
            channels_meta.append({"filter": filter_name, "distance_modulus": dm})

    map_stack = np.stack(map_channels, axis=0).astype(np.float32)
    return Sample(
        map_stack=map_stack,
        label_stack=np.zeros_like(map_stack, dtype=np.float32),
        valid_mask=valid_mask,
        params={},
        metadata={
            "window": dataclasses.asdict(window),
            "distance_moduli": distance_moduli,
            "channels": channels_meta,
        },
    )
