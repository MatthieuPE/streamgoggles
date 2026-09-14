"""Test stream injection: resolve_richness_to_nstars, place_stream_in_footprint,
StreamInjector.inject_single_stream, and inject_background_only.

Everything runs against real streamobs (survey loading, StreamInjector,
StreamModel via stream_sources.py) -- no mocking. StreamInjector now works
with a *named dict* of matched filters (2026-09-09 pivot: typically a real
isochrone filter plus one or more deliberately "bad" decoy filters --
matched_filter.ShiftedColorBoxFilter -- per trial distance), and its default
label ("stream_count") is the true stream-only raw count per (filter,
distance) channel, built directly here rather than via rasterize.py.
rasterize.py-based labels (binary/density/soft_distance) are still reachable
via label_policy and remain tested; only "soft_distance" is still a stub.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from streamgoggles import inject_utils
from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.data_preparation import Cut
from streamgoggles.injector import (
    StreamInjector,
    inject_background_only,
    place_stream_in_footprint,
    resolve_richness_to_nstars,
)
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.storage import BackgroundMapStore
from streamgoggles.stream_sources import StreamObsSource
from streamgoggles.windows import sample_random_window

pytestmark = pytest.mark.injector

# ---------------------------------------------------------------------------
# resolve_richness_to_nstars
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_params():
    return {
        "morphology": "uniform",
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }


def test_resolve_richness_nstars_passthrough(uniform_params):
    params = dict(uniform_params, richness=2500)
    out = resolve_richness_to_nstars(params, "nstars")
    assert out["nstars"] == 2500
    assert "richness" not in out


def test_resolve_richness_no_richness_key_is_noop(uniform_params):
    out = resolve_richness_to_nstars(uniform_params, "nstars")
    assert out == uniform_params


def test_resolve_richness_mass_matches_inject_utils(uniform_params):
    params = dict(uniform_params, richness=1000.0)
    out = resolve_richness_to_nstars(params, "mass")

    isochrone_params = {
        "isochrone": {
            "name": "Marigo2017",
            "survey": "lsst",
            "age": 12.5,
            "z": 0.0002,
            "band_1": "g",
            "band_2": "r",
        }
    }
    expected_n = inject_utils.convert_Mass_to_N(
        1000.0, isochrone_params=isochrone_params
    )
    assert out["nstars"] == round(expected_n)


def test_resolve_richness_surface_brightness_uniform(uniform_params):
    params = dict(uniform_params, richness=30.0)
    out = resolve_richness_to_nstars(params, "surface_brightness")
    assert out["nstars"] > 0
    assert isinstance(out["nstars"], int)


def test_resolve_richness_surface_brightness_spline_derives_length():
    params = {
        "morphology": "spline",
        "control_points": [
            {"phi1": -6.0, "phi2": 0.0},
            {"phi1": 0.0, "phi2": 0.5},
            {"phi1": 6.0, "phi2": -0.2},
        ],
        "width": 0.2,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
        "richness": 30.0,
    }
    out = resolve_richness_to_nstars(params, "surface_brightness")
    assert out["nstars"] > 0


def test_resolve_richness_surface_brightness_requires_width(uniform_params):
    params = dict(uniform_params, richness=30.0)
    del params["width"]
    with pytest.raises(ValueError, match="width"):
        resolve_richness_to_nstars(params, "surface_brightness")


def test_resolve_richness_unknown_kind_raises(uniform_params):
    params = dict(uniform_params, richness=30.0)
    with pytest.raises(ValueError, match="richness_kind"):
        resolve_richness_to_nstars(params, "luminosity")


def test_resolve_richness_larger_mass_gives_more_stars(uniform_params):
    small = resolve_richness_to_nstars(dict(uniform_params, richness=100.0), "mass")
    large = resolve_richness_to_nstars(dict(uniform_params, richness=10000.0), "mass")
    assert large["nstars"] > small["nstars"]


def test_resolve_richness_brighter_surface_brightness_gives_more_stars(uniform_params):
    # Surface brightness in mag/arcsec^2: SMALLER value = brighter = more stars.
    faint = resolve_richness_to_nstars(
        dict(uniform_params, richness=32.0), "surface_brightness"
    )
    bright = resolve_richness_to_nstars(
        dict(uniform_params, richness=26.0), "surface_brightness"
    )
    assert bright["nstars"] > faint["nstars"]


# ---------------------------------------------------------------------------
# place_stream_in_footprint
# ---------------------------------------------------------------------------


@pytest.fixture
def small_footprint_and_nside():
    nside = 64
    npix = 12 * nside**2
    footprint = np.zeros(npix, dtype=bool)
    # A modest disk of valid pixels around (ra=10, dec=-30).
    import healpy as hp

    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    center = SkyCoord(ra=10 * u.deg, dec=-30 * u.deg)
    sep = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).separation(center).deg
    footprint[sep < 5.0] = True
    return footprint, nside


def test_place_stream_in_footprint_adds_radec(small_footprint_and_nside):
    footprint, nside = small_footprint_and_nside
    stream_df = StreamObsSource().realize(
        {
            "morphology": "uniform",
            "nstars": 200,
            "width": 0.2,
            "length": 4.0,
            "distance_modulus": 16.8,
            "age": 12.5,
            "z": 0.0002,
        },
        np.random.default_rng(0),
    )
    out = place_stream_in_footprint(
        stream_df, footprint, nside, np.random.default_rng(0)
    )
    assert "ra" in out.columns and "dec" in out.columns
    assert len(out) == len(stream_df)
    assert np.isfinite(out["ra"]).all() and np.isfinite(out["dec"]).all()


def test_place_stream_in_footprint_reproducible_with_same_seed(
    small_footprint_and_nside,
):
    footprint, nside = small_footprint_and_nside
    stream_df = StreamObsSource().realize(
        {
            "morphology": "uniform",
            "nstars": 50,
            "width": 0.2,
            "length": 4.0,
            "distance_modulus": 16.8,
            "age": 12.5,
            "z": 0.0002,
        },
        np.random.default_rng(0),
    )
    out1 = place_stream_in_footprint(
        stream_df, footprint, nside, np.random.default_rng(5)
    )
    out2 = place_stream_in_footprint(
        stream_df, footprint, nside, np.random.default_rng(5)
    )
    np.testing.assert_array_equal(out1["ra"].to_numpy(), out2["ra"].to_numpy())
    np.testing.assert_array_equal(out1["dec"].to_numpy(), out2["dec"].to_numpy())


def test_place_stream_in_footprint_respects_explicit_rotation():
    """config/streams/injection_grid.yaml's free 'orientation' parameter must
    actually control placement -- the on-sky position angle of the phi1
    axis -- not just get silently ignored in favor of a random draw."""
    import healpy as hp
    import pandas as pd

    nside = 32
    footprint = np.zeros(hp.nside2npix(nside), dtype=bool)
    footprint[100] = True  # single valid pixel -> deterministic placement position
    stream_df = pd.DataFrame({"phi1": [0.0, 5.0], "phi2": [0.0, 0.0]})

    out_0 = place_stream_in_footprint(
        stream_df, footprint, nside, np.random.default_rng(1), rotation_deg=0.0
    )
    out_90 = place_stream_in_footprint(
        stream_df, footprint, nside, np.random.default_rng(2), rotation_deg=90.0
    )

    # The origin (phi1=0, phi2=0) lands at the same ra/dec regardless of
    # rotation -- rotation only affects points away from the origin.
    np.testing.assert_allclose(
        out_0[["ra", "dec"]].iloc[0].to_numpy(),
        out_90[["ra", "dec"]].iloc[0].to_numpy(),
        atol=1e-8,
    )
    # An off-axis point must differ between the two explicit rotations.
    assert not np.allclose(
        out_0[["ra", "dec"]].iloc[1].to_numpy(),
        out_90[["ra", "dec"]].iloc[1].to_numpy(),
    )


def test_place_stream_in_footprint_empty_footprint_raises(small_footprint_and_nside):
    _footprint, nside = small_footprint_and_nside
    import healpy as hp

    empty = np.zeros(hp.nside2npix(nside), dtype=bool)
    stream_df = StreamObsSource().realize(
        {
            "morphology": "uniform",
            "nstars": 10,
            "width": 0.2,
            "length": 4.0,
            "distance_modulus": 16.8,
            "age": 12.5,
            "z": 0.0002,
        },
        np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="footprint"):
        place_stream_in_footprint(stream_df, empty, nside, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# StreamInjector.inject_single_stream / inject_background_only
# (real streamobs, real background, real cuts, real multi-filter channels)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_background(tmp_path_factory):
    """Two named filters ("good" isochrone + "decoy" shifted color box) x two
    trial distances -> 4 channels, matching the real, current design."""
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(20, 20))
    store = BackgroundMapStore(tmp_path_factory.mktemp("bgmaps"))
    good = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    decoy = ShiftedColorBoxFilter(reference_filter=good, color_shift=0.5)
    filters = {"good": good, "decoy": decoy}
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=[16.8, 17.5],
        finalize_cfg=None,
        store=store,
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_configs={"good": {"age": 12.5, "z": 0.0002}, "decoy": {"shift": 0.5}},
    )
    return bg, filters, pix


@pytest.fixture
def real_injector(real_background):
    bg, filters, pix = real_background
    return StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        bands=("g", "r"),
        richness_kind="nstars",
        finalize_cfg=None,
    )


@pytest.fixture
def stream_params():
    return {
        "morphology": "uniform",
        "nstars": 3000,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }


class _CountingFilter:
    """Wraps a MatchedFilter, counting select() calls -- used to prove the
    matched filter is applied to the background exactly once (at
    Background.load_or_cache time) and never re-run per stream injection."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def select(self, catalog, bands, distance_modulus):
        self.calls += 1
        return self.inner.select(catalog, bands, distance_modulus)


def test_inject_single_stream_builds_valid_sample(real_injector, stream_params):
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))

    assert sample.map_stack.shape == (4, 20, 20)  # 2 distances x 2 filters
    assert sample.map_stack.dtype == np.float32
    assert sample.label_stack.shape == (4, 20, 20)
    assert sample.label_stack.dtype == np.float32
    assert sample.valid_mask.shape == (20, 20)
    assert sample.valid_mask.dtype == bool
    assert sample.params["nstars"] == 3000
    assert "window" in sample.metadata
    assert sample.metadata["distance_moduli"] == [16.8, 17.5]
    assert sample.metadata["channels"] == [
        {"filter": "good", "distance_modulus": 16.8},
        {"filter": "decoy", "distance_modulus": 16.8},
        {"filter": "good", "distance_modulus": 17.5},
        {"filter": "decoy", "distance_modulus": 17.5},
    ]


def test_inject_single_stream_map_has_signal(real_injector, stream_params):
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert sample.map_stack.sum() > 0


def test_inject_single_stream_stream_count_label_never_exceeds_map(
    real_injector, stream_params
):
    """The label is the stream-only contribution to each channel's combined
    map -- it can never exceed what the combined map itself shows there."""
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert sample.label_stack.sum() > 0
    assert (sample.label_stack <= sample.map_stack + 1e-4).all()


def test_inject_single_stream_good_filter_label_beats_decoy_at_true_distance(
    real_injector, stream_params
):
    """The whole point of a decoy filter: at the stream's own true distance,
    the real isochrone filter's label count must be well above the decoy's."""
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    channels = sample.metadata["channels"]
    good_idx = channels.index({"filter": "good", "distance_modulus": 16.8})
    decoy_idx = channels.index({"filter": "decoy", "distance_modulus": 16.8})
    assert sample.label_stack[good_idx].sum() > sample.label_stack[decoy_idx].sum()


def test_inject_single_stream_label_lower_at_larger_trial_distance(
    real_injector, stream_params
):
    """The "good" filter's label total must drop as the trial distance
    modulus moves further past the stream's own true value (16.8): the
    isochrone locus shifts away from where the true (fixed-distance) stream
    actually sits in color-magnitude space, so fewer true members fall
    inside an increasingly mismatched filter."""
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    channels = sample.metadata["channels"]
    at_true_dm = channels.index({"filter": "good", "distance_modulus": 16.8})
    at_larger_dm = channels.index({"filter": "good", "distance_modulus": 17.5})
    assert sample.label_stack[at_larger_dm].sum() < sample.label_stack[at_true_dm].sum()


def test_inject_single_stream_label_excludes_members_outside_window(real_background):
    """The label must only ever reflect what's inside the sampled window
    (decision 21: partial-stream inclusion is expected -- a stream much
    longer than the window WILL have most of its members outside it). Same
    nstars, spread over a much longer track: if members outside the window
    still counted, the label total wouldn't meaningfully drop; since
    map_stack/label_stack are cropped to the window's own fixed pixel grid
    (matched_filter.crop_window/project, already proven not to leak in
    tests/test_rasterize.py's boundary tests), it should drop substantially.
    """
    bg, filters, pix = real_background
    injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    short_params = {
        "morphology": "uniform",
        "nstars": 3000,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }
    long_params = dict(short_params, length=100.0)  # far longer than the ~20 deg window

    sample_short = injector.inject_single_stream(
        short_params, np.random.default_rng(99)
    )
    sample_long = injector.inject_single_stream(long_params, np.random.default_rng(99))

    idx_short = sample_short.metadata["channels"].index(
        {"filter": "good", "distance_modulus": 16.8}
    )
    idx_long = sample_long.metadata["channels"].index(
        {"filter": "good", "distance_modulus": 16.8}
    )
    assert (
        sample_long.label_stack[idx_long].sum()
        < sample_short.label_stack[idx_short].sum()
    )


def test_inject_single_stream_density_policy_label_is_broadcast_across_channels(
    real_background, stream_params
):
    """density doesn't depend on distance or filter, so the same 2D label
    must be broadcast identically to every one of the 4 channels."""
    bg, filters, pix = real_background
    injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="density",
    )
    sample = injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert sample.label_stack.sum() > 0
    for c in range(1, sample.label_stack.shape[0]):
        np.testing.assert_array_equal(sample.label_stack[0], sample.label_stack[c])


def test_inject_single_stream_binary_policy_label_is_binary(
    real_background, stream_params
):
    bg, filters, pix = real_background
    injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="binary",
        label_config={"dilate_to_width": True},
    )
    sample = injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert set(np.unique(sample.label_stack)) <= {0.0, 1.0}
    assert sample.label_stack.sum() > 0


def test_inject_single_stream_reuses_background_selection_across_injections(
    real_background, stream_params
):
    """The matched filter must be applied to the background exactly once (at
    Background.load_or_cache time, already covered by
    test_background.py::test_load_or_cache_reuses_cached_maps_without_recompute)
    and reused unchanged across every stream injection -- never re-run per
    stream realization. select() is only ever expected to fire on the
    (much smaller) stream catalog, once per distance modulus per injection,
    per filter it's wrapping.
    """
    bg, filters, pix = real_background
    counting_good = _CountingFilter(filters["good"])
    injector = StreamInjector(
        background=bg,
        matched_filters={"good": counting_good, "decoy": filters["decoy"]},
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    n_distances = len(bg.raw_map_full_dict["good"])
    background_raw_before = {
        filter_name: {dm: arr.copy() for dm, arr in per_dm.items()}
        for filter_name, per_dm in bg.raw_map_full_dict.items()
    }

    injector.inject_single_stream(stream_params, np.random.default_rng(11))
    assert counting_good.calls == n_distances

    injector.inject_single_stream(stream_params, np.random.default_rng(12))
    assert counting_good.calls == 2 * n_distances

    injector.inject_single_stream(stream_params, np.random.default_rng(13))
    assert counting_good.calls == 3 * n_distances

    # The background's own cached raw maps must be byte-for-byte untouched
    # across all three stream injections -- proof nothing recomputed them.
    for filter_name, per_dm in background_raw_before.items():
        for dm, arr in per_dm.items():
            np.testing.assert_array_equal(bg.raw_map_full_dict[filter_name][dm], arr)


def test_inject_single_stream_richness_resolved_before_realize(
    real_injector, stream_params
):
    params = dict(stream_params)
    del params["nstars"]
    params["richness"] = 3000
    real_injector.richness_kind = "nstars"
    sample = real_injector.inject_single_stream(params, np.random.default_rng(3))
    assert sample.params["nstars"] == 3000
    assert "richness" not in sample.params


def test_inject_single_stream_forwards_orientation_param(real_injector, stream_params):
    """config/streams/injection_grid.yaml's free 'orientation' parameter must
    flow through to placement (place_stream_in_footprint's rotation_deg),
    not get silently dropped -- precise geometric effect is unit-tested
    directly on place_stream_in_footprint; this just confirms the full
    pipeline accepts and preserves it end to end."""
    params = dict(stream_params, orientation=45.0)
    sample = real_injector.inject_single_stream(params, np.random.default_rng(3))
    assert sample.params["orientation"] == 45.0


def test_inject_single_stream_strict_cuts_reduce_signal(real_background, stream_params):
    bg, filters, pix = real_background
    lenient = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    strict = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[Cut(quantity="mag", band="g", op="<", value=18.0)],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )

    sample_lenient = lenient.inject_single_stream(
        stream_params, np.random.default_rng(9)
    )
    sample_strict = strict.inject_single_stream(stream_params, np.random.default_rng(9))

    assert sample_strict.map_stack.sum() < sample_lenient.map_stack.sum()


def test_inject_single_stream_soft_distance_policy_raises_not_implemented(
    real_background, stream_params
):
    """Documents the current boundary: stream_count/binary/density labels
    are fully real; only soft_distance (rasterize.py, decision 6) is still a
    stub, and everything up through crop_window already ran for real before
    it does."""
    bg, filters, pix = real_background
    injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="soft_distance",
    )
    with pytest.raises(NotImplementedError):
        injector.inject_single_stream(stream_params, np.random.default_rng(3))


def test_inject_background_only_builds_valid_sample(real_background):
    bg, _filters, pix = real_background
    window = sample_random_window(
        bg.footprint, pix.nside, size_deg=20.0, rng=np.random.default_rng(4)
    )
    sample = inject_background_only(bg, window, pix)

    assert sample.map_stack.shape == (4, 20, 20)
    assert sample.map_stack.dtype == np.float32
    assert np.all(sample.label_stack == 0.0)
    assert sample.params == {}
    assert sample.metadata["distance_moduli"] == [16.8, 17.5]
    assert sample.metadata["channels"] == [
        {"filter": "good", "distance_modulus": 16.8},
        {"filter": "decoy", "distance_modulus": 16.8},
        {"filter": "good", "distance_modulus": 17.5},
        {"filter": "decoy", "distance_modulus": 17.5},
    ]


def test_inject_background_only_matches_manual_crop(real_background):
    from streamgoggles.matched_filter import crop_window

    bg, _filters, pix = real_background
    window = sample_random_window(
        bg.footprint, pix.nside, size_deg=20.0, rng=np.random.default_rng(4)
    )
    sample = inject_background_only(bg, window, pix)

    filter_name = next(iter(bg.raw_map_full_dict))
    dm = min(bg.raw_map_full_dict[filter_name])
    expected_map, expected_valid = crop_window(
        bg.raw_map_full_dict[filter_name][dm], bg.valid_mask_full, window, pix
    )
    np.testing.assert_allclose(sample.map_stack[0], expected_map, atol=1e-4)
    np.testing.assert_array_equal(sample.valid_mask, expected_valid)


# ---------------------------------------------------------------------------
# StreamInjector.__init__
# ---------------------------------------------------------------------------


def test_injector_namespace_derivation(real_background):
    bg, filters, pix = real_background
    inj = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    assert inj.namespace == "lsst_yr1"


def test_injector_default_bands(real_background):
    # release must still name a real streamobs survey config (only "yr1" is
    # available in this environment); only `bands` is left at its default here.
    bg, filters, pix = real_background
    inj = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    assert inj.bands == ("g", "r")


def test_injector_filter_names_preserve_dict_order(real_background):
    bg, filters, pix = real_background
    inj = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    assert inj.filter_names == ["good", "decoy"]
