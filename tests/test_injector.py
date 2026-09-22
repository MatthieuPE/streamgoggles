"""Test stream injection: resolve_richness_to_nstars, place_stream_in_footprint,
StreamInjector.inject_single_stream, and inject_background_only.

Everything runs against real streamobs (survey loading, StreamInjector,
StreamModel via stream_sources.py) -- no mocking. StreamInjector now works
with a *named dict* of matched filters (2026-09-09 pivot: typically a real
isochrone filter plus one or more deliberately "bad" decoy filters --
matched_filter.ShiftedColorBoxFilter -- per trial distance), and its default
label ("stream_count") is the true stream-only raw count per (filter,
distance) channel, built directly here rather than via rasterize.py.
"stream_detection" (2026-09-15, PLAN.md section 6.12) hard-thresholds that
same per-channel count into a binary {0, 1} detection target instead --
retargeting Stage 1 to the actual detection goal rather than trying to fix
stream_count's amplitude-under-recovery problem. rasterize.py-based labels
(binary/density/soft_distance) are still reachable via label_policy and
remain tested; only "soft_distance" is still a stub.
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
    crop_window,
)
from streamgoggles.storage import BackgroundMapStore
from streamgoggles.stream_sources import StreamObsSource
from streamgoggles.windows import Window, sample_random_window

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


def test_inject_single_stream_places_the_stream_again_when_no_window_fits(
    monkeypatch, real_injector, stream_params
):
    """A placement along the footprint edge can admit no window holding
    5 deg of stream. That used to raise inside a DataLoader worker and stop a
    whole training run; the stream is now placed again instead."""
    from streamgoggles import injector as injector_module

    real_sampler = injector_module.sample_stream_window
    calls = {"placements": 0}
    real_realize = real_injector._realize_and_inject

    def counting_realize(params, rng):
        calls["placements"] += 1
        return real_realize(params, rng)

    def failing_twice(*args, **kwargs):
        if calls["placements"] <= 2:
            raise RuntimeError("sample_stream_window: no valid window found")
        return real_sampler(*args, **kwargs)

    monkeypatch.setattr(real_injector, "_realize_and_inject", counting_realize)
    monkeypatch.setattr(injector_module, "sample_stream_window", failing_twice)

    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert calls["placements"] == 3
    assert sample.map_stack.sum() > 0


def test_inject_single_stream_gives_up_after_max_placements(
    monkeypatch, real_injector, stream_params
):
    from streamgoggles import injector as injector_module

    def always_failing(*args, **kwargs):
        raise RuntimeError("sample_stream_window: no valid window found")

    monkeypatch.setattr(injector_module, "sample_stream_window", always_failing)
    with pytest.raises(RuntimeError, match="no valid window in 3 placements"):
        real_injector.inject_single_stream(
            stream_params, np.random.default_rng(3), max_placements=3
        )


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

    # Asserted on label_stack (the STREAM-ONLY count), not map_stack.
    # map_stack is background + stream, and the background dominates it by
    # ~3 orders of magnitude; worse, the two runs get *different windows*
    # (sample_stream_window is driven by the detected stream stars, which
    # the strict cut changes), so comparing map_stack sums really compares
    # two unrelated background patches. Measured directly: strict < lenient
    # on map_stack came out True/False/False across three seeds -- a coin
    # flip that had nothing to do with cuts, and the source of a long-running
    # intermittent failure here (PLAN.md section 6.15). On the stream-only
    # label the effect is unambiguous: a g < 18 cut removes essentially
    # every stream star at this distance modulus.
    assert sample_lenient.label_stack.sum() > 0.0
    assert sample_strict.label_stack.sum() < sample_lenient.label_stack.sum()


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


def test_injector_default_count_threshold_is_one(real_background):
    """Default count_threshold=1.0 (PLAN.md section 6.13), calibrated
    empirically against nside=512 -- the lowest threshold that still
    detects the faintest streams in this project's working richness range
    (surface_brightness up to 35, where count_threshold>=2 already leaves
    the label entirely empty; see create_data.ipynb's calibration section
    (§4) for the full sweep)."""
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
        label_policy="stream_detection",
    )
    assert injector.count_threshold == 1.0


def test_inject_single_stream_detection_policy_label_is_binary(
    real_background, stream_params
):
    """label_policy="stream_detection" hard-thresholds the same per-channel
    stream-only raw count stream_count would return directly (PLAN.md
    section 6.12) -- the label itself must come out strictly binary."""
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
        label_policy="stream_detection",
        count_threshold=10.0,
    )
    sample = injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert set(np.unique(sample.label_stack)) <= {0.0, 1.0}
    assert sample.label_stack.sum() > 0
    assert sample.label_stack.dtype == np.float32


def test_inject_single_stream_detection_policy_matches_manual_threshold_of_count_policy(
    real_background, stream_params
):
    """The detection label must be exactly (stream_count label >
    count_threshold) -- same underlying stream_raw, same window/rng, just
    thresholded -- not some independently-recomputed quantity."""
    bg, filters, pix = real_background
    count_injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="stream_count",
    )
    detection_injector = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="stream_detection",
        count_threshold=10.0,
    )
    count_sample = count_injector.inject_single_stream(
        stream_params, np.random.default_rng(7)
    )
    detection_sample = detection_injector.inject_single_stream(
        stream_params, np.random.default_rng(7)
    )
    expected = (count_sample.label_stack > 10.0).astype(np.float32)
    np.testing.assert_array_equal(detection_sample.label_stack, expected)


def test_inject_single_stream_detection_policy_decoy_stays_below_threshold_at_true_distance(
    real_background, stream_params
):
    """The whole point of the decoy filter (2026-09-09 pivot) must still
    hold under the hard-thresholded detection label: a decoy channel should
    have far fewer (typically zero) detected pixels than the real filter's
    channel at the stream's own true distance."""
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
        label_policy="stream_detection",
        count_threshold=10.0,
    )
    sample = injector.inject_single_stream(stream_params, np.random.default_rng(3))
    channels = sample.metadata["channels"]
    good_idx = channels.index({"filter": "good", "distance_modulus": 16.8})
    decoy_idx = channels.index({"filter": "decoy", "distance_modulus": 16.8})
    assert sample.label_stack[good_idx].sum() > sample.label_stack[decoy_idx].sum()


def test_inject_single_stream_detection_policy_higher_threshold_detects_fewer_pixels(
    real_background, stream_params
):
    bg, filters, pix = real_background
    loose = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="stream_detection",
        count_threshold=1.0,
    )
    strict = StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="stream_detection",
        count_threshold=200.0,
    )
    loose_sample = loose.inject_single_stream(stream_params, np.random.default_rng(3))
    strict_sample = strict.inject_single_stream(stream_params, np.random.default_rng(3))
    assert strict_sample.label_stack.sum() < loose_sample.label_stack.sum()


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


def test_inject_stream_full_sky_matches_the_windowed_path(
    real_background, stream_params
):
    """inject_single_stream and inject_stream_full_sky share their
    realization and channel construction, so cropping the full-sky maps to
    the windowed path's own window must reproduce that path's channels. If
    the two ever drift apart, inference would silently be looking at
    different data from training."""
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
        label_policy="stream_detection",
        count_threshold=1.0,
    )

    sample = injector.inject_single_stream(stream_params, np.random.default_rng(5))
    full = injector.inject_stream_full_sky(stream_params, np.random.default_rng(5))

    assert full["channels"] == sample.metadata["channels"]
    assert full["params"]["nstars"] == sample.params["nstars"]

    window = Window(**sample.metadata["window"])
    for channel in range(len(full["channels"])):
        cropped, _ = crop_window(
            full["map_full"][channel], full["valid_mask_full"], window, pix
        )
        np.testing.assert_allclose(cropped, sample.map_stack[channel], rtol=1e-6)

        raw_cropped, _ = crop_window(
            full["stream_raw_full"][channel], full["valid_mask_full"], window, pix
        )
        expected = (raw_cropped > injector.count_threshold).astype(raw_cropped.dtype)
        np.testing.assert_allclose(expected, sample.label_stack[channel], rtol=1e-6)


def test_inject_stream_full_sky_returns_unthresholded_counts(
    real_background, stream_params
):
    """The stream-only maps must come back as raw counts, not a binary
    label: thresholding before cropping would be wrong (cropping
    interpolates, so a {0,1} map becomes fractional), so the caller has to
    be able to threshold afterwards."""
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
        label_policy="stream_detection",
        count_threshold=1.0,
    )

    full = injector.inject_stream_full_sky(stream_params, np.random.default_rng(5))
    stacked = np.concatenate([m for m in full["stream_raw_full"]])
    assert stacked.max() > 1.0, "raw counts should exceed a {0,1} range"
    assert not set(np.unique(stacked)) <= {0.0, 1.0}


def test_injector_realizes_streams_in_its_own_survey(real_injector, stream_params):
    """A stream's true magnitudes must be in the bands of the survey that then
    observes it. Left unset, the stream source falls back to LSST, so a DES
    injector would observe LSST-band stars (and streamobs refuses to).
    """
    seen = []

    class Recording(StreamObsSource):
        def realize(self, params, rng):
            seen.append(dict(params))
            return super().realize(params, rng)

    real_injector.stream_source = Recording()
    real_injector.inject_stream_full_sky(stream_params, np.random.default_rng(5))
    assert seen[-1]["survey"] == real_injector.survey
    assert seen[-1]["release"] == real_injector.release

    # An explicit choice in the stream parameters is kept.
    real_injector.inject_stream_full_sky(
        {**stream_params, "survey": "lsst", "release": "yr1"},
        np.random.default_rng(5),
    )
    assert (seen[-1]["survey"], seen[-1]["release"]) == ("lsst", "yr1")


def test_place_stream_at_a_given_center():
    import healpy as hp
    import pandas as pd

    nside = 64
    footprint = np.zeros(hp.nside2npix(nside), dtype=bool)
    footprint[hp.ang2pix(nside, 10.0, -40.0, lonlat=True)] = True
    stream = pd.DataFrame({"phi1": [0.0, 1.0], "phi2": [0.0, 0.0]})

    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    placed = place_stream_in_footprint(
        stream, footprint, nside, rng, rotation_deg=30.0, center=(12.0, -35.0)
    )
    # The given center is the frame origin, and no random position is drawn.
    assert placed.attrs["placement"]["center_ra"] == 12.0
    assert placed.attrs["placement"]["center_dec"] == -35.0
    assert placed.loc[0, "ra"] == pytest.approx(12.0)
    assert placed.loc[0, "dec"] == pytest.approx(-35.0)
    assert rng.bit_generator.state == before


def test_inject_streams_full_sky_keeps_the_first_stream_identical(
    real_injector, stream_params
):
    """The paired design: stream A alone and stream A with a neighbour B must
    share A exactly, so any change in A's detection is caused by B."""
    import healpy as hp

    footprint = np.flatnonzero(real_injector.background.footprint)
    ra, dec = hp.pix2ang(
        real_injector.pix.nside, int(footprint[len(footprint) // 2]), lonlat=True
    )
    a = {**stream_params, "orientation": 20.0}
    b = {**stream_params, "orientation": 20.0}
    alone = real_injector.inject_streams_full_sky(
        [a], np.random.default_rng(7), centers=[(ra, dec)]
    )
    paired = real_injector.inject_streams_full_sky(
        [a, b], np.random.default_rng(7), centers=[(ra, dec), (ra, dec + 2.0)]
    )

    assert paired["placement"][0] == alone["placement"][0]
    assert paired["params"][0] == alone["params"][0]
    assert len(paired["placement"]) == 2
    # B only adds stars: the pooled stream-only map is A's plus B's.
    for with_b, a_only in zip(
        paired["stream_raw_full"], alone["stream_raw_full"], strict=True
    ):
        assert np.all(with_b >= a_only - 1e-9)
    assert sum(m.sum() for m in paired["stream_raw_full"]) > sum(
        m.sum() for m in alone["stream_raw_full"]
    )


# ---------------------------------------------------------------------------
# The label follows the matched filter's trial distance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def distance_background(tmp_path_factory):
    """The isochrone filter alone, at four trial distances: 15, 16, 18, 19."""
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(20, 20))
    good = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={"seed": 3},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters={"good": good},
        bands=["g", "r"],
        distance_moduli=[15.0, 16.0, 18.0, 19.0],
        finalize_cfg=None,
        store=BackgroundMapStore(tmp_path_factory.mktemp("distance_bgmaps")),
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_configs={"good": {"age": 12.5, "z": 0.0002}},
    )
    return bg, {"good": good}, pix


def _distance_injector(distance_background, stream_source):
    bg, filters, pix = distance_background
    return StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=stream_source,
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        bands=("g", "r"),
        richness_kind="nstars",
        label_policy="stream_detection",
        count_threshold=1.0,
    )


def _label_pixels(full, distance, count_threshold=1.0):
    """HEALPix pixels labelled as stream by the filter at `distance`."""
    index = [ch["distance_modulus"] for ch in full["channels"]].index(distance)
    return np.flatnonzero(full["stream_raw_full"][index] > count_threshold)


def test_label_is_almost_empty_when_the_filter_is_at_the_wrong_distance(
    distance_background,
):
    """Full data generation for a stream at distance modulus 15: the label
    built with the matched filter at 19 must be almost empty, since the filter
    selects stars 4 magnitudes fainter than the stream's."""
    injector = _distance_injector(distance_background, StreamObsSource())
    params = {
        "morphology": "uniform",
        "nstars": 3000,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 15.0,
        "age": 12.5,
        "z": 0.0002,
    }
    sample = injector.inject_single_stream(params, np.random.default_rng(3))
    distances = [ch["distance_modulus"] for ch in sample.metadata["channels"]]
    right = sample.label_stack[distances.index(15.0)].sum()
    wrong = sample.label_stack[distances.index(19.0)].sum()
    assert right >= 5
    assert wrong <= 0.05 * right

    # The same on the full sky, before any window is cut.
    full = injector.inject_stream_full_sky(params, np.random.default_rng(3))
    assert len(_label_pixels(full, 15.0)) >= 5
    assert len(_label_pixels(full, 19.0)) <= 0.05 * len(_label_pixels(full, 15.0))


class _TwoDistanceStream(StreamObsSource):
    """One stream, its phi1 < 0 half at `near` and its phi1 > 0 half at `far`."""

    def __init__(self, near, far):
        self.near, self.far = near, far

    def realize(self, params, rng):
        import pandas as pd

        near = super().realize({**params, "distance_modulus": self.near}, rng)
        far = super().realize({**params, "distance_modulus": self.far}, rng)
        return pd.concat(
            [near[near["phi1"] < 0], far[far["phi1"] > 0]], ignore_index=True
        )


def test_label_follows_the_half_of_the_stream_at_the_filters_distance(
    distance_background,
):
    """A stream whose first half is at distance modulus 16 and second half at
    18, seen through the matched filter at each of the two distances.

    The label is the stream stars the filter selects (user decision,
    2026-09-22), and the filter is not blind to other distances in the same
    way both ways:

    - the filter at 16 (the closer one) keeps the half at 16 and almost
      nothing of the half at 18 (measured 518 stars against 15);
    - the filter at 18 keeps the half at 18, but also sees the half at 16 at
      reduced strength (measured 146 against 69). A closer stream has about
      three times more stars bright enough to be detected, and roughly 10% of
      them fall inside the farther isochrone's polygon.

    So a model trained on this label learns "stars compatible with this
    distance": a stream shows up, weaker, at larger queried distances. What
    must always hold is that each filter's own half dominates.
    """
    import healpy as hp

    from streamgoggles.evaluation.footprint import stream_frame_coordinates

    injector = _distance_injector(distance_background, _TwoDistanceStream(16.0, 18.0))
    params = {
        "morphology": "uniform",
        "nstars": 8000,
        "width": 0.2,
        "length": 16.0,
        "distance_modulus": 17.0,
        "age": 12.5,
        "z": 0.0002,
    }
    full = injector.inject_stream_full_sky(params, np.random.default_rng(11))
    distances = [ch["distance_modulus"] for ch in full["channels"]]
    pixels = np.flatnonzero(np.sum(full["stream_raw_full"], axis=0) > 0)
    vectors = np.array(hp.pix2vec(injector.pix.nside, pixels)).T
    phi1, _ = stream_frame_coordinates(vectors, **full["placement"])
    # Pixels about 1 degree wide straddle phi1 = 0; leave that seam out.
    near_half, far_half = phi1 < -1.0, phi1 > 1.0

    def stars(distance, half):
        counts = full["stream_raw_full"][distances.index(distance)][pixels]
        return counts[half].sum()

    # Each filter's own half dominates.
    assert stars(16.0, near_half) > 2 * stars(16.0, far_half)
    assert stars(18.0, far_half) > stars(18.0, near_half)
    # The closer filter is nearly blind to the farther half ...
    assert stars(16.0, far_half) <= 0.1 * stars(16.0, near_half)
    # ... while the farther filter still sees the closer half.
    assert stars(18.0, near_half) > 0.1 * stars(18.0, far_half)


def test_injector_minimum_stream_length_reaches_the_window_sampler(
    real_background, stream_params, monkeypatch
):
    """Streams can be shorter than the default 5 degrees (Tucana III: 4.8), so
    the minimum length a window must hold is an injector setting."""
    import streamgoggles.injector as injector_module

    seen = []
    original = injector_module.sample_stream_window

    def spy(*args, **kwargs):
        seen.append(kwargs.get("min_stream_length_deg"))
        return original(*args, **kwargs)

    monkeypatch.setattr(injector_module, "sample_stream_window", spy)
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
        min_stream_length_deg=3.0,
    )
    injector.inject_single_stream(stream_params, np.random.default_rng(0))
    assert seen and all(value == 3.0 for value in seen)
