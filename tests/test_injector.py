"""Test stream injection: resolve_richness_to_nstars, place_stream_in_footprint,
StreamInjector.inject_single_stream, and inject_background_only.

Everything runs against real streamobs (survey loading, StreamInjector,
StreamModel via stream_sources.py) -- no mocking, except that
inject_single_stream's end-to-end tests monkeypatch rasterize.rasterize()
since rasterize.py isn't implemented yet (verified separately: the real,
unpatched call raises NotImplementedError, and every step before it already
runs for real).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import streamgoggles.injector as injector_module
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
from streamgoggles.matched_filter import PixelizationSpec, StreamobsSplineFilter
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
# (real streamobs, real background, real cuts -- rasterize monkeypatched
# where noted)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_background(tmp_path_factory):
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(20, 20))
    store = BackgroundMapStore(tmp_path_factory.mktemp("bgmaps"))
    mf = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filter=mf,
        bands=["g", "r"],
        distance_moduli=[16.8, 17.5],
        finalize_cfg=None,
        store=store,
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_config={"age": 12.5, "z": 0.0002},
    )
    return bg, mf, pix


@pytest.fixture
def real_injector(real_background):
    bg, mf, pix = real_background
    return StreamInjector(
        background=bg,
        matched_filter=mf,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        bands=("g", "r"),
        richness_kind="nstars",
        label_policy="density",
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


@pytest.fixture
def stub_rasterize(monkeypatch):
    """Replace rasterize.rasterize with a zero map of the right shape, so
    inject_single_stream's non-rasterize steps (all real) can be tested
    end-to-end despite rasterize.py's real implementation not existing yet.
    """

    def _fake(**kwargs):
        ny, nx = kwargs["pix"].image_size_pix
        return np.zeros((ny, nx), dtype=float)

    monkeypatch.setattr(injector_module.rasterize, "rasterize", _fake)


def test_inject_single_stream_builds_valid_sample(
    real_injector, stream_params, stub_rasterize
):
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))

    assert sample.map_stack.shape == (2, 20, 20)
    assert sample.map_stack.dtype == np.float32
    assert sample.label_stack.shape == (2, 20, 20)
    assert sample.valid_mask.shape == (20, 20)
    assert sample.valid_mask.dtype == bool
    assert sample.params["nstars"] == 3000
    assert "window" in sample.metadata
    assert sample.metadata["distance_moduli"] == [16.8, 17.5]


def test_inject_single_stream_map_has_signal(
    real_injector, stream_params, stub_rasterize
):
    sample = real_injector.inject_single_stream(stream_params, np.random.default_rng(3))
    assert sample.map_stack.sum() > 0


def test_inject_single_stream_richness_resolved_before_realize(
    real_injector, stream_params, stub_rasterize
):
    params = dict(stream_params)
    del params["nstars"]
    params["richness"] = 3000
    real_injector.richness_kind = "nstars"
    sample = real_injector.inject_single_stream(params, np.random.default_rng(3))
    assert sample.params["nstars"] == 3000
    assert "richness" not in sample.params


def test_inject_single_stream_strict_cuts_reduce_signal(
    real_background, stream_params, stub_rasterize
):
    bg, mf, pix = real_background
    lenient = StreamInjector(
        background=bg,
        matched_filter=mf,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    strict = StreamInjector(
        background=bg,
        matched_filter=mf,
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


def test_inject_single_stream_rasterize_not_implemented_is_the_only_blocker(
    real_injector, stream_params
):
    """Documents the current boundary: everything up through crop_window runs
    for real; only the final rasterize step is unimplemented."""
    with pytest.raises(NotImplementedError):
        real_injector.inject_single_stream(stream_params, np.random.default_rng(3))


def test_inject_background_only_builds_valid_sample(real_background):
    bg, _mf, pix = real_background
    window = sample_random_window(
        bg.footprint, pix.nside, size_deg=20.0, rng=np.random.default_rng(4)
    )
    sample = inject_background_only(bg, window, pix)

    assert sample.map_stack.shape == (2, 20, 20)
    assert sample.map_stack.dtype == np.float32
    assert np.all(sample.label_stack == 0.0)
    assert sample.params == {}
    assert sample.metadata["distance_moduli"] == [16.8, 17.5]


def test_inject_background_only_matches_manual_crop(real_background):
    from streamgoggles.matched_filter import crop_window

    bg, _mf, pix = real_background
    window = sample_random_window(
        bg.footprint, pix.nside, size_deg=20.0, rng=np.random.default_rng(4)
    )
    sample = inject_background_only(bg, window, pix)

    dm = min(bg.raw_map_full_dict)
    expected_map, expected_valid = crop_window(
        bg.raw_map_full_dict[dm], bg.valid_mask_full, window, pix
    )
    np.testing.assert_allclose(sample.map_stack[0], expected_map, atol=1e-4)
    np.testing.assert_array_equal(sample.valid_mask, expected_valid)


# ---------------------------------------------------------------------------
# StreamInjector.__init__
# ---------------------------------------------------------------------------


def test_injector_namespace_derivation(real_background):
    bg, mf, pix = real_background
    inj = StreamInjector(
        background=bg,
        matched_filter=mf,
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
    bg, mf, pix = real_background
    inj = StreamInjector(
        background=bg,
        matched_filter=mf,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )
    assert inj.bands == ("g", "r")
