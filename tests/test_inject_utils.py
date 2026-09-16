"""Test inject_utils.py's richness conversions: mass <-> N <-> surface
brightness. Real ugali isochrone sampling throughout -- no mocking.

Focus is on the monotonic relationships these conversions must respect
(more mass -> more stars -> brighter -> smaller surface brightness number)
and round-trips between each pair, since a broken monotonicity or a broken
inverse would silently corrupt config.py's richness_kind resolution
(injector.py's resolve_richness_to_nstars).
"""

import pytest
import ugali.isochrone

from streamgoggles import inject_utils

pytestmark = pytest.mark.inject_utils


@pytest.fixture
def isochrone_params():
    return {
        "isochrone": {
            "name": "Marigo2017",
            "survey": "lsst",
            "age": 12.0,
            "z": 0.0004,
            "band_1": "g",
            "band_2": "r",
        },
        "distance_modulus": {"center": {"value": 16.8}},
    }


@pytest.fixture
def built_isochrone():
    iso = ugali.isochrone.factory(
        name="Marigo2017", survey="lsst", age=12.0, z=0.0004, band_1="g", band_2="r"
    )
    iso.distance_modulus = 16.8
    return iso


# ---------------------------------------------------------------------------
# N <-> mass
# ---------------------------------------------------------------------------


def test_convert_N_to_Mass_positive(isochrone_params):
    mass = inject_utils.convert_N_to_Mass(1000.0, isochrone_params=isochrone_params)
    assert mass > 0


def test_convert_N_to_Mass_is_linear_in_N(isochrone_params):
    # mass = N * mean_stellar_mass_per_star -- exactly linear, not just monotonic.
    m1 = inject_utils.convert_N_to_Mass(500.0, isochrone_params=isochrone_params)
    m2 = inject_utils.convert_N_to_Mass(5000.0, isochrone_params=isochrone_params)
    assert m2 > m1
    assert m2 / m1 == pytest.approx(10.0)


def test_convert_Mass_to_N_monotonic_increasing(isochrone_params):
    n1 = inject_utils.convert_Mass_to_N(100.0, isochrone_params=isochrone_params)
    n2 = inject_utils.convert_Mass_to_N(1000.0, isochrone_params=isochrone_params)
    assert n2 > n1


def test_convert_N_Mass_round_trip(isochrone_params):
    N = 3000.0
    mass = inject_utils.convert_N_to_Mass(N, isochrone_params=isochrone_params)
    n_back = inject_utils.convert_Mass_to_N(mass, isochrone_params=isochrone_params)
    assert n_back == pytest.approx(N)


def test_convert_N_to_Mass_requires_isochrone_source():
    with pytest.raises(ValueError, match="isochrone_config_path or isochrone_params"):
        inject_utils.convert_N_to_Mass(1000.0)


# ---------------------------------------------------------------------------
# N -> surface brightness
# ---------------------------------------------------------------------------


def test_more_stars_gives_smaller_brighter_surface_brightness(isochrone_params):
    # More stars -> more total flux -> brighter -> SMALLER mag/arcsec^2.
    sb_few = inject_utils.convert_N_SurfaceBrightness(
        500, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    sb_many = inject_utils.convert_N_SurfaceBrightness(
        5000, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    assert sb_many < sb_few


def test_larger_area_gives_fainter_surface_brightness(isochrone_params):
    # Same N spread over a larger area -> fainter -> LARGER mag/arcsec^2.
    sb_small_area = inject_utils.convert_N_SurfaceBrightness(
        2000, isochrone_params=isochrone_params, stream_length=4.0, stream_width=0.1
    )
    sb_large_area = inject_utils.convert_N_SurfaceBrightness(
        2000, isochrone_params=isochrone_params, stream_length=16.0, stream_width=0.4
    )
    assert sb_large_area > sb_small_area


# ---------------------------------------------------------------------------
# surface brightness -> N (numeric inverse)
# ---------------------------------------------------------------------------


def test_convert_SurfaceBrightness_to_N_round_trip(isochrone_params):
    N = 2000.0
    sb = inject_utils.convert_N_SurfaceBrightness(
        N, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    n_back = inject_utils.convert_SurfaceBrightness_to_N(
        sb, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    assert n_back == pytest.approx(N, rel=1e-3)


def test_brighter_surface_brightness_gives_more_stars(isochrone_params):
    # Surface brightness in mag/arcsec^2: SMALLER value = brighter.
    n_faint = inject_utils.convert_SurfaceBrightness_to_N(
        32.0, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    n_bright = inject_utils.convert_SurfaceBrightness_to_N(
        26.0, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )
    assert n_bright > n_faint


# ---------------------------------------------------------------------------
# N -> luminosity
# ---------------------------------------------------------------------------


def test_more_stars_gives_brighter_smaller_total_magnitude(built_isochrone):
    m_few = inject_utils.convert_N_to_luminosity(
        500, isochrone=built_isochrone, band="r"
    )
    m_many = inject_utils.convert_N_to_luminosity(
        5000, isochrone=built_isochrone, band="r"
    )
    assert m_many < m_few


# ---------------------------------------------------------------------------
# Composed chains (mass <-> N <-> surface brightness) -- the actual
# end-to-end relationship injector.py's richness_kind resolution relies on.
# ---------------------------------------------------------------------------


def _surface_brightness_for_mass(mass, isochrone_params):
    n = inject_utils.convert_Mass_to_N(mass, isochrone_params=isochrone_params)
    return inject_utils.convert_N_SurfaceBrightness(
        n, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
    )


def test_more_massive_stream_has_smaller_surface_brightness(isochrone_params):
    """The more massive the stream, the smaller (brighter) its surface
    brightness -- the specific relationship injector.py's
    resolve_richness_to_nstars depends on being consistent across units."""
    sb_light = _surface_brightness_for_mass(500.0, isochrone_params)
    sb_heavy = _surface_brightness_for_mass(5000.0, isochrone_params)
    assert sb_heavy < sb_light


def test_brighter_surface_brightness_implies_more_mass(isochrone_params):
    def _mass_for_surface_brightness(sb):
        n = inject_utils.convert_SurfaceBrightness_to_N(
            sb, isochrone_params=isochrone_params, stream_length=8.0, stream_width=0.2
        )
        return inject_utils.convert_N_to_Mass(n, isochrone_params=isochrone_params)

    mass_bright = _mass_for_surface_brightness(26.0)
    mass_faint = _mass_for_surface_brightness(32.0)
    assert mass_bright > mass_faint


# ---------------------------------------------------------------------------
# Isochrone construction cache (PLAN.md section 6.16)
# ---------------------------------------------------------------------------


def test_build_isochrone_is_cached(isochrone_params):
    """Constructing a ugali isochrone globs and parses its whole data
    directory (~11,500 files), and convert_SurfaceBrightness_to_N's brentq
    inversion asks for the same one ~31 times per stream. Caching it is the
    difference between ~1.2s and ~0.07s per training sample."""
    inject_utils._isochrone_factory_cached.cache_clear()
    cfg = isochrone_params["isochrone"]

    inject_utils._build_isochrone(cfg)
    inject_utils._build_isochrone(cfg)
    inject_utils._build_isochrone(
        dict(reversed(list(cfg.items())))
    )  # key order must not matter

    info = inject_utils._isochrone_factory_cached.cache_info()
    assert info.misses == 1, "the isochrone should be constructed exactly once"
    assert info.hits == 2


def test_build_isochrone_returns_isolated_copies(isochrone_params):
    """Every caller sets `distance_modulus` on the isochrone before using
    it, so a shared cached object would let one caller's distance modulus
    silently become the next one's. ugali keeps that value in mutable
    Parameter objects held in dicts on the instance, so a plain shallow
    copy does NOT isolate it -- this test is what caught that."""
    inject_utils._isochrone_factory_cached.cache_clear()
    cfg = isochrone_params["isochrone"]

    first = inject_utils._build_isochrone(cfg)
    baseline = first.distance_modulus
    first.distance_modulus = 25.0

    second = inject_utils._build_isochrone(cfg)
    assert second.distance_modulus == baseline
    assert first.distance_modulus == 25.0


def test_build_isochrone_falls_back_when_params_unhashable(monkeypatch):
    """Unhashable parameters (a list from someone's YAML, say) must still
    reach the factory uncached, so adding the cache can never restrict what
    callers are allowed to pass.

    This is the one test here that stubs ugali rather than calling it for
    real: the point is to exercise *this module's* hashability branch, and
    every parameter the real factory accepts happens to be a scalar, so
    there is no unhashable input it would also accept."""
    seen = {}

    def fake_factory(**kwargs):
        seen.update(kwargs)
        return "built-uncached"

    monkeypatch.setattr(ugali.isochrone, "factory", fake_factory)
    cfg = {"name": "Marigo2017", "nodes": [1, 2, 3]}

    assert inject_utils._build_isochrone(cfg) == "built-uncached"
    assert seen == cfg


def test_cached_isochrone_gives_identical_conversions(isochrone_params):
    """The cache must be a pure speedup: identical numbers, to the bit."""
    inject_utils._isochrone_factory_cached.cache_clear()
    kwargs = dict(
        stream_length=8.0, stream_width=0.2, isochrone_params=isochrone_params, band="r"
    )

    original = inject_utils._build_isochrone
    inject_utils._build_isochrone = lambda p: ugali.isochrone.factory(**p)
    try:
        uncached = inject_utils.convert_SurfaceBrightness_to_N(32.0, **kwargs)
    finally:
        inject_utils._build_isochrone = original

    cached = inject_utils.convert_SurfaceBrightness_to_N(32.0, **kwargs)
    assert cached == uncached
