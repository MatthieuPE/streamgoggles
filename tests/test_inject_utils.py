"""Test inject_utils.py's richness conversions: mass <-> N <-> surface
brightness. Real ugali isochrone sampling throughout -- no mocking.

Focus is on the monotonic relationships these conversions must respect
(more mass -> more stars -> brighter -> smaller surface brightness number)
and round-trips between each pair, since a broken monotonicity or a broken
inverse would silently corrupt config.py's richness_kind resolution
(injector.py's resolve_richness_to_nstars).
"""

import numpy as np
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
    kwargs = {
        "stream_length": 8.0,
        "stream_width": 0.2,
        "isochrone_params": isochrone_params,
        "band": "r",
    }

    original = inject_utils._build_isochrone
    inject_utils._build_isochrone = lambda p: ugali.isochrone.factory(**p)
    try:
        uncached = inject_utils.convert_SurfaceBrightness_to_N(32.0, **kwargs)
    finally:
        inject_utils._build_isochrone = original

    cached = inject_utils.convert_SurfaceBrightness_to_N(32.0, **kwargs)
    assert cached == uncached


def test_build_isochrone_distinguishes_different_parameters(isochrone_params):
    """The failure mode that would matter most once a full (age, z, ...)
    grid is being scanned: a cache keyed too coarsely would hand back one
    combination's isochrone for another, silently, with no error anywhere.
    Different parameters must give genuinely different isochrones."""
    inject_utils._isochrone_factory_cached.cache_clear()
    base = dict(isochrone_params["isochrone"])

    young = inject_utils._build_isochrone({**base, "age": 8.0})
    old = inject_utils._build_isochrone({**base, "age": 13.5})
    metal_poor = inject_utils._build_isochrone({**base, "z": 0.0001})

    assert young.age != old.age
    assert metal_poor.z != young.z
    # Three distinct parameter sets -> three distinct constructions.
    assert inject_utils._isochrone_factory_cached.cache_info().misses == 3


def test_cached_conversions_differ_across_age_and_metallicity(isochrone_params):
    """End-to-end version of the same guard: varying age or z must change
    the resolved star count, and re-asking must stay stable (no drift from
    one combination's cached state leaking into another's)."""
    inject_utils._isochrone_factory_cached.cache_clear()

    def n_for(age, z):
        params = {
            "isochrone": {**isochrone_params["isochrone"], "age": age, "z": z},
            "distance_modulus": {"center": {"value": 16.8}},
        }
        return inject_utils.convert_SurfaceBrightness_to_N(
            32.0, stream_length=8.0, stream_width=0.2, isochrone_params=params, band="r"
        )

    a = n_for(10.0, 0.0004)
    b = n_for(13.0, 0.0004)
    c = n_for(10.0, 0.0016)

    assert len({a, b, c}) == 3, "different (age, z) must give different N"
    # Interleaving must not contaminate: re-asking gives the same answers.
    assert n_for(10.0, 0.0004) == a
    assert n_for(13.0, 0.0004) == b


# ---------------------------------------------------------------------------
# isochrone.sample() cache (PLAN.md section 6.18)
# ---------------------------------------------------------------------------


def test_sample_isochrone_is_cached_within_and_across_conversions(isochrone_params):
    """brentq calls sample() 44x per conversion with an isochrone that never
    changes during the inversion, at ~1.3ms each. Caching it pays off even
    when (age, z) never repeat -- the case that matters for a grid scan."""
    inject_utils._isochrone_factory_cached.cache_clear()
    inject_utils._isochrone_sample_cached.cache_clear()

    inject_utils.convert_SurfaceBrightness_to_N(
        32.0,
        stream_length=8.0,
        stream_width=0.2,
        isochrone_params=isochrone_params,
        band="r",
    )

    info = inject_utils._isochrone_sample_cached.cache_info()
    assert info.misses == 1, "sample() should be computed once per isochrone"
    assert info.hits > 10, "the rest of the inversion should hit the cache"


def test_sample_isochrone_cache_is_independent_of_distance_modulus(isochrone_params):
    """The subtle half of the key design: ugali's sample() returns absolute
    magnitudes and the distance modulus is added afterwards, so dm must NOT
    be part of the sample() cache key -- while it must still change the
    conversion result. Both halves are asserted here, because getting either
    backwards is silent."""
    cfg = isochrone_params["isochrone"]
    isochrone = inject_utils._build_isochrone(cfg)

    isochrone.distance_modulus = 16.0
    near = inject_utils.sample_isochrone(isochrone, 10000)
    isochrone.distance_modulus = 19.0
    far = inject_utils.sample_isochrone(isochrone, 10000)
    for a, b in zip(near, far):
        np.testing.assert_array_equal(a, b)

    def n_at(dm):
        params = {"isochrone": cfg, "distance_modulus": {"center": {"value": dm}}}
        return inject_utils.convert_SurfaceBrightness_to_N(
            32.0,
            stream_length=8.0,
            stream_width=0.2,
            isochrone_params=params,
            band="r",
        )

    assert n_at(16.0) != n_at(17.0), "distance must still change the answer"


def test_cached_sample_arrays_are_read_only(isochrone_params):
    """The arrays are shared between callers, so an in-place write would
    silently corrupt every later conversion. Fail loudly instead."""
    isochrone = inject_utils._build_isochrone(isochrone_params["isochrone"])
    arrays = inject_utils.sample_isochrone(isochrone, 10000)
    with pytest.raises(ValueError):
        arrays[0][0] = 1.0


def test_conversion_still_responds_to_every_parameter(isochrone_params):
    """End-to-end guard over all the axes a full grid scan will vary: each
    must change the resolved star count, with both caches active."""
    cfg = isochrone_params["isochrone"]

    def n_for(age=12.5, z=0.0002, dm=16.0, width=0.2, length=8.0):
        params = {
            "isochrone": {**cfg, "age": age, "z": z},
            "distance_modulus": {"center": {"value": dm}},
        }
        return inject_utils.convert_SurfaceBrightness_to_N(
            32.0,
            stream_length=length,
            stream_width=width,
            isochrone_params=params,
            band="r",
        )

    base = n_for()
    assert n_for(dm=17.0) != base
    assert n_for(age=10.0) != base
    assert n_for(z=0.0008) != base
    assert n_for(width=0.4) != base
    assert n_for(length=12.0) != base
    assert n_for() == base, "re-asking must be stable"


def test_sample_cache_differs_across_isochrones(isochrone_params):
    """The sample() cache is keyed on the isochrone's identity, so a wrong
    key would serve one population's magnitudes for another -- silently, and
    only visibly as subtly wrong star counts."""
    inject_utils._isochrone_sample_cached.cache_clear()
    base = isochrone_params["isochrone"]

    young = inject_utils.sample_isochrone(
        inject_utils._build_isochrone({**base, "age": 8.0}), 10000
    )
    old = inject_utils.sample_isochrone(
        inject_utils._build_isochrone({**base, "age": 13.5}), 10000
    )
    metal_rich = inject_utils.sample_isochrone(
        inject_utils._build_isochrone({**base, "z": 0.002}), 10000
    )

    assert not np.array_equal(young, old)
    assert not np.array_equal(young, metal_rich)


def test_caches_stay_correct_under_eviction(isochrone_params):
    """'Works for different values' has to keep holding once there are more
    distinct parameter sets than the caches can hold. Cycle through more
    than maxsize, forcing eviction, and every combination must still return
    exactly what it returned before anything was evicted."""
    inject_utils._isochrone_factory_cached.cache_clear()
    inject_utils._isochrone_sample_cached.cache_clear()
    maxsize = inject_utils._isochrone_factory_cached.cache_info().maxsize
    base = isochrone_params["isochrone"]

    # Vary metallicity in small steps: enough distinct cache KEYS to force
    # eviction, while every value stays inside ugali's accepted range.
    metallicities = [0.0002 + 1e-6 * i for i in range(maxsize + 4)]
    first_pass = {
        z: inject_utils.sample_isochrone(
            inject_utils._build_isochrone({**base, "z": z}), 10000
        ).copy()
        for z in metallicities
    }
    assert inject_utils._isochrone_factory_cached.cache_info().currsize <= maxsize, (
        "the cache must stay bounded rather than growing without limit"
    )

    # The earliest entries have certainly been evicted by now; recomputing
    # them must reproduce the original values exactly.
    probes = (
        metallicities[0],
        metallicities[1],
        metallicities[len(metallicities) // 2],
        metallicities[-1],
    )
    for z in probes:
        again = inject_utils.sample_isochrone(
            inject_utils._build_isochrone({**base, "z": z}), 10000
        )
        np.testing.assert_array_equal(again, first_pass[z])


def test_conversions_stable_when_parameter_sets_are_interleaved(isochrone_params):
    """Round-robin across several parameter sets, the way a grid scan will:
    each must keep returning its own answer rather than drifting toward
    whichever was asked for most recently."""
    base = isochrone_params["isochrone"]
    combos = [(10.0, 0.0002, 16.0), (12.5, 0.0008, 17.0), (13.5, 0.0002, 15.5)]

    def n_for(age, z, dm):
        params = {
            "isochrone": {**base, "age": age, "z": z},
            "distance_modulus": {"center": {"value": dm}},
        }
        return inject_utils.convert_SurfaceBrightness_to_N(
            32.0,
            stream_length=8.0,
            stream_width=0.2,
            isochrone_params=params,
            band="r",
        )

    expected = {combo: n_for(*combo) for combo in combos}
    assert len(set(expected.values())) == len(combos), "combos must be distinguishable"

    for _ in range(3):
        for combo in combos:
            assert n_for(*combo) == expected[combo]
