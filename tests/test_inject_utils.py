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
