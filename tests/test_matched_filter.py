"""Test matched-filter selection, pixelization, and gnomonic projection."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import angular_separation

from streamgoggles.matched_filter import (
    PixelizationSpec,
    StreamobsSplineFilter,
    _tangent_plane_radec,
    combine_full_maps,
    crop_window,
    finalize_full,
    make_raw_map,
    native_pixel_scale_deg,
    project,
)

pytestmark = pytest.mark.matched_filter

# ---------------------------------------------------------------------------
# PixelizationSpec
# ---------------------------------------------------------------------------


def test_pixelization_spec_defaults():
    pix = PixelizationSpec()
    assert pix.nside == 128
    assert pix.image_size_pix == (256, 256)


def test_pixelization_spec_pixel_scale_auto_derived_from_nside():
    pix = PixelizationSpec(nside=256, pixel_scale_deg=None)
    assert pix.pixel_scale_deg == pytest.approx(native_pixel_scale_deg(256))


def test_pixelization_spec_raises_on_oversampling():
    with pytest.raises(ValueError, match="native HEALPix resolution"):
        PixelizationSpec(nside=128, pixel_scale_deg=0.01)


def test_pixelization_spec_accepts_coarser_than_native():
    pix = PixelizationSpec(
        nside=128, pixel_scale_deg=0.5
    )  # coarser than native -> fine
    assert pix.pixel_scale_deg == 0.5


def test_pixelization_spec_invalid_nside_raises():
    # Power-of-2 is only required for NESTED ordering; RING (the default,
    # nest=False) accepts any positive integer nside.
    with pytest.raises(ValueError):
        PixelizationSpec(nside=100, nest=True)


def test_pixelization_spec_non_power_of_two_nside_ok_for_ring():
    pix = PixelizationSpec(nside=100, nest=False, pixel_scale_deg=1.0)
    assert pix.nside == 100


# ---------------------------------------------------------------------------
# StreamobsSplineFilter (real streamobs — no data download needed, only the
# bundled isochrone tables via ugali)
# ---------------------------------------------------------------------------


def test_streamobs_filter_requires_age_and_z():
    with pytest.raises(KeyError):
        StreamobsSplineFilter(iso_config={"age": 12.0})


def test_streamobs_filter_select_returns_bool_array_of_right_length():
    filt = StreamobsSplineFilter(iso_config={"age": 12.5, "z": 0.0002})
    catalog = pd.DataFrame(
        {
            "g_obs": np.array([20.0, 22.0, 24.0]),
            "r_obs": np.array([19.5, 21.5, 23.5]),
        }
    )
    selected = filt.select(catalog, ["g", "r"], distance_modulus=16.8)
    assert selected.dtype == bool
    assert len(selected) == 3


def test_streamobs_filter_select_with_namespace_columns():
    filt = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    catalog = pd.DataFrame(
        {
            "lsst_yr1_g_obs": np.array([20.0, 22.0, 24.0]),
            "lsst_yr1_r_obs": np.array([19.5, 21.5, 23.5]),
        }
    )
    selected = filt.select(catalog, ["g", "r"], distance_modulus=16.8)
    assert len(selected) == 3


def test_streamobs_filter_polygon_cache_reused():
    filt = StreamobsSplineFilter(iso_config={"age": 12.5, "z": 0.0002})
    poly_a = filt._polygon(["g", "r"], 16.8)
    poly_b = filt._polygon(["g", "r"], 16.8)
    poly_c = filt._polygon(["g", "r"], 18.0)
    assert poly_a is poly_b  # same (bands, distance_modulus) -> cached
    assert poly_a is not poly_c


def test_streamobs_filter_rejects_stars_far_from_isochrone():
    """Stars shifted well blueward of the isochrone locus should be rejected —
    mirrors streamobs's own test_match_filter_rejects_offset_stars."""
    filt = StreamobsSplineFilter(iso_config={"age": 12.5, "z": 0.0002})
    dm = 16.8
    rng = np.random.default_rng(0)
    n = 200
    mag_r = rng.uniform(dm + 1.0, dm + 9.0, n)
    mag_g = mag_r - 2.0  # 2 mag blueward of any realistic MS/RGB color
    catalog = pd.DataFrame({"g_obs": mag_g, "r_obs": mag_r})
    selected = filt.select(catalog, ["g", "r"], distance_modulus=dm)
    assert selected.sum() / n < 0.05


# ---------------------------------------------------------------------------
# make_raw_map
# ---------------------------------------------------------------------------


def test_make_raw_map_basic_counts():
    pix = PixelizationSpec(nside=4, pixel_scale_deg=10.0)
    # Two stars at the exact same (ra, dec) -> same HEALPix pixel.
    catalog = pd.DataFrame({"ra": [10.0, 10.0, 200.0], "dec": [5.0, 5.0, -30.0]})
    selected = np.array([True, True, False])
    raw_map, valid_mask = make_raw_map(catalog, selected, pix)

    npix = hp.nside2npix(4)
    assert raw_map.shape == (npix,)
    assert valid_mask.shape == (npix,)
    assert raw_map.sum() == 2  # only the two selected stars are counted
    # Both touched pixels are valid, regardless of selection.
    touched = hp.ang2pix(4, [10.0, 200.0], [5.0, -30.0], lonlat=True)
    assert valid_mask[touched].all()


def test_make_raw_map_empty_catalog_returns_zeros():
    pix = PixelizationSpec(nside=4, pixel_scale_deg=10.0)
    catalog = pd.DataFrame({"ra": [], "dec": []})
    raw_map, valid_mask = make_raw_map(catalog, np.array([], dtype=bool), pix)
    npix = hp.nside2npix(4)
    assert raw_map.shape == (npix,)
    assert not valid_mask.any()
    assert (raw_map == 0).all()


def test_make_raw_map_handles_invalid_pixels():
    """Dedicated empty/invalid-pixel test (decision 22): a catalog covering
    only half the sky must leave the other half's pixels explicitly invalid,
    not silently zero-and-indistinguishable."""
    pix = PixelizationSpec(nside=8, pixel_scale_deg=5.0)
    rng = np.random.default_rng(1)
    n = 2000
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(0, 90, n)  # northern hemisphere only
    catalog = pd.DataFrame({"ra": ra, "dec": dec})
    selected = np.ones(n, dtype=bool)

    raw_map, valid_mask = make_raw_map(catalog, selected, pix)

    npix = hp.nside2npix(8)
    pix_dec = 90.0 - np.degrees(hp.pix2ang(8, np.arange(npix))[0])
    southern = pix_dec < -10.0  # well clear of any northern-hemisphere pixel
    assert not valid_mask[southern].any()
    assert (raw_map[southern] == 0).all()
    assert valid_mask[~southern].any()  # some northern pixels are valid


def test_make_raw_map_additivity():
    """raw(background) + raw(stream) == raw(background ∪ stream) (decision 13)."""
    pix = PixelizationSpec(nside=8, pixel_scale_deg=5.0)
    rng = np.random.default_rng(2)
    background = pd.DataFrame(
        {
            "ra": rng.uniform(0, 360, 500),
            "dec": rng.uniform(-40, 40, 500),
        }
    )
    stream = pd.DataFrame(
        {
            "ra": rng.uniform(100, 110, 50),
            "dec": rng.uniform(-5, 5, 50),
        }
    )
    combined = pd.concat([background, stream], ignore_index=True)

    bg_selected = np.ones(len(background), dtype=bool)
    stream_selected = np.ones(len(stream), dtype=bool)
    combined_selected = np.ones(len(combined), dtype=bool)

    raw_bg, _ = make_raw_map(background, bg_selected, pix)
    raw_stream, _ = make_raw_map(stream, stream_selected, pix)
    raw_combined, _ = make_raw_map(combined, combined_selected, pix)

    np.testing.assert_array_equal(raw_bg + raw_stream, raw_combined)


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


def test_project_uniform_map_is_flat():
    pix = PixelizationSpec(
        nside=64,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(16, 16),
        pixel_scale_deg=0.5,
    )
    npix = hp.nside2npix(64)
    raw_map = np.ones(npix, dtype=float)
    valid_mask = np.ones(npix, dtype=bool)

    image, image_valid_mask = project(raw_map, valid_mask, pix)

    assert image_valid_mask.all()
    np.testing.assert_allclose(image, 1.0, atol=1e-6)


def test_project_handles_invalid_pixels():
    """Dedicated empty/invalid-pixel test: a window straddling a footprint
    edge must not crash, and pixels touched by any invalid HEALPix neighbor
    must be marked invalid rather than silently interpolated."""
    nside = 64
    pix = PixelizationSpec(
        nside=nside,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(20, 20),
        pixel_scale_deg=0.5,
        interpolate=True,
    )
    npix = hp.nside2npix(nside)
    raw_map = np.ones(npix, dtype=float)
    _ra_pix, dec_pix = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    valid_mask = dec_pix > 0.0  # the window (centered at dec=0) straddles this edge

    image, image_valid_mask = project(raw_map, valid_mask, pix)  # must not raise

    assert image.shape == (20, 20)
    assert image_valid_mask.any()
    assert not image_valid_mask.all()  # part of the window is invalid
    assert (image[~image_valid_mask] == 0.0).all()


def test_project_no_ra_stretching_at_high_dec():
    """A stream/window near the pole must not show artificial RA-stretching:
    equal pixel offsets must correspond to equal angular separations on sky,
    regardless of declination (decision 15)."""
    pix_low = PixelizationSpec(
        nside=128,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(5, 5),
        pixel_scale_deg=1.0,
    )
    pix_high = PixelizationSpec(
        nside=128,
        center_ra=180.0,
        center_dec=80.0,
        image_size_pix=(5, 5),
        pixel_scale_deg=1.0,
    )

    ra_low, dec_low = _tangent_plane_radec(pix_low)
    ra_high, dec_high = _tangent_plane_radec(pix_high)

    def sep_deg(ra, dec, i0, j0, i1, j1):
        return np.degrees(
            angular_separation(
                np.radians(ra[i0, j0]),
                np.radians(dec[i0, j0]),
                np.radians(ra[i1, j1]),
                np.radians(dec[i1, j1]),
            )
        )

    sep_low = sep_deg(ra_low, dec_low, 2, 2, 2, 3)
    sep_high = sep_deg(ra_high, dec_high, 2, 2, 2, 3)

    assert sep_low == pytest.approx(1.0, rel=1e-3)
    assert sep_high == pytest.approx(1.0, rel=1e-3)


def test_project_nearest_neighbor_matches_source_pixel():
    nside = 32
    pix = PixelizationSpec(
        nside=nside,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(4, 4),
        pixel_scale_deg=1.0,
        interpolate=False,
    )
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(3)
    raw_map = rng.uniform(size=npix)
    valid_mask = np.ones(npix, dtype=bool)

    image, image_valid_mask = project(raw_map, valid_mask, pix)

    ra_grid, dec_grid = _tangent_plane_radec(pix)
    expected_pix = hp.ang2pix(nside, ra_grid.ravel(), dec_grid.ravel(), lonlat=True)
    np.testing.assert_allclose(image.ravel(), raw_map[expected_pix])
    assert image_valid_mask.all()


def test_project_rotation_swaps_axes_near_90_degrees():
    """A 90-degree window rotation should swap the roles of the row/column
    tangent-plane axes (sanity check that rotation is a real tangent-plane
    rotation, not a no-op)."""
    pix_0 = PixelizationSpec(
        nside=128,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(5, 5),
        pixel_scale_deg=1.0,
        rotation_deg=0.0,
    )
    pix_90 = PixelizationSpec(
        nside=128,
        center_ra=180.0,
        center_dec=0.0,
        image_size_pix=(5, 5),
        pixel_scale_deg=1.0,
        rotation_deg=90.0,
    )

    ra_0, dec_0 = _tangent_plane_radec(pix_0)
    ra_90, dec_90 = _tangent_plane_radec(pix_90)

    # Offset one column to the right at rotation=0 ~= offset one row up at rotation=90.
    assert ra_0[2, 3] == pytest.approx(ra_90[1, 2], abs=1e-6)
    assert dec_0[2, 3] == pytest.approx(dec_90[1, 2], abs=1e-6)


# ---------------------------------------------------------------------------
# combine_full_maps
# ---------------------------------------------------------------------------


def test_combine_full_maps_additivity():
    valid_mask = np.array([True, True, True, False])
    bg = np.array([1.0, 2.0, 3.0, 0.0])
    stream = np.array([0.0, 1.0, 0.0, 0.0])
    combined = combine_full_maps(bg, stream, valid_mask)
    np.testing.assert_array_equal(combined, [1.0, 3.0, 3.0, 0.0])


def test_combine_full_maps_forces_invalid_to_zero():
    valid_mask = np.array([True, False])
    bg = np.array([1.0, 5.0])
    stream = np.array([1.0, 5.0])
    combined = combine_full_maps(bg, stream, valid_mask)
    assert combined[1] == 0.0


# ---------------------------------------------------------------------------
# finalize_full
# ---------------------------------------------------------------------------


def test_finalize_full_none_cfg_is_identity():
    combined = np.array([1.0, 2.0, 3.0])
    valid_mask = np.ones(3, dtype=bool)
    result = finalize_full(combined, valid_mask, cfg=None)
    np.testing.assert_array_equal(result, combined)


def test_finalize_full_disabled_is_identity():
    combined = np.array([1.0, 2.0, 3.0])
    valid_mask = np.ones(3, dtype=bool)
    result = finalize_full(combined, valid_mask, cfg={"enabled": False})
    np.testing.assert_array_equal(result, combined)


def test_finalize_full_enabled_raises_not_implemented():
    combined = np.array([1.0, 2.0, 3.0])
    valid_mask = np.ones(3, dtype=bool)
    with pytest.raises(NotImplementedError):
        finalize_full(combined, valid_mask, cfg={"enabled": True})


# ---------------------------------------------------------------------------
# crop_window
# ---------------------------------------------------------------------------


class _FakeWindow:
    def __init__(self, center_ra, center_dec, rotation_deg):
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.rotation_deg = rotation_deg


def test_crop_window_matches_manual_project_with_overrides():
    nside = 32
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(4)
    finalized = rng.uniform(size=npix)
    valid_mask = np.ones(npix, dtype=bool)

    base_pix = PixelizationSpec(
        nside=nside,
        center_ra=0.0,
        center_dec=0.0,
        rotation_deg=0.0,
        image_size_pix=(6, 6),
        pixel_scale_deg=1.0,
    )
    window = _FakeWindow(center_ra=45.0, center_dec=-10.0, rotation_deg=30.0)

    image, image_valid_mask = crop_window(finalized, valid_mask, window, base_pix)

    expected_pix = PixelizationSpec(
        nside=nside,
        center_ra=45.0,
        center_dec=-10.0,
        rotation_deg=30.0,
        image_size_pix=(6, 6),
        pixel_scale_deg=1.0,
    )
    expected_image, expected_valid_mask = project(finalized, valid_mask, expected_pix)

    np.testing.assert_array_equal(image, expected_image)
    np.testing.assert_array_equal(image_valid_mask, expected_valid_mask)
