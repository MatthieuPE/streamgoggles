"""Test label rasterization: rasterize_binary, rasterize_density, the
rasterize() dispatcher, and rasterize_soft_distance's stub.

Pure HEALPix/gnomonic-projection math (matched_filter.py's own machinery,
reused here) -- no streamobs/survey loading needed, so these tests are fast
and fully self-contained.
"""

import numpy as np
import pytest

from streamgoggles.matched_filter import PixelizationSpec
from streamgoggles.rasterize import (
    LabelPolicy,
    rasterize,
    rasterize_binary,
    rasterize_density,
    rasterize_soft_distance,
)
from streamgoggles.windows import Window

pytestmark = pytest.mark.rasterize

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def window():
    return Window(center_ra=10.0, center_dec=-20.0, rotation_deg=0.0)


@pytest.fixture
def pix_odd():
    # Odd image size -> one pixel lands exactly on the window center
    # (row/col offset 0), giving an exact, deterministic alignment test.
    return PixelizationSpec(nside=128, pixel_scale_deg=0.5, image_size_pix=(11, 11))


@pytest.fixture
def pix_large():
    return PixelizationSpec(nside=128, pixel_scale_deg=0.5, image_size_pix=(41, 41))


# ---------------------------------------------------------------------------
# rasterize_density
# ---------------------------------------------------------------------------


def test_density_member_at_center_peaks_at_center_pixel(window, pix_odd):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    image = rasterize_density(ra, dec, window, pix_odd, normalization="none")

    assert image.shape == (11, 11)
    assert image.dtype == np.float32
    center = image[5, 5]
    assert center == pytest.approx(1.0)
    assert image.sum() == pytest.approx(1.0)  # the one member, nowhere else


def test_density_normalization_max(window, pix_odd):
    ra = np.full(5, window.center_ra)
    dec = np.full(5, window.center_dec)
    image = rasterize_density(ra, dec, window, pix_odd, normalization="max")
    assert image.max() == pytest.approx(1.0)


def test_density_normalization_sum(window, pix_odd):
    ra = np.full(5, window.center_ra)
    dec = np.full(5, window.center_dec)
    image = rasterize_density(ra, dec, window, pix_odd, normalization="sum")
    assert image.sum() == pytest.approx(1.0)


def test_density_normalization_none_conserves_total_count_at_exact_center(
    window, pix_odd
):
    # All members at the exact same sky position -> one HEALPix pixel, hit
    # exactly once by the grid point that sits exactly at the window center
    # (odd image size) -- an exact-conservation case. Off-center placement
    # cannot generally conserve totals through nearest-neighbor projection:
    # project() samples the HEALPix map at discrete grid points rather than
    # rebinning it, so some source pixels can be missed or (when
    # oversampling) sampled more than once.
    ra = np.full(37, window.center_ra)
    dec = np.full(37, window.center_dec)
    image = rasterize_density(ra, dec, window, pix_odd, normalization="none")
    assert image.sum() == pytest.approx(37.0)


def test_density_unknown_normalization_raises(window, pix_odd):
    with pytest.raises(ValueError, match="normalization"):
        rasterize_density(
            np.array([10.0]), np.array([-20.0]), window, pix_odd, normalization="bogus"
        )


def test_density_empty_members_is_all_zero(window, pix_odd):
    image = rasterize_density(
        np.array([]), np.array([]), window, pix_odd, normalization="max"
    )
    assert image.shape == (11, 11)
    np.testing.assert_array_equal(image, 0.0)


def test_density_smoothing_spreads_out_peak(window, pix_large):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    unsmoothed = rasterize_density(ra, dec, window, pix_large, normalization="none")
    smoothed = rasterize_density(
        ra, dec, window, pix_large, normalization="none", smooth_sigma_deg=1.0
    )
    assert smoothed.max() < unsmoothed.max()
    assert (smoothed > 0).sum() > (unsmoothed > 0).sum()  # spread to more pixels
    assert np.all(smoothed >= 0.0)  # no negative ringing leaked through


# ---------------------------------------------------------------------------
# rasterize_binary
# ---------------------------------------------------------------------------


def test_binary_member_at_center_marks_center_pixel(window, pix_odd):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    image = rasterize_binary(ra, dec, stream_width_deg=0.1, window=window, pix=pix_odd)

    assert image.shape == (11, 11)
    assert image.dtype == np.float32
    assert set(np.unique(image)) <= {0.0, 1.0}
    assert image[5, 5] == 1.0
    assert image.sum() == pytest.approx(1.0)


def test_binary_empty_members_is_all_zero(window, pix_odd):
    image = rasterize_binary(
        np.array([]), np.array([]), stream_width_deg=0.2, window=window, pix=pix_odd
    )
    np.testing.assert_array_equal(image, 0.0)


def test_binary_no_dilation_by_default(window, pix_large):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    image = rasterize_binary(
        ra,
        dec,
        stream_width_deg=2.0,
        window=window,
        pix=pix_large,
        dilate_to_width=False,
    )
    assert image.sum() == pytest.approx(1.0)


def test_binary_dilate_to_width_grows_mask(window, pix_large):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    small = rasterize_binary(
        ra,
        dec,
        stream_width_deg=0.5,
        window=window,
        pix=pix_large,
        dilate_to_width=True,
    )
    large = rasterize_binary(
        ra,
        dec,
        stream_width_deg=4.0,
        window=window,
        pix=pix_large,
        dilate_to_width=True,
    )
    assert small.sum() > 1.0  # some dilation happened
    assert large.sum() > small.sum()  # wider width -> more dilation


def test_binary_dilate_to_width_zero_is_noop(window, pix_large):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    image = rasterize_binary(
        ra,
        dec,
        stream_width_deg=0.0,
        window=window,
        pix=pix_large,
        dilate_to_width=True,
    )
    assert image.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rasterize() dispatcher
# ---------------------------------------------------------------------------


def test_rasterize_binary_policy_matches_direct_call(window, pix_odd):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    via_dispatch = rasterize(
        ra,
        dec,
        0.1,
        params={},
        policy="binary",
        window=window,
        pix=pix_odd,
        dilate_to_width=True,
    )
    direct = rasterize_binary(
        ra, dec, stream_width_deg=0.1, window=window, pix=pix_odd, dilate_to_width=True
    )
    np.testing.assert_array_equal(via_dispatch, direct)


def test_rasterize_density_policy_matches_direct_call(window, pix_odd):
    ra = np.array([window.center_ra])
    dec = np.array([window.center_dec])
    via_dispatch = rasterize(
        ra,
        dec,
        0.1,
        params={},
        policy="density",
        window=window,
        pix=pix_odd,
        normalization="sum",
    )
    direct = rasterize_density(ra, dec, window, pix_odd, normalization="sum")
    np.testing.assert_array_equal(via_dispatch, direct)


def test_rasterize_soft_distance_policy_raises_not_implemented(window, pix_odd):
    with pytest.raises(NotImplementedError):
        rasterize(
            np.array([10.0]),
            np.array([-20.0]),
            0.1,
            params={"distance_modulus": 16.8},
            policy="soft_distance",
            window=window,
            pix=pix_odd,
        )


def test_rasterize_unknown_policy_raises_value_error(window, pix_odd):
    with pytest.raises(ValueError, match="Unknown label policy"):
        rasterize(
            np.array([10.0]),
            np.array([-20.0]),
            0.1,
            params={},
            policy="sinusoid",
            window=window,
            pix=pix_odd,
        )


def test_rasterize_policy_matches_label_policy_enum_values():
    assert LabelPolicy.BINARY.value == "binary"
    assert LabelPolicy.DENSITY.value == "density"
    assert LabelPolicy.SOFT_DISTANCE.value == "soft_distance"


# ---------------------------------------------------------------------------
# rasterize_soft_distance
# ---------------------------------------------------------------------------


def test_rasterize_soft_distance_raises_not_implemented(window, pix_odd):
    with pytest.raises(NotImplementedError):
        rasterize_soft_distance(
            np.array([10.0]),
            np.array([-20.0]),
            params={"distance_modulus": 16.8},
            distance_moduli=[16.0, 16.8, 17.5],
            window=window,
            pix=pix_odd,
        )
