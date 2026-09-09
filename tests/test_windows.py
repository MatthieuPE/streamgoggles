"""Test window sampling and placement: Window, random/stream windows, tiling."""

import healpy as hp
import numpy as np
import pytest
from scipy import stats

from streamgoggles.windows import (
    Window,
    sample_random_window,
    sample_stream_window,
    tile_footprint,
)

pytestmark = pytest.mark.windows


@pytest.fixture
def footprint():
    """A simple HEALPix footprint: |dec| < 20 deg, nside=32."""
    nside = 32
    npix = hp.nside2npix(nside)
    _, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    return {"nside": nside, "mask": np.abs(dec) < 20.0}


@pytest.fixture
def straight_stream():
    """A straight, 8-deg-long stream centered at (ra=180, dec=0)."""
    t = np.linspace(0.0, 1.0, 300)
    return {
        "ra": 176.0 + 8.0 * t,
        "dec": np.zeros_like(t),
        "width_deg": 0.2,
        "length_deg": 8.0,
    }


def _window_overlaps_footprint(window, mask, nside):
    """True if any valid footprint pixel falls inside `window` -- a stronger
    check than "the center pixel is valid": it confirms the window as a
    whole region actually intersects the footprint, not just its center."""
    valid_pixels = np.flatnonzero(mask)
    ra, dec = hp.pix2ang(nside, valid_pixels, lonlat=True)
    return window.contains(ra, dec).any()


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def test_window_defaults():
    w = Window(center_ra=10.0, center_dec=-5.0)
    assert w.width_deg == 12.8
    assert w.height_deg == 12.8
    assert w.rotation_deg == 0.0


def test_window_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        Window(center_ra=0.0, center_dec=0.0, width_deg=0.0)
    with pytest.raises(ValueError):
        Window(center_ra=0.0, center_dec=0.0, height_deg=-1.0)


def test_window_rotation_normalized_into_0_360():
    assert Window(
        center_ra=0.0, center_dec=0.0, rotation_deg=400.0
    ).rotation_deg == pytest.approx(40.0)
    assert Window(
        center_ra=0.0, center_dec=0.0, rotation_deg=-30.0
    ).rotation_deg == pytest.approx(330.0)


def test_window_contains_center_point():
    w = Window(center_ra=180.0, center_dec=0.0, width_deg=10.0, height_deg=10.0)
    assert w.contains(np.array([180.0]), np.array([0.0]))[0]


def test_window_contains_excludes_far_point():
    w = Window(center_ra=180.0, center_dec=0.0, width_deg=10.0, height_deg=10.0)
    assert not w.contains(np.array([250.0]), np.array([40.0]))[0]


def test_window_contains_edge_boundary_near_equator():
    # Gnomonic projection is nonlinear (xi ~ tan(delta_ra), not delta_ra itself),
    # so use a comfortable margin either side of the +-5deg boundary rather than
    # assuming a flat-sky approximation holds right at the edge.
    w = Window(
        center_ra=180.0,
        center_dec=0.0,
        rotation_deg=0.0,
        width_deg=10.0,
        height_deg=10.0,
    )
    inside = w.contains(np.array([184.5, 185.5]), np.array([0.0, 0.0]))
    assert inside[0]
    assert not inside[1]


def test_window_contains_respects_rotation():
    # A point just east of center is inside at rotation=0, but a 90deg rotation
    # swaps the roles of the width/height axes -- a point offset far in dec
    # (outside the un-rotated height) can become "inside" after rotating.
    w0 = Window(
        center_ra=180.0,
        center_dec=0.0,
        rotation_deg=0.0,
        width_deg=10.0,
        height_deg=2.0,
    )
    w90 = Window(
        center_ra=180.0,
        center_dec=0.0,
        rotation_deg=90.0,
        width_deg=10.0,
        height_deg=2.0,
    )
    point_ra, point_dec = np.array([180.0]), np.array([4.0])
    assert not w0.contains(point_ra, point_dec)[0]
    assert w90.contains(point_ra, point_dec)[0]


# ---------------------------------------------------------------------------
# sample_random_window
# ---------------------------------------------------------------------------


def test_sample_random_window_center_in_footprint(footprint):
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = sample_random_window(
            footprint["mask"], footprint["nside"], size_deg=10.0, rng=rng
        )
        pixel = hp.ang2pix(footprint["nside"], w.center_ra, w.center_dec, lonlat=True)
        assert footprint["mask"][pixel]


def test_sample_random_window_tilt_uniform_ks_test(footprint):
    rng = np.random.default_rng(1)
    tilts = [
        sample_random_window(
            footprint["mask"], footprint["nside"], rng=rng
        ).rotation_deg
        for _ in range(500)
    ]
    _ks_stat, p_value = stats.kstest(tilts, stats.uniform(loc=0, scale=360).cdf)
    assert p_value > 0.01, (
        f"tilt distribution not consistent with uniform (p={p_value})"
    )


def test_sample_random_window_empty_footprint_raises():
    mask = np.zeros(hp.nside2npix(32), dtype=bool)
    with pytest.raises(ValueError):
        sample_random_window(mask, nside=32)


def test_sample_random_window_reproducible_with_seed(footprint):
    w1 = sample_random_window(
        footprint["mask"], footprint["nside"], rng=np.random.default_rng(42)
    )
    w2 = sample_random_window(
        footprint["mask"], footprint["nside"], rng=np.random.default_rng(42)
    )
    assert w1 == w2


def test_sample_random_window_overlaps_footprint(footprint):
    rng = np.random.default_rng(9)
    for _ in range(20):
        w = sample_random_window(
            footprint["mask"], footprint["nside"], size_deg=10.0, rng=rng
        )
        assert _window_overlaps_footprint(w, footprint["mask"], footprint["nside"])


def test_sample_random_window_no_acceptance_condition_needed():
    """Decision 21: for pure-background negatives, the window is drawn with
    NO acceptance condition beyond "center in footprint". A single-valid-pixel
    footprint would make almost any additional condition (e.g. "surrounding
    area also covered") fail/retry -- this must still succeed immediately."""
    nside = 16
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[1000] = True
    rng = np.random.default_rng(0)

    w = sample_random_window(mask, nside, size_deg=5.0, rng=rng)

    pixel = hp.ang2pix(nside, w.center_ra, w.center_dec, lonlat=True)
    assert pixel == 1000


# ---------------------------------------------------------------------------
# sample_stream_window
# ---------------------------------------------------------------------------


def test_sample_stream_window_overlap_constraint_holds_across_many_samples(
    footprint, straight_stream
):
    """Dedicated windowing test (§4.4b): the >=5deg (or full-length) overlap
    floor must hold for every sampled stream window, across many draws."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        w = sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            footprint["mask"],
            footprint["nside"],
            size_deg=10.0,
            min_stream_length_deg=5.0,
            rng=rng,
        )
        inside = w.contains(straight_stream["ra"], straight_stream["dec"])
        assert inside.any()
        in_ra = straight_stream["ra"][inside]
        # Straight stream along constant dec -> covered length ~= ra range (deg) here.
        covered_length = float(in_ra.max() - in_ra.min()) if inside.sum() > 1 else 0.0
        assert covered_length >= 5.0 - 1e-6 or covered_length == pytest.approx(
            straight_stream["length_deg"], abs=0.5
        )


def test_sample_stream_window_tilt_uniform_ks_test(footprint, straight_stream):
    rng = np.random.default_rng(3)
    tilts = [
        sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            footprint["mask"],
            footprint["nside"],
            size_deg=10.0,
            min_stream_length_deg=5.0,
            rng=rng,
        ).rotation_deg
        for _ in range(300)
    ]
    _ks_stat, p_value = stats.kstest(tilts, stats.uniform(loc=0, scale=360).cdf)
    assert p_value > 0.01, (
        f"tilt distribution not consistent with uniform (p={p_value})"
    )


def test_sample_stream_window_sometimes_cuts_off_stream(footprint, straight_stream):
    """The overlap constraint is a FLOOR, not a guarantee of full inclusion --
    a nontrivial fraction of windows should still cut the stream off."""
    rng = np.random.default_rng(4)
    fully_inside_count = 0
    n_trials = 100
    for _ in range(n_trials):
        w = sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            footprint["mask"],
            footprint["nside"],
            size_deg=10.0,
            min_stream_length_deg=5.0,
            rng=rng,
        )
        inside = w.contains(straight_stream["ra"], straight_stream["dec"])
        if inside.all():
            fully_inside_count += 1

    assert 0 < fully_inside_count < n_trials


def test_sample_stream_window_short_stream_requires_full_inclusion(footprint):
    """A stream shorter than min_stream_length_deg must be entirely inside
    the sampled window (the floor becomes 'the whole stream fits')."""
    t = np.linspace(0.0, 1.0, 50)
    short_ra = 179.0 + 2.0 * t  # 2 deg long, well under the 5 deg floor
    short_dec = np.zeros_like(t)

    rng = np.random.default_rng(5)
    for _ in range(20):
        w = sample_stream_window(
            short_ra,
            short_dec,
            stream_width_deg=0.2,
            background_footprint=footprint["mask"],
            nside=footprint["nside"],
            size_deg=10.0,
            min_stream_length_deg=5.0,
            rng=rng,
        )
        assert w.contains(short_ra, short_dec).all()


def test_sample_stream_window_empty_stream_raises(footprint):
    with pytest.raises(ValueError):
        sample_stream_window(
            np.array([]),
            np.array([]),
            stream_width_deg=0.2,
            background_footprint=footprint["mask"],
            nside=footprint["nside"],
        )


def test_sample_stream_window_footprint_constraint_respected(straight_stream):
    # Footprint excludes the northern half of the stream's neighborhood --
    # only southern-leaning windows should ever be returned.
    nside = 32
    npix = hp.nside2npix(nside)
    _, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    restrictive_footprint = dec < 0.5

    rng = np.random.default_rng(6)
    for _ in range(20):
        w = sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            restrictive_footprint,
            nside,
            size_deg=10.0,
            min_stream_length_deg=5.0,
            max_attempts=500,
            rng=rng,
        )
        pixel = hp.ang2pix(nside, w.center_ra, w.center_dec, lonlat=True)
        assert restrictive_footprint[pixel]


def test_sample_stream_window_overlaps_footprint(footprint, straight_stream):
    rng = np.random.default_rng(10)
    for _ in range(20):
        w = sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            footprint["mask"],
            footprint["nside"],
            size_deg=10.0,
            min_stream_length_deg=5.0,
            rng=rng,
        )
        assert _window_overlaps_footprint(w, footprint["mask"], footprint["nside"])


def test_sample_stream_window_succeeds_when_stream_longer_than_window(footprint):
    """If the stream is longer than the window itself, full inclusion is
    geometrically impossible -- sample_stream_window must still succeed by
    satisfying just the >=5deg floor, never requiring the whole stream."""
    t = np.linspace(0.0, 1.0, 500)
    long_ra = 170.0 + 20.0 * t  # 20 deg long
    long_dec = np.zeros_like(t)

    rng = np.random.default_rng(11)
    for _ in range(20):
        w = sample_stream_window(
            long_ra,
            long_dec,
            stream_width_deg=0.2,
            background_footprint=footprint["mask"],
            nside=footprint["nside"],
            size_deg=8.0,  # smaller than the 20 deg stream
            min_stream_length_deg=5.0,
            max_attempts=500,
            rng=rng,
        )
        inside = w.contains(long_ra, long_dec)
        assert inside.any()
        assert (
            not inside.all()
        )  # the full 20 deg stream can never fit in an 8 deg window

        in_ra = long_ra[inside]
        covered_length = float(in_ra.max() - in_ra.min()) if inside.sum() > 1 else 0.0
        assert covered_length >= 5.0 - 1e-6


def test_sample_stream_window_raises_after_max_attempts(footprint, straight_stream):
    # An impossibly large min_stream_length_deg (exceeding the stream's own
    # total length) can never be satisfied.
    rng = np.random.default_rng(7)
    with pytest.raises(RuntimeError):
        sample_stream_window(
            straight_stream["ra"],
            straight_stream["dec"],
            straight_stream["width_deg"],
            footprint["mask"],
            footprint["nside"],
            size_deg=1.0,
            min_stream_length_deg=1000.0,
            max_attempts=10,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# tile_footprint
# ---------------------------------------------------------------------------


def test_tile_footprint_empty_footprint_returns_empty_list():
    mask = np.zeros(hp.nside2npix(32), dtype=bool)
    assert tile_footprint(mask, nside=32) == []


def test_tile_footprint_produces_a_reasonable_number_of_tiles(footprint):
    tiles = tile_footprint(footprint["mask"], footprint["nside"], tile_size_deg=15.0)
    # Footprint spans |dec|<20 (40 deg tall) x full 360 deg in ra -> a rough
    # order-of-magnitude check, not an exact tiling-geometry assertion.
    assert 10 < len(tiles) < 300


def test_tile_footprint_uses_fixed_tilt(footprint):
    tiles = tile_footprint(
        footprint["mask"], footprint["nside"], tile_size_deg=15.0, tilt_deg=27.0
    )
    assert all(t.rotation_deg == pytest.approx(27.0) for t in tiles)


def test_tile_footprint_centers_are_valid_footprint_pixels(footprint):
    tiles = tile_footprint(footprint["mask"], footprint["nside"], tile_size_deg=15.0)
    for w in tiles:
        pixel = hp.ang2pix(footprint["nside"], w.center_ra, w.center_dec, lonlat=True)
        assert footprint["mask"][pixel]


def test_tile_footprint_windows_overlap_footprint(footprint):
    tiles = tile_footprint(footprint["mask"], footprint["nside"], tile_size_deg=15.0)
    for w in tiles:
        assert _window_overlaps_footprint(w, footprint["mask"], footprint["nside"])
