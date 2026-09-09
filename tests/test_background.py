"""Test Background.load_or_cache and build_raw_background_maps (real
streamobs, no mocking -- StreamObsLightBackgroundSource's precomputed CMD
resource for lsst/yr1 is genuinely present in this environment).

Cut/apply_cuts/apply_magnitude_clipping themselves are tested in
tests/test_data_preparation.py (they live in data_preparation.py, shared
with injector.py -- decision 13); this file only exercises Background's use
of them (test_load_or_cache_applies_cuts) alongside its own caching logic.
"""

import healpy as hp
import numpy as np
import pytest

from streamgoggles.background import Background, build_raw_background_maps
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.data_preparation import Cut
from streamgoggles.matched_filter import PixelizationSpec, StreamobsSplineFilter
from streamgoggles.storage import BackgroundMapStore

pytestmark = pytest.mark.background

# ---------------------------------------------------------------------------
# build_raw_background_maps / Background.load_or_cache
# (real streamobs: StreamObsLightBackgroundSource + StreamobsSplineFilter)
# ---------------------------------------------------------------------------


class _CountingFilter:
    """Wraps a MatchedFilter, counting select() calls -- used to prove
    Background.load_or_cache actually skips recomputation on a cache hit."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def select(self, catalog, bands, distance_modulus):
        self.calls += 1
        return self.inner.select(catalog, bands, distance_modulus)


@pytest.fixture
def small_region():
    return StudyRegion(center_ra=0.0, center_dec=-30.0, width_deg=6.0, height_deg=4.0)


@pytest.fixture
def small_pix():
    return PixelizationSpec(nside=64, pixel_scale_deg=1.0)


@pytest.fixture
def real_filter():
    return StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )


def test_build_raw_background_maps_shapes(small_region, small_pix, real_filter):
    catalog = StreamObsLightBackgroundSource().load(
        survey="lsst", release="yr1", region=small_region, cfg={}
    )
    result = build_raw_background_maps(
        catalog, real_filter, ["g", "r"], [16.0, 17.0], small_pix
    )

    assert set(result.keys()) == {16.0, 17.0}
    npix = hp.nside2npix(small_pix.nside)
    for raw_map, valid_mask in result.values():
        assert raw_map.shape == (npix,)
        assert valid_mask.shape == (npix,)
        assert valid_mask.dtype == bool


def test_load_or_cache_basic(tmp_path, small_region, small_pix, real_filter):
    store = BackgroundMapStore(tmp_path / "background_maps")
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={},
        study_region=small_region,
        cuts=[],
        clipping=None,
        matched_filter=real_filter,
        bands=["g", "r"],
        distance_moduli=[16.8],
        finalize_cfg=None,
        store=store,
        pix=small_pix,
        survey="lsst",
        release="yr1",
        filter_config={"age": 12.5, "z": 0.0002},
    )

    assert bg.catalog is not None and len(bg.catalog) > 0
    assert set(bg.raw_map_full_dict.keys()) == {16.8}
    assert bg.finalized_map_full_dict[16.8] is None  # finalize disabled
    np.testing.assert_array_equal(bg.footprint, bg.valid_mask_full)


def test_load_or_cache_reuses_cached_maps_without_recompute(
    tmp_path, small_region, small_pix, real_filter
):
    filt = _CountingFilter(real_filter)
    store = BackgroundMapStore(tmp_path / "background_maps")
    kwargs = {
        "source": StreamObsLightBackgroundSource(),
        "source_cfg": {},
        "study_region": small_region,
        "cuts": [],
        "clipping": None,
        "matched_filter": filt,
        "bands": ["g", "r"],
        "distance_moduli": [16.8],
        "finalize_cfg": None,
        "store": store,
        "pix": small_pix,
        "survey": "lsst",
        "release": "yr1",
        "filter_config": {"age": 12.5, "z": 0.0002},
    }

    bg1 = Background.load_or_cache(**kwargs)
    assert filt.calls == 1

    bg2 = Background.load_or_cache(**kwargs)
    assert filt.calls == 1  # cache hit: select() not called again
    assert bg2.catalog is None  # nothing needed loading either

    np.testing.assert_array_equal(
        bg1.raw_map_full_dict[16.8], bg2.raw_map_full_dict[16.8]
    )
    np.testing.assert_array_equal(bg1.valid_mask_full, bg2.valid_mask_full)


def test_load_or_cache_partial_cache_hit_only_computes_missing(
    tmp_path, small_region, small_pix, real_filter
):
    filt = _CountingFilter(real_filter)
    store = BackgroundMapStore(tmp_path / "background_maps")
    base_kwargs = {
        "source": StreamObsLightBackgroundSource(),
        "source_cfg": {},
        "study_region": small_region,
        "cuts": [],
        "clipping": None,
        "matched_filter": filt,
        "bands": ["g", "r"],
        "finalize_cfg": None,
        "store": store,
        "pix": small_pix,
        "survey": "lsst",
        "release": "yr1",
        "filter_config": {"age": 12.5, "z": 0.0002},
    }

    Background.load_or_cache(distance_moduli=[16.8], **base_kwargs)
    assert filt.calls == 1

    bg = Background.load_or_cache(distance_moduli=[16.8, 17.2], **base_kwargs)
    assert filt.calls == 2  # only the new distance modulus triggered select()
    assert set(bg.raw_map_full_dict.keys()) == {16.8, 17.2}


def test_load_or_cache_applies_cuts(tmp_path, small_region, small_pix, real_filter):
    store_no_cuts = BackgroundMapStore(tmp_path / "no_cuts")
    store_with_cuts = BackgroundMapStore(tmp_path / "with_cuts")

    common = {
        "source": StreamObsLightBackgroundSource(),
        "source_cfg": {},
        "study_region": small_region,
        "matched_filter": real_filter,
        "bands": ["g", "r"],
        "distance_moduli": [16.8],
        "finalize_cfg": None,
        "pix": small_pix,
        "survey": "lsst",
        "release": "yr1",
        "filter_config": {"age": 12.5, "z": 0.0002},
    }

    bg_no_cuts = Background.load_or_cache(
        cuts=[], clipping=None, store=store_no_cuts, **common
    )
    strict_cut = [Cut(quantity="mag", band="g", op="<", value=18.0)]
    bg_with_cuts = Background.load_or_cache(
        cuts=strict_cut, clipping=None, store=store_with_cuts, **common
    )

    assert len(bg_with_cuts.catalog) < len(bg_no_cuts.catalog)
