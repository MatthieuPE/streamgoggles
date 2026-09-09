"""Test pluggable background sources: StudyRegion, harmonize_columns, and the
three BackgroundSource implementations.

DataFileBackgroundSource is tested against a small synthetic DP2-schema
parquet, NOT the real data/background/dp2_star_gmax_27_skim.parquet — that
file is ~13GB and a plain full read OOM-kills a normal dev machine (confirmed
empirically). Running against the real file is a separate, manual, opt-in
concern (needs a bigger machine / a chunked-read strategy), not part of this
automated suite.

StreamObsLightBackgroundSource and StreamObsCatalogueBackgroundSource ARE
tested against the real streamobs machinery (Survey.load, Background) — no
mocking — since the LSST yr1 survey config and the light-generation CMD
resource files are genuinely present in this environment; no DP2 download or
GPU is needed for any of it.
"""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from streamgoggles.background_sources import (
    DataFileBackgroundSource,
    StreamObsCatalogueBackgroundSource,
    StreamObsLightBackgroundSource,
    StudyRegion,
    _study_region_to_phi_box,
    harmonize_columns,
)
from streamgoggles.data_preparation import deredden_dataframe

pytestmark = pytest.mark.background_sources


@pytest.fixture
def dp2_like_df():
    """A small synthetic DataFrame matching the real DP2 skim's schema
    (coord_ra, coord_dec, <band>_psfMag[Err], refExtendedness, ebv)."""
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame(
        {
            "coord_ra": rng.uniform(0, 360, n),
            "coord_dec": rng.uniform(-40, 40, n),
            "g_psfMag": rng.uniform(18, 26, n),
            "r_psfMag": rng.uniform(18, 26, n),
            "g_psfMagErr": rng.uniform(0.01, 0.1, n),
            "r_psfMagErr": rng.uniform(0.01, 0.1, n),
            "refExtendedness": rng.uniform(0, 1, n),
            "ebv": rng.uniform(0.01, 0.9, n),
        }
    )


# ---------------------------------------------------------------------------
# StudyRegion
# ---------------------------------------------------------------------------


def test_study_region_defaults_match_2026_09_08_decision():
    region = StudyRegion()
    assert region.center_ra == 0.0
    assert region.center_dec == -30.0
    assert region.width_deg == 70.0
    assert region.height_deg == 20.0


def test_study_region_is_inside_basic():
    region = StudyRegion(
        center_ra=100.0, center_dec=-10.0, width_deg=10.0, height_deg=6.0
    )
    ra = np.array([100.0, 108.0, 100.0])
    dec = np.array([-10.0, -10.0, -20.0])
    inside = region.is_inside(ra, dec)
    np.testing.assert_array_equal(inside, [True, False, False])


def test_study_region_is_inside_handles_ra_wraparound():
    region = StudyRegion(center_ra=0.0, center_dec=0.0, width_deg=10.0, height_deg=10.0)
    ra = np.array([358.0, 4.0, 180.0])
    dec = np.array([0.0, 0.0, 0.0])
    inside = region.is_inside(ra, dec)
    np.testing.assert_array_equal(inside, [True, True, False])


# ---------------------------------------------------------------------------
# _study_region_to_phi_box
# ---------------------------------------------------------------------------


def test_study_region_to_phi_box_center_maps_to_origin():
    region = StudyRegion(
        center_ra=123.0, center_dec=-40.0, width_deg=20.0, height_deg=8.0
    )
    gc_frame, phi1_limits, phi2_limits = _study_region_to_phi_box(region)

    center = SkyCoord(ra=region.center_ra * u.deg, dec=region.center_dec * u.deg)
    in_frame = center.transform_to(gc_frame)

    assert in_frame.phi1.deg == pytest.approx(0.0, abs=1e-8)
    assert in_frame.phi2.deg == pytest.approx(0.0, abs=1e-8)
    assert phi1_limits == (-10.0, 10.0)
    assert phi2_limits == (-4.0, 4.0)


def test_study_region_to_phi_box_offset_point_recovers_expected_dec():
    # A point directly "north" of center by phi2 degrees should land close to
    # center_dec + phi2 (small-angle regime).
    region = StudyRegion(center_ra=0.0, center_dec=-30.0)
    gc_frame, _, _ = _study_region_to_phi_box(region)
    point = SkyCoord(phi1=0 * u.deg, phi2=5 * u.deg, frame=gc_frame).transform_to(
        "icrs"
    )
    assert point.dec.deg == pytest.approx(-25.0, abs=0.1)


# ---------------------------------------------------------------------------
# harmonize_columns
# ---------------------------------------------------------------------------


def test_harmonize_columns_basic_renaming(dp2_like_df):
    out = harmonize_columns(
        dp2_like_df, namespace="lsst_dp2", bands=["g", "r"], dust_corrected=False
    )

    assert "ra" in out.columns and "dec" in out.columns
    np.testing.assert_array_equal(
        out["ra"].to_numpy(), dp2_like_df["coord_ra"].to_numpy()
    )
    assert "extendedness" in out.columns
    assert "lsst_dp2_g_obs" in out.columns
    assert "lsst_dp2_r_obs" in out.columns
    assert "lsst_dp2_g_err" in out.columns
    np.testing.assert_array_equal(
        out["lsst_dp2_g_obs"].to_numpy(), dp2_like_df["g_psfMag"].to_numpy()
    )


def test_harmonize_columns_uses_dereddened_magnitudes_when_dust_corrected(dp2_like_df):
    corrected = deredden_dataframe(dp2_like_df, bands=["g", "r"], ebv_col="ebv")
    out = harmonize_columns(
        corrected, namespace="lsst_dp2", bands=["g", "r"], dust_corrected=True
    )

    np.testing.assert_array_equal(
        out["lsst_dp2_g_obs"].to_numpy(), corrected["g_psfMag_dered"].to_numpy()
    )
    assert "A_g" in out.columns
    # Dereddened magnitude must actually differ from the raw one (real correction applied).
    assert not np.allclose(
        out["lsst_dp2_g_obs"].to_numpy(), dp2_like_df["g_psfMag"].to_numpy()
    )


def test_harmonize_columns_unsupported_source_raises(dp2_like_df):
    with pytest.raises(ValueError):
        harmonize_columns(dp2_like_df, namespace="lsst_dp2", source="not_dp2")


def test_harmonize_columns_infers_bands_when_not_given(dp2_like_df):
    out = harmonize_columns(dp2_like_df, namespace="lsst_dp2", dust_corrected=False)
    assert "lsst_dp2_g_obs" in out.columns
    assert "lsst_dp2_r_obs" in out.columns


# ---------------------------------------------------------------------------
# DataFileBackgroundSource (small synthetic parquet, never the real 13GB file)
# ---------------------------------------------------------------------------


def test_data_file_background_source_load(tmp_path, dp2_like_df):
    path = tmp_path / "tiny_dp2_skim.parquet"
    dp2_like_df.to_parquet(path)

    source = DataFileBackgroundSource()
    out = source.load(
        survey="lsst", release="dp2", region=None, cfg={"path": str(path)}
    )

    assert len(out) == len(dp2_like_df)
    assert "lsst_dp2_g_obs" in out.columns
    assert "lsst_dp2_g_err" in out.columns
    # Dust correction is mandatory (decision 25) -> must differ from raw magnitudes.
    assert not np.allclose(
        out["lsst_dp2_g_obs"].to_numpy(), dp2_like_df["g_psfMag"].to_numpy()
    )


def test_data_file_background_source_ignores_region(tmp_path, dp2_like_df):
    path = tmp_path / "tiny_dp2_skim.parquet"
    dp2_like_df.to_parquet(path)
    source = DataFileBackgroundSource()

    restrictive_region = StudyRegion(
        center_ra=0.0, center_dec=0.0, width_deg=1.0, height_deg=1.0
    )
    out_no_region = source.load(
        survey="lsst", release="dp2", region=None, cfg={"path": str(path)}
    )
    out_with_region = source.load(
        survey="lsst", release="dp2", region=restrictive_region, cfg={"path": str(path)}
    )
    assert len(out_no_region) == len(out_with_region) == len(dp2_like_df)


def test_data_file_background_source_missing_file_raises(tmp_path):
    source = DataFileBackgroundSource()
    with pytest.raises(FileNotFoundError):
        source.load(
            survey="lsst",
            release="dp2",
            region=None,
            cfg={"path": str(tmp_path / "does_not_exist.parquet")},
        )


def test_data_file_background_source_default_path_constant():
    assert (
        DataFileBackgroundSource._DEFAULT_PATH
        == "data/background/dp2_star_gmax_27_skim.parquet"
    )


# ---------------------------------------------------------------------------
# StreamObsLightBackgroundSource (real streamobs, no mocking)
# ---------------------------------------------------------------------------


def test_streamobs_light_source_load_real():
    source = StreamObsLightBackgroundSource()
    region = StudyRegion(center_ra=0.0, center_dec=-30.0, width_deg=6.0, height_deg=4.0)

    catalog = source.load(survey="lsst", release="yr1", region=region, cfg={})

    assert len(catalog) > 0
    assert "lsst_yr1_g_obs" in catalog.columns
    assert "lsst_yr1_r_obs" in catalog.columns
    assert "ra" in catalog.columns and "dec" in catalog.columns
    # dec should stay close to the region (phi2 in [-2, 2] around center_dec=-30).
    assert catalog["dec"].between(-34.0, -26.0).all()


def test_streamobs_light_source_custom_bands_respected():
    # bands is genuinely forwarded to Background/LightBackgroundGenerator: an
    # unbuilt band-pair CMD resource (only g,r is built in this environment)
    # must fail cleanly rather than silently falling back to g,r.
    source = StreamObsLightBackgroundSource()
    region = StudyRegion(center_ra=0.0, center_dec=-30.0, width_deg=6.0, height_deg=4.0)

    with pytest.raises(FileNotFoundError):
        source.load(
            survey="lsst", release="yr1", region=region, cfg={"bands": ["r", "i"]}
        )


def test_streamobs_light_source_unsupported_release_raises():
    source = StreamObsLightBackgroundSource()
    region = StudyRegion()
    with pytest.raises(FileNotFoundError):
        source.load(survey="lsst", release="not_a_real_release", region=region, cfg={})


def test_streamobs_light_source_defaults_region_when_none():
    # Must not raise when region=None -- falls back to StudyRegion() defaults.
    source = StreamObsLightBackgroundSource()
    catalog = source.load(survey="lsst", release="yr1", region=None, cfg={})
    assert len(catalog) > 0


# ---------------------------------------------------------------------------
# StreamObsCatalogueBackgroundSource (real streamobs injection, no mocking)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_catalog_stars():
    """A tiny true-star catalog for StreamInjector.inject(), all inside a
    (center_ra=0, center_dec=-30, width=10, height=6) StudyRegion."""
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame(
        {
            "ra": rng.uniform(-3.0, 3.0, n) % 360.0,
            "dec": rng.uniform(-32.0, -28.0, n),
            "lsst_g_true": rng.uniform(20.0, 25.0, n),
            "lsst_r_true": rng.uniform(20.0, 25.0, n),
        }
    )


def test_streamobs_catalogue_source_requires_catalog_stars():
    source = StreamObsCatalogueBackgroundSource()
    with pytest.raises(ValueError, match="catalog_stars"):
        source.load(survey="lsst", release="yr1", region=StudyRegion(), cfg={})


def test_streamobs_catalogue_source_load_real(synthetic_catalog_stars):
    source = StreamObsCatalogueBackgroundSource()
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=10.0, height_deg=6.0
    )

    out = source.load(
        survey="lsst",
        release="yr1",
        region=region,
        cfg={"catalog_stars": synthetic_catalog_stars, "source_type": "stars"},
    )

    assert len(out) > 0
    assert "lsst_yr1_g_obs" in out.columns
    assert "lsst_yr1_flag_observed" in out.columns
    assert (
        "lsst_g_true" in out.columns
    )  # true_col is namespaced by survey only, not release


def test_streamobs_catalogue_source_applies_region_post_filter(synthetic_catalog_stars):
    """streamobs's own injection ignores phi1/phi2/gc_frame entirely (verified
    against its source) -- region restriction must come from an explicit
    post-filter here, not from streamobs itself."""
    far_away = pd.DataFrame(
        {
            "ra": [170.0, 175.0],
            "dec": [60.0, 61.0],  # far outside the region
            "lsst_g_true": [22.0, 22.0],
            "lsst_r_true": [22.0, 22.0],
        }
    )
    catalog_stars = pd.concat([synthetic_catalog_stars, far_away], ignore_index=True)

    source = StreamObsCatalogueBackgroundSource()
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=10.0, height_deg=6.0
    )
    out = source.load(
        survey="lsst",
        release="yr1",
        region=region,
        cfg={"catalog_stars": catalog_stars, "source_type": "stars"},
    )

    assert out["dec"].max() < 50.0  # the two far-away stars were filtered out
