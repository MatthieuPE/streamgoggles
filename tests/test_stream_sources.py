"""Test stream generation: StreamObsSource (real streamobs, no mocking) and the
ExternalSimSource stub.
"""

import numpy as np
import pandas as pd
import pytest

from streamgoggles.stream_sources import (
    ExternalSimSource,
    StreamObsSource,
)

pytestmark = pytest.mark.stream_sources


@pytest.fixture
def uniform_params():
    return {
        "morphology": "uniform",
        "nstars": 3000,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 17.5,
        "age": 12.0,
        "z": 0.0004,
    }


@pytest.fixture
def spline_control_points():
    return [
        {"phi1": -6.0, "phi2": 0.0},
        {"phi1": -3.0, "phi2": 0.5},
        {"phi1": 0.0, "phi2": 0.8},
        {"phi1": 3.0, "phi2": 0.3},
        {"phi1": 6.0, "phi2": -0.2},
    ]


@pytest.fixture
def spline_params(spline_control_points):
    return {
        "morphology": "spline",
        "nstars": 3000,
        "control_points": spline_control_points,
        "width": 0.2,
        "distance_modulus": 17.5,
        "age": 12.0,
        "z": 0.0004,
    }


# ---------------------------------------------------------------------------
# StreamObsSource — general
# ---------------------------------------------------------------------------


def test_realize_unknown_morphology_raises(uniform_params):
    params = dict(
        uniform_params, morphology="sinusoid"
    )  # explicitly excluded, decision 5
    with pytest.raises(ValueError, match="morphology"):
        StreamObsSource().realize(params, np.random.default_rng(0))


def test_realize_uniform_returns_expected_columns_and_count(uniform_params):
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(0))

    assert (
        len(df) == uniform_params["nstars"]
    )  # true count, before any selection (decision 16)
    assert df["is_stream"].all()


def test_realize_output_contains_only_minimal_columns(uniform_params):
    """The output must contain exactly phi1/phi2/dist/is_stream + the two
    true-magnitude columns -- no mu1/mu2/rv (always NaN, no velocity model)
    or the internal-only initial `mass` column, which streamobs's raw
    `StreamModel.sample()` output otherwise carries along unused."""
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(0))
    assert set(df.columns) == {
        "phi1",
        "phi2",
        "dist",
        "is_stream",
        "lsst_g_true",
        "lsst_r_true",
    }


def test_realize_distance_modulus_recovered_exactly(uniform_params):
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(0))
    np.testing.assert_array_equal(
        df["dist"].to_numpy(), uniform_params["distance_modulus"]
    )


def test_realize_reproducible_with_same_seed(uniform_params):
    df1 = StreamObsSource().realize(uniform_params, np.random.default_rng(123))
    df2 = StreamObsSource().realize(uniform_params, np.random.default_rng(123))
    pd.testing.assert_frame_equal(df1, df2)


def test_realize_population_z_changes_true_magnitudes(uniform_params):
    low_z = dict(uniform_params, z=0.0001)
    high_z = dict(uniform_params, z=0.01)
    df_low = StreamObsSource().realize(low_z, np.random.default_rng(7))
    df_high = StreamObsSource().realize(high_z, np.random.default_rng(7))
    assert not np.allclose(
        df_low["lsst_g_true"].to_numpy(), df_high["lsst_g_true"].to_numpy()
    )


def test_realize_population_age_changes_true_magnitudes(uniform_params):
    young = dict(uniform_params, age=8.0)
    old = dict(uniform_params, age=13.5)
    df_young = StreamObsSource().realize(young, np.random.default_rng(7))
    df_old = StreamObsSource().realize(old, np.random.default_rng(7))
    assert not np.allclose(
        df_young["lsst_g_true"].to_numpy(), df_old["lsst_g_true"].to_numpy()
    )


def test_realize_custom_survey_and_bands(uniform_params):
    params = dict(uniform_params, survey="lsst", release="yr1", band_1="r", band_2="i")
    df = StreamObsSource().realize(params, np.random.default_rng(0))
    # true_col is namespaced by survey name only (release-independent, streamobs convention).
    assert "lsst_r_true" in df.columns
    assert "lsst_i_true" in df.columns


# ---------------------------------------------------------------------------
# StreamObsSource — uniform morphology geometry
# ---------------------------------------------------------------------------


def test_realize_uniform_track_is_straight(uniform_params):
    """The track center is a straight line at phi2=0 -- residual scatter
    should be centered at 0, not systematically offset or curved."""
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(1))
    assert df["phi2"].mean() == pytest.approx(0.0, abs=0.02)


def test_realize_uniform_width_matches_requested(uniform_params):
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(1))
    assert df["phi2"].std() == pytest.approx(uniform_params["width"], rel=0.1)


def test_realize_uniform_length_matches_requested(uniform_params):
    df = StreamObsSource().realize(uniform_params, np.random.default_rng(1))
    length = uniform_params["length"]
    # Uniform sampler is bounded exactly within [-length/2, length/2] by construction.
    assert df["phi1"].min() >= -length / 2.0 - 1e-9
    assert df["phi1"].max() <= length / 2.0 + 1e-9
    # With 3000 stars the observed span should closely approach the full length.
    observed_span = df["phi1"].max() - df["phi1"].min()
    assert observed_span == pytest.approx(length, rel=0.05)


# ---------------------------------------------------------------------------
# StreamObsSource — spline morphology geometry
# ---------------------------------------------------------------------------


def test_realize_spline_track_follows_control_points(
    spline_params, spline_control_points
):
    df = StreamObsSource().realize(spline_params, np.random.default_rng(2))

    for point in spline_control_points:
        near = df[np.abs(df["phi1"] - point["phi1"]) < 0.5]
        assert len(near) > 10, f"too few stars near control point {point}"
        assert near["phi2"].mean() == pytest.approx(point["phi2"], abs=0.15)


def test_realize_spline_requires_at_least_two_control_points(spline_params):
    params = dict(spline_params, control_points=[{"phi1": 0.0, "phi2": 0.0}])
    with pytest.raises(ValueError, match="control_points"):
        StreamObsSource().realize(params, np.random.default_rng(0))


def test_realize_spline_missing_width_without_fallback_raises(spline_control_points):
    params = {
        "morphology": "spline",
        "nstars": 10,
        "control_points": spline_control_points,  # no "width" per point
        # no top-level "width" fallback either
        "distance_modulus": 17.5,
        "age": 12.0,
        "z": 0.0004,
    }
    with pytest.raises(ValueError, match="width"):
        StreamObsSource().realize(params, np.random.default_rng(0))


def test_realize_spline_per_point_width_overrides_fallback():
    control_points = [
        {"phi1": -5.0, "phi2": 0.0, "width": 0.5},
        {"phi1": 0.0, "phi2": 0.0},  # falls back to params["width"]
        {"phi1": 5.0, "phi2": 0.0, "width": 0.1},
    ]
    _density_cfg, track_cfg = StreamObsSource._spline_density_track_config(
        {"control_points": control_points, "width": 0.2}
    )
    spread_values = track_cfg["spread"]["node_values"]
    np.testing.assert_array_equal(spread_values, [0.5, 0.2, 0.1])


# ---------------------------------------------------------------------------
# ExternalSimSource
# ---------------------------------------------------------------------------


def test_external_sim_source_realize_not_implemented():
    with pytest.raises(NotImplementedError):
        ExternalSimSource().realize(
            {"stream_id": "stream_001"}, np.random.default_rng(0)
        )
