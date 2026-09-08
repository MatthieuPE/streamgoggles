"""Test configuration loading and parameter spec management."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from streamgoggles.config import (
    BackgroundConfig,
    DistributionType,
    EvalGrid,
    MatchedFilterConfig,
    ParameterSpec,
    StreamConfig,
    build_eval_grid,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


def test_parameter_spec_is_fixed():
    fixed = ParameterSpec(name="width", dist_type=DistributionType.FIXED, value=0.2)
    free = ParameterSpec(
        name="width", dist_type=DistributionType.UNIFORM, min_val=0.1, max_val=0.5
    )
    assert fixed.is_fixed()
    assert not free.is_fixed()


def test_parameter_spec_sample_fixed(rng):
    spec = ParameterSpec(name="age", dist_type=DistributionType.FIXED, value=12.0)
    assert spec.sample(rng) == 12.0


def test_parameter_spec_sample_uniform_within_bounds(rng):
    spec = ParameterSpec(
        name="width", dist_type=DistributionType.UNIFORM, min_val=0.05, max_val=0.5
    )
    samples = [spec.sample(rng) for _ in range(200)]
    assert all(0.05 <= s <= 0.5 for s in samples)
    assert len(set(samples)) > 1  # actually varies


def test_parameter_spec_sample_log_uniform_within_bounds(rng):
    spec = ParameterSpec(
        name="mass",
        dist_type=DistributionType.LOG_UNIFORM,
        min_val=1.0e3,
        max_val=1.0e5,
    )
    samples = [spec.sample(rng) for _ in range(200)]
    assert all(1.0e3 <= s <= 1.0e5 for s in samples)


def test_parameter_spec_sample_discrete_returns_member(rng):
    spec = ParameterSpec(
        name="age", dist_type=DistributionType.DISCRETE, values=[10.0, 12.0, 13.5]
    )
    samples = [spec.sample(rng) for _ in range(50)]
    assert all(s in (10.0, 12.0, 13.5) for s in samples)
    assert len(set(samples)) > 1


def test_parameter_spec_sample_discrete_empty_raises(rng):
    spec = ParameterSpec(name="age", dist_type=DistributionType.DISCRETE, values=[])
    with pytest.raises(ValueError):
        spec.sample(rng)


# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------


def test_stream_config_load_injection_grid():
    config = StreamConfig.load(CONFIG_DIR / "streams" / "injection_grid.yaml")

    assert config.params["morphology"].is_fixed()
    assert config.params["morphology"].value == "uniform"
    assert config.params["position"].value == "uniform_in_footprint"

    assert config.params["orientation"].dist_type == DistributionType.UNIFORM
    assert config.params["orientation"].min_val == 0.0
    assert config.params["orientation"].max_val == 180.0

    assert config.richness_kind == "surface_brightness"
    assert config.params["richness"].is_fixed()
    assert config.params["richness"].value == 30.0

    assert config.background_fraction == 0.0
    assert config.label_policy == "density"
    assert config.persist is False


def test_stream_config_load_eval_grid_richness_is_discrete():
    config = StreamConfig.load(CONFIG_DIR / "streams" / "eval_grid.yaml")
    assert config.richness_kind == "surface_brightness"
    assert config.params["richness"].dist_type == DistributionType.DISCRETE
    assert config.params["richness"].values == [26.0, 27.2, 28.4, 29.6, 30.8, 32.0]
    assert config.persist is True


def test_stream_config_free_parameters(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "morphology": "uniform",
                "width": 0.2,
                "orientation": {"min": 0.0, "max": 180.0},
                "age": {"values": [10.0, 12.0]},
            }
        )
    )
    config = StreamConfig.load(path)
    assert set(config.free_parameters()) == {"orientation", "age"}


def test_stream_config_fix_is_immutable(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"width": {"min": 0.05, "max": 0.5}}))
    original = StreamConfig.load(path)

    fixed = original.fix("width", 0.3)

    # Original untouched
    assert not original.params["width"].is_fixed()
    # New config has the fix applied
    assert fixed.params["width"].is_fixed()
    assert fixed.params["width"].value == 0.3
    assert fixed is not original


def test_stream_config_fix_unknown_parameter_raises(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"width": 0.2}))
    config = StreamConfig.load(path)
    with pytest.raises(KeyError):
        config.fix("does_not_exist", 1.0)


def test_stream_config_free_is_immutable(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"width": 0.2}))
    original = StreamConfig.load(path)

    freed = original.free("width", {"min": 0.1, "max": 0.3})

    assert original.params["width"].is_fixed()
    assert not freed.params["width"].is_fixed()
    assert freed is not original


def test_stream_config_free_requires_dict(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"width": 0.2}))
    config = StreamConfig.load(path)
    with pytest.raises(TypeError):
        config.free("width", 0.4)


def test_stream_config_richness_requires_single_kind(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"richness": {"mass": 1.0e4, "nstars": 2000}}))
    with pytest.raises(ValueError):
        StreamConfig.load(path)


# ---------------------------------------------------------------------------
# BackgroundConfig
# ---------------------------------------------------------------------------


def test_background_config_load_default_light_source():
    config = BackgroundConfig.load(CONFIG_DIR / "background.yaml")
    assert config.source == "light"
    assert config.survey == "lsst"
    assert config.release == "yr1"
    # light already accounts for extinction internally -> dust correction off by default
    assert config.dust_correction_enabled is False
    assert config.study_region == {
        "center_ra": 0.0,
        "center_dec": -30.0,
        "width_deg": 70.0,
        "height_deg": 20.0,
    }
    assert len(config.cuts) == 2


def test_background_config_dust_correction_defaults_true_for_data_file(tmp_path):
    path = tmp_path / "background.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "background": {
                    "source": "data_file",
                    "survey": "lsst",
                    "release": "dp2",
                },
            }
        )
    )
    config = BackgroundConfig.load(path)
    assert config.dust_correction_enabled is True


def test_background_config_dust_correction_explicit_overrides_default(tmp_path):
    path = tmp_path / "background.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "background": {
                    "source": "data_file",
                    "dust_correction": {"enabled": False},
                },
            }
        )
    )
    config = BackgroundConfig.load(path)
    assert config.dust_correction_enabled is False


# ---------------------------------------------------------------------------
# MatchedFilterConfig
# ---------------------------------------------------------------------------


def test_matched_filter_config_load():
    config = MatchedFilterConfig.load(CONFIG_DIR / "matched_filter.yaml")
    assert config.bands == ["g", "r"]
    assert config.reference_isochrone == {"age": 12.5, "z": 0.0002}
    assert config.pixelization["nside"] == 128
    assert config.finalize_config["enabled"] is False


def test_matched_filter_config_distance_moduli_fixed():
    config = MatchedFilterConfig.load(CONFIG_DIR / "matched_filter.yaml")
    assert config.distance_moduli() == [17.5]


def test_matched_filter_config_distance_moduli_scan(tmp_path):
    path = tmp_path / "mf.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reference_isochrone": {"age": 12.5, "z": 0.0002},
                "bands": ["g", "r"],
                "pixelization": {},
                "distance": {"mode": "scan", "min": 14.0, "max": 15.0, "step": 0.5},
            }
        )
    )
    config = MatchedFilterConfig.load(path)
    np.testing.assert_allclose(config.distance_moduli(), [14.0, 14.5, 15.0])


def test_matched_filter_config_distance_moduli_fixed_without_value_raises(tmp_path):
    path = tmp_path / "mf.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reference_isochrone": {"age": 12.5, "z": 0.0002},
                "bands": ["g", "r"],
                "pixelization": {},
                "distance": {"mode": "fixed"},
            }
        )
    )
    config = MatchedFilterConfig.load(path)
    with pytest.raises(ValueError):
        config.distance_moduli()


# ---------------------------------------------------------------------------
# EvalGrid / build_eval_grid
# ---------------------------------------------------------------------------


def test_eval_grid_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        EvalGrid(points=[{"a": 1}], seeds=[1, 2])


def test_build_eval_grid_no_free_parameters_raises(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"width": 0.2, "age": 12.0}))
    config = StreamConfig.load(path)
    with pytest.raises(ValueError):
        build_eval_grid(config)


def test_build_eval_grid_cartesian_product_size(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "orientation": {"min": 0.0, "max": 180.0},
                "age": {"values": [10.0, 12.0, 13.5]},
            }
        )
    )
    config = StreamConfig.load(path)
    grid = build_eval_grid(config, n_points_per_range=4)
    assert len(grid.points) == 4 * 3
    assert len(grid.seeds) == len(grid.points)
    for point in grid.points:
        assert set(point) == {"orientation", "age"}
        assert point["age"] in (10.0, 12.0, 13.5)
        assert 0.0 <= point["orientation"] <= 180.0


def test_build_eval_grid_discrete_truncation(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"age": {"values": [10.0, 11.0, 12.0, 13.0, 13.5]}}))
    config = StreamConfig.load(path)
    grid = build_eval_grid(config, n_points_discrete=3)
    ages = {point["age"] for point in grid.points}
    assert len(ages) == 3
    assert ages.issubset({10.0, 11.0, 12.0, 13.0, 13.5})


def test_build_eval_grid_reproducible_with_same_seed(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump({"orientation": {"min": 0.0, "max": 180.0}}))
    config = StreamConfig.load(path)
    grid1 = build_eval_grid(config, seed=7)
    grid2 = build_eval_grid(config, seed=7)
    assert grid1.seeds == grid2.seeds
    assert grid1.points == grid2.points


def test_build_eval_grid_eval_config_matches_real_file():
    config = StreamConfig.load(CONFIG_DIR / "streams" / "eval_grid.yaml")
    grid = build_eval_grid(config)
    assert len(grid.points) == 6
    values = sorted(point["richness"] for point in grid.points)
    assert values == [26.0, 27.2, 28.4, 29.6, 30.8, 32.0]
