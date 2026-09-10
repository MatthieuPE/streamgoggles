"""Test evaluation/{metrics,baseline_threshold,completeness_purity}.py.

metrics.py / baseline_threshold.py (pure numpy, no streamobs needed):
hand-computed values on tiny arrays, valid_mask exclusion, the
NaN-for-undefined-ratio convention.

completeness_purity.py: evaluate_on_grid real-integration test (real
streamobs background/injector + a real UNet, same fixture pattern as
test_datasets.py/test_training.py), plot_recovery_vs_parameter with real
matplotlib, and compute_completeness_purity on hand-built results frames
(both the all-true-streams case and the is_true_stream-labeled case).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.config import DistributionType, ParameterSpec, StreamConfig
from streamgoggles.datasets.stream_map_dataset import StreamMapDataset
from streamgoggles.evaluation.baseline_threshold import build_baseline
from streamgoggles.evaluation.completeness_purity import (
    compute_completeness_purity,
    evaluate_on_grid,
    plot_recovery_vs_parameter,
)
from streamgoggles.evaluation.metrics import (
    correlation,
    dice,
    iou,
    mse,
    precision_recall,
    weighted_recall,
)
from streamgoggles.injector import StreamInjector
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.models.unet import UNet
from streamgoggles.storage import BackgroundMapStore, SimulationStore
from streamgoggles.stream_sources import StreamObsSource

pytestmark = pytest.mark.evaluation


# ---------------------------------------------------------------------------
# metrics.py: shape validation
# ---------------------------------------------------------------------------


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape"):
        iou(np.ones((2, 2)), np.ones((3, 3)))


def test_bad_ndim_raises():
    with pytest.raises(ValueError, match="ny, nx"):
        iou(np.ones((2, 2, 2, 2)), np.ones((2, 2, 2, 2)))


def test_bad_valid_mask_shape_raises():
    with pytest.raises(ValueError, match="valid_mask"):
        iou(np.ones((2, 2)), np.ones((2, 2)), valid_mask=np.ones((3, 3), dtype=bool))


# ---------------------------------------------------------------------------
# iou / dice
# ---------------------------------------------------------------------------


def test_iou_hand_computed():
    pred = np.array([[1.0, 1.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    # pred_bin={(0,0),(0,1)}, target_bin={(0,0),(1,1)}
    # intersection={(0,0)}=1, union={(0,0),(0,1),(1,1)}=3
    assert iou(pred, target) == pytest.approx(1 / 3)


def test_iou_both_empty_is_nan():
    pred = np.zeros((2, 2))
    target = np.zeros((2, 2))
    assert np.isnan(iou(pred, target))


def test_iou_respects_valid_mask():
    pred = np.array([[1.0, 1.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, True], [True, False]])
    # excluding (1,1): pred_bin={(0,0),(0,1)}, target_bin={(0,0)}
    # intersection=1, union=2
    assert iou(pred, target, valid_mask=mask) == pytest.approx(0.5)


def test_dice_hand_computed():
    pred = np.array([[1.0, 1.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    # intersection=1, |pred_bin|=2, |target_bin|=2 -> dice=2*1/4=0.5
    assert dice(pred, target) == pytest.approx(0.5)


def test_dice_both_zero_is_nan():
    pred = np.zeros((2, 2))
    target = np.zeros((2, 2))
    assert np.isnan(dice(pred, target))


def test_iou_dice_accept_multichannel_input():
    pred = np.ones((3, 2, 2))
    target = np.ones((3, 2, 2))
    assert iou(pred, target) == pytest.approx(1.0)
    assert dice(pred, target) == pytest.approx(1.0)


def test_valid_mask_none_equals_all_true():
    pred = np.array([[1.0, 0.6], [0.2, 0.9]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    all_true = np.ones((2, 2), dtype=bool)
    assert iou(pred, target) == pytest.approx(iou(pred, target, valid_mask=all_true))


# ---------------------------------------------------------------------------
# precision_recall
# ---------------------------------------------------------------------------


def test_precision_recall_hand_computed():
    pred = np.array([[1.0, 1.0], [0.0, 1.0]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    # pred_bin = {(0,0),(0,1),(1,1)}; target_bin = {(0,0),(1,1)}
    # tp={(0,0),(1,1)}=2, fp={(0,1)}=1, fn={}=0
    precision, recall = precision_recall(pred, target)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)


def test_precision_nan_when_no_positive_predictions():
    pred = np.zeros((2, 2))
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    precision, recall = precision_recall(pred, target)
    assert np.isnan(precision)
    assert recall == pytest.approx(0.0)


def test_recall_nan_when_no_true_positives_exist():
    pred = np.array([[1.0, 1.0], [1.0, 1.0]])
    target = np.zeros((2, 2))
    precision, recall = precision_recall(pred, target)
    assert precision == pytest.approx(0.0)
    assert np.isnan(recall)


# ---------------------------------------------------------------------------
# mse / correlation
# ---------------------------------------------------------------------------


def test_mse_hand_computed():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.0]])
    # squared errors: 0, 1, 4, 9 -> mean=3.5
    assert mse(pred, target) == pytest.approx(3.5)


def test_mse_respects_mask():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.0]])
    mask = np.array([[True, True], [True, False]])
    # excluding (1,1) (error 9): mean(0, 1, 4) = 5/3
    assert mse(pred, target, valid_mask=mask) == pytest.approx(5 / 3)


def test_mse_nan_when_no_valid_pixels():
    pred = np.ones((2, 2))
    target = np.ones((2, 2))
    mask = np.zeros((2, 2), dtype=bool)
    assert np.isnan(mse(pred, target, valid_mask=mask))


def test_correlation_perfect_linear_relationship():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = 2.0 * pred + 1.0
    assert correlation(pred, target) == pytest.approx(1.0)


def test_correlation_nan_for_constant_array():
    pred = np.ones((2, 2))
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isnan(correlation(pred, target))


def test_correlation_nan_for_fewer_than_two_valid_pixels():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [False, False]])
    assert np.isnan(correlation(pred, target, valid_mask=mask))


# ---------------------------------------------------------------------------
# weighted_recall
# ---------------------------------------------------------------------------


def test_weighted_recall_hand_computed():
    # target: count-like values; pred binarized at threshold=0.5.
    pred = np.array([[1.0, 0.0], [1.0, 1.0]])
    target = np.array([[5.0, 3.0], [0.0, 2.0]])
    # target_bin (target>0.5): (0,0)=5, (0,1)=3, (1,1)=2 -> denom=10
    # pred_bin & target_bin: (0,0) yes (pred=1), (0,1) no (pred=0),
    #                        (1,1) yes (pred=1) -> numerator=5+2=7
    assert weighted_recall(pred, target) == pytest.approx(7 / 10)


def test_weighted_recall_nan_when_no_true_signal():
    pred = np.ones((2, 2))
    target = np.zeros((2, 2))
    assert np.isnan(weighted_recall(pred, target))


def test_weighted_recall_full_recovery_is_one():
    pred = np.array([[1.0, 1.0], [0.0, 1.0]])
    target = np.array([[5.0, 3.0], [0.0, 2.0]])
    assert weighted_recall(pred, target) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# baseline_threshold.py
# ---------------------------------------------------------------------------


def test_build_baseline_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        build_baseline(np.ones((2, 2)), np.ones((3, 3), dtype=bool))


def test_build_baseline_hand_computed():
    finalized_map = np.array([0.0, 1.0, 2.0, 100.0])
    valid_mask = np.array([True, True, True, True])
    # mean=25.75, std of [0,1,2,100] -> threshold = mean + 1*std
    threshold = finalized_map.mean() + 1.0 * finalized_map.std()
    expected = (finalized_map > threshold).astype(int)
    result = build_baseline(finalized_map, valid_mask, k=1.0)
    np.testing.assert_array_equal(result, expected)


def test_build_baseline_invalid_pixels_always_zero():
    finalized_map = np.array([0.0, 1000.0, 2.0])
    valid_mask = np.array([True, False, True])
    result = build_baseline(finalized_map, valid_mask, k=0.0)
    assert result[1] == 0


def test_build_baseline_all_invalid_returns_all_zero():
    finalized_map = np.array([1.0, 2.0, 3.0])
    valid_mask = np.zeros(3, dtype=bool)
    result = build_baseline(finalized_map, valid_mask)
    np.testing.assert_array_equal(result, np.zeros(3, dtype=int))


def test_build_baseline_2d_shape_preserved():
    finalized_map = np.random.default_rng(0).normal(size=(5, 5))
    valid_mask = np.ones((5, 5), dtype=bool)
    result = build_baseline(finalized_map, valid_mask)
    assert result.shape == (5, 5)
    assert set(np.unique(result)).issubset({0, 1})


# ---------------------------------------------------------------------------
# compute_completeness_purity
# ---------------------------------------------------------------------------


def test_compute_completeness_purity_missing_metric_raises():
    results = pd.DataFrame({"dice": [0.9, 0.1]})
    with pytest.raises(KeyError):
        compute_completeness_purity(results, lambda x: x > 0.5, metric="missing")


def test_compute_completeness_purity_all_true_streams():
    results = pd.DataFrame({"dice": [0.9, 0.6, 0.2, 0.1]})
    out = compute_completeness_purity(results, lambda x: x > 0.5, metric="dice")
    assert out["n_streams_total"] == 4
    assert out["n_detected"] == 2
    assert out["completeness"] == pytest.approx(0.5)
    # No is_true_stream column -> every row is a true stream -> purity trivially 1.0
    assert out["purity"] == pytest.approx(1.0)


def test_compute_completeness_purity_with_is_true_stream_column():
    results = pd.DataFrame(
        {
            "dice": [0.9, 0.8, 0.7, 0.2],
            "is_true_stream": [True, True, False, False],
        }
    )
    # detected: dice>0.5 -> rows 0,1,2 detected; row 3 not.
    out = compute_completeness_purity(results, lambda x: x > 0.5, metric="dice")
    assert out["n_streams_total"] == 2
    assert out["n_detected"] == 3
    # true detections among true streams: rows 0,1 -> 2
    assert out["completeness"] == pytest.approx(2 / 2)
    # purity: true detections / all detections = 2/3
    assert out["purity"] == pytest.approx(2 / 3)


def test_compute_completeness_purity_no_detections_purity_nan():
    results = pd.DataFrame({"dice": [0.1, 0.2]})
    out = compute_completeness_purity(results, lambda x: x > 0.9, metric="dice")
    assert out["n_detected"] == 0
    assert np.isnan(out["purity"])


def test_compute_completeness_purity_no_true_streams_completeness_nan():
    results = pd.DataFrame({"dice": [0.9], "is_true_stream": [False]})
    out = compute_completeness_purity(results, lambda x: x > 0.5, metric="dice")
    assert out["n_streams_total"] == 0
    assert np.isnan(out["completeness"])


# ---------------------------------------------------------------------------
# plot_recovery_vs_parameter (real matplotlib)
# ---------------------------------------------------------------------------


def test_plot_recovery_vs_parameter_missing_column_raises():
    results = pd.DataFrame({"nstars": [100, 200], "dice": [0.5, 0.6]})
    with pytest.raises(KeyError):
        plot_recovery_vs_parameter(results, param_name="missing", metric="dice")


def test_plot_recovery_vs_parameter_plots_one_line():
    results = pd.DataFrame({"nstars": [200, 100, 300], "dice": [0.6, 0.5, 0.7]})
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_recovery_vs_parameter(results, param_name="nstars", metric="dice", ax=ax)
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "nstars"
    assert ax.get_ylabel() == "dice"
    plt.close(fig)


def test_plot_recovery_vs_parameter_with_baseline_plots_two_lines():
    results = pd.DataFrame(
        {"nstars": [200, 100], "dice": [0.6, 0.5], "baseline_dice": [0.3, 0.2]}
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_recovery_vs_parameter(
        results,
        param_name="nstars",
        metric="dice",
        ax=ax,
        baseline_metric="baseline_dice",
    )
    assert len(ax.lines) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# evaluate_on_grid: real integration (real streamobs background/injector,
# real UNet, real eval-mode StreamMapDataset)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_background(tmp_path_factory):
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    store = BackgroundMapStore(tmp_path_factory.mktemp("bgmaps"))
    good = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    decoy = ShiftedColorBoxFilter(reference_filter=good, color_shift=0.5)
    filters = {"good": good, "decoy": decoy}
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=[16.8],
        finalize_cfg=None,
        store=store,
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_configs={"good": {"age": 12.5, "z": 0.0002}, "decoy": {"shift": 0.5}},
    )
    return bg, filters, pix


@pytest.fixture
def real_injector(real_background):
    bg, filters, pix = real_background
    return StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )


def _fixed_params_dict(**overrides):
    base = {
        "morphology": "uniform",
        "richness": 2500,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }
    base.update(overrides)
    return {
        name: ParameterSpec(name=name, dist_type=DistributionType.FIXED, value=value)
        for name, value in base.items()
    }


@pytest.fixture
def eval_config():
    """richness is DISCRETE with 2 values -> a small, fast 2-point eval grid."""
    params = _fixed_params_dict()
    params["richness"] = ParameterSpec(
        name="richness", dist_type=DistributionType.DISCRETE, values=[2000, 4000]
    )
    return StreamConfig(
        params=params, background_fraction=0.0, persist=False, richness_kind="nstars"
    )


@pytest.fixture
def eval_dataset(real_background, real_injector, eval_config, tmp_path):
    bg, _filters, _pix = real_background
    sim_store = SimulationStore(tmp_path / "sims")
    return StreamMapDataset(
        config=eval_config,
        background=bg,
        injector=real_injector,
        store=sim_store,
        eval_mode=True,
    )


@pytest.fixture
def eval_model():
    return UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="softplus")


def test_evaluate_on_grid_not_eval_mode_raises(
    real_background, real_injector, eval_config, tmp_path, eval_model
):
    bg, _filters, _pix = real_background
    sim_store = SimulationStore(tmp_path / "sims2")
    training_dataset = StreamMapDataset(
        config=eval_config,
        background=bg,
        injector=real_injector,
        store=sim_store,
        eval_mode=False,
    )
    with pytest.raises(ValueError, match="eval_mode"):
        evaluate_on_grid(eval_model, training_dataset, metrics=["mse"])


def test_evaluate_on_grid_unknown_metric_raises(eval_model, eval_dataset):
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_on_grid(eval_model, eval_dataset, metrics=["not_a_real_metric"])


def test_evaluate_on_grid_returns_expected_shape_and_columns(eval_model, eval_dataset):
    results = evaluate_on_grid(eval_model, eval_dataset, metrics=["mse", "dice"])
    assert len(results) == len(eval_dataset) == 2
    assert "id" in results.columns
    assert "nstars" in results.columns
    assert "mse" in results.columns
    assert "dice" in results.columns
    assert set(results["nstars"]) == {2000, 4000}
    assert set(results["id"]) == {0, 1}


def test_evaluate_on_grid_precision_recall_both_present(eval_model, eval_dataset):
    results = evaluate_on_grid(
        eval_model, eval_dataset, metrics=["precision", "recall"]
    )
    assert "precision" in results.columns
    assert "recall" in results.columns


def test_evaluate_on_grid_deterministic_across_calls(eval_model, eval_dataset):
    torch.manual_seed(0)
    results_a = evaluate_on_grid(eval_model, eval_dataset, metrics=["mse"])
    results_b = evaluate_on_grid(eval_model, eval_dataset, metrics=["mse"])
    # check_like=True: column/row order isn't guaranteed to be stable across
    # calls once a sample round-trips through SimulationStore's on-disk
    # persistence (eval mode always persists) -- only the actual values need
    # to match, which is what "deterministic" is actually testing for here.
    pd.testing.assert_frame_equal(results_a, results_b, check_like=True)
