"""Test footprint-level detection evaluation.

Scoring and plotting are checked on synthetic HEALPix maps (fast, exact).
The realization loop is checked end to end against the real streamobs
injection machinery at a coarse nside, with an untrained model -- what is
under test is the plumbing (rows, reproducibility, independence, the
vanished-label path), not detection quality.
"""

import healpy as hp
import matplotlib
import numpy as np
import pytest

from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.evaluation import footprint
from streamgoggles.evaluation.completeness_purity import aggregate_over_replicates
from streamgoggles.evaluation.footprint import (
    background_only_sky,
    evaluate_footprint_realizations,
    plot_confusion_matrix,
    plot_detection_rates,
    score_footprint,
    tiles_around_stream,
)
from streamgoggles.injector import StreamInjector
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.models.unet import UNet
from streamgoggles.storage import BackgroundMapStore
from streamgoggles.stream_sources import StreamObsSource

matplotlib.use("Agg")

pytestmark = pytest.mark.footprint


# ---------------------------------------------------------------------------
# score_footprint on synthetic HEALPix maps
# ---------------------------------------------------------------------------


def _synthetic_maps(nside=16):
    npix = hp.nside2npix(nside)
    label = np.zeros(npix)
    prediction = np.zeros(npix)
    covered = np.zeros(npix, dtype=bool)
    covered[:100] = True
    label[:10] = 1.0  # 10 true stream pixels
    prediction[:6] = 0.9  # 6 of them found -> tp=6, fn=4
    prediction[10:13] = 0.9  # 3 false alarms among the 90 non-stream -> fp=3, tn=87
    return prediction, label, covered


def test_score_footprint_counts_and_rates():
    prediction, label, covered = _synthetic_maps()
    scores = score_footprint(prediction, label, covered)

    assert (scores["tp"], scores["fn"], scores["fp"], scores["tn"]) == (6, 4, 3, 87)
    assert scores["tpr"] == pytest.approx(6 / 10)
    assert scores["fnr"] == pytest.approx(4 / 10)
    assert scores["fpr"] == pytest.approx(3 / 90)
    assert scores["tnr"] == pytest.approx(87 / 90)
    assert scores["precision"] == pytest.approx(6 / 9)
    assert scores["n_true_pixels"] == 10


def test_score_footprint_rows_sum_to_one():
    prediction, label, covered = _synthetic_maps()
    scores = score_footprint(prediction, label, covered)
    assert scores["tpr"] + scores["fnr"] == pytest.approx(1.0)
    assert scores["fpr"] + scores["tnr"] == pytest.approx(1.0)


def test_score_footprint_never_scores_uncovered_or_nan_pixels():
    """The trap this guards: stitched maps hold NaN where no window reached
    or the footprint has a hole, and `NaN > 0.5` is False -- so an unscored
    NaN pixel would be silently counted as correctly rejected background,
    inflating the true-negative rate by however much sky was never looked at."""
    prediction, label, covered = _synthetic_maps()
    baseline = score_footprint(prediction, label, covered)

    # Fill every uncovered pixel with NaN, and punch NaN holes inside coverage.
    prediction_with_gaps = np.where(covered, prediction, np.nan)
    label_with_gaps = np.where(covered, label, np.nan)
    prediction_with_gaps[50:60] = np.nan

    scores = score_footprint(prediction_with_gaps, label_with_gaps, covered)

    assert scores["tn"] == baseline["tn"] - 10, "NaN holes must not be counted"
    assert scores["tp"] + scores["fp"] + scores["tn"] + scores["fn"] == 90


def test_score_footprint_reports_an_empty_label_as_undefined_not_zero():
    """A faint enough stream leaves no pixel above the detection threshold.
    That must read as 'nothing to detect' (NaN), not as 'detected nothing'
    (0.0) -- the two mean opposite things about the model."""
    prediction, label, covered = _synthetic_maps()
    scores = score_footprint(prediction, np.zeros_like(label), covered)

    assert scores["n_true_pixels"] == 0
    assert np.isnan(scores["tpr"]) and np.isnan(scores["fnr"])
    assert scores["precision"] == 0.0, "flagged pixels, none of them stream"
    assert np.isfinite(scores["fpr"]) and np.isfinite(scores["tnr"])


# ---------------------------------------------------------------------------
# tiles_around_stream
# ---------------------------------------------------------------------------


def test_tiles_around_stream_returns_nothing_when_no_stream_pixels():
    nside = 64
    npix = hp.nside2npix(nside)
    pix = PixelizationSpec(nside=nside, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    full_sky = {
        "stream_raw_full": [np.zeros(npix)],
        "valid_mask_full": np.ones(npix, dtype=bool),
    }
    assert tiles_around_stream(full_sky, channel=0, pix=pix) == []


def test_tiles_around_stream_overlap_for_the_stitcher():
    nside = 64
    npix = hp.nside2npix(nside)
    pix = PixelizationSpec(nside=nside, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    stream = np.zeros(npix)
    stream[
        hp.query_disc(nside, hp.ang2vec(10.0, -30.0, lonlat=True), np.radians(2))
    ] = 5
    full_sky = {
        "stream_raw_full": [stream],
        "valid_mask_full": np.ones(npix, dtype=bool),
    }

    overlapping = tiles_around_stream(full_sky, 0, pix, stride_fraction=0.5)
    abutting = tiles_around_stream(full_sky, 0, pix, stride_fraction=1.0)

    assert len(overlapping) > len(abutting) > 0


# ---------------------------------------------------------------------------
# Plots (smoke: they must render without error on real-shaped input)
# ---------------------------------------------------------------------------


def test_plot_confusion_matrix_renders_including_undefined_cells():
    ax = plot_confusion_matrix(
        {"tpr": float("nan"), "fnr": float("nan"), "fpr": 0.01, "tnr": 0.99}
    )
    texts = [t.get_text() for t in ax.texts]
    assert any("n/a" in t for t in texts), "an undefined rate must render as n/a"


def test_plot_detection_rates_marks_vanished_labels():
    import pandas as pd

    aggregated = pd.DataFrame(
        {
            "richness": [31.0, 33.0, 35.0],
            "tpr_mean": [0.9, 0.5, float("nan")],
            "tpr_std": [0.05, 0.2, float("nan")],
            "fpr_mean": [0.01, 0.01, 0.01],
            "fpr_std": [0.002, 0.002, 0.002],
        }
    )
    ax_found, ax_false = plot_detection_rates(
        aggregated, "richness", training_range=(31, 34)
    )
    labels = ax_found.get_legend_handles_labels()[1]
    assert any("label vanished" in label for label in labels)
    # Missed is the complement of found; it is not drawn.
    assert not any("missed" in label for label in labels)
    # Without a no-stream column there is only the with-stream line.
    false_labels = ax_false.get_legend_handles_labels()[1]
    assert not any("no stream" in label for label in false_labels)


def test_plot_detection_rates_draws_the_no_stream_control():
    import pandas as pd

    aggregated = pd.DataFrame(
        {
            "richness": [31.0, 33.0],
            "tpr_mean": [0.9, 0.5],
            "tpr_std": [0.05, 0.2],
            "fpr_mean": [0.012, 0.014],
            "fpr_std": [0.002, 0.002],
            "fpr_no_stream_mean": [0.009, 0.009],
            "fpr_no_stream_std": [0.001, 0.001],
        }
    )
    _, ax_false = plot_detection_rates(aggregated, "richness")
    labels = ax_false.get_legend_handles_labels()[1]
    assert any("no stream" in label for label in labels)


# ---------------------------------------------------------------------------
# evaluate_footprint_realizations, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def injector(tmp_path_factory):
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    good = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    filters = {
        "good": good,
        "decoy": ShiftedColorBoxFilter(reference_filter=good, color_shift=0.5),
    }
    background = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={"seed": 1},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=[16.8],
        finalize_cfg=None,
        store=BackgroundMapStore(tmp_path_factory.mktemp("bgmaps")),
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_configs={"good": {}, "decoy": {}},
    )
    return StreamInjector(
        background=background,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
        label_policy="stream_detection",
        count_threshold=1.0,
    )


def _identity_transform(item):
    return item


def _param_sets():
    base = {
        "morphology": "uniform",
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }
    return [{**base, "richness": 3000}, {**base, "richness": 6000}]


def test_evaluate_footprint_realizations_one_row_per_realization(injector):
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    results = evaluate_footprint_realizations(
        model, injector, _identity_transform, _param_sets(), n_realizations=2, channel=0
    )

    assert len(results) == 2 * 2
    for column in (
        "richness",
        "realization",
        "nstars",
        "n_tiles",
        "tp",
        "fp",
        "tn",
        "fn",
        "tpr",
        "fnr",
        "fpr",
        "tnr",
        "precision",
        "n_true_pixels",
        "fp_no_stream",
        "tn_no_stream",
        "fpr_no_stream",
    ):
        assert column in results.columns
    assert (results["n_tiles"] > 0).all()
    # The control scores the same tiles, so it covers the same non-stream
    # pixels plus the stream's own (which are background there).
    with_stream = results["fp"] + results["tn"]
    control = results["fp_no_stream"] + results["tn_no_stream"]
    assert (control >= with_stream).all()
    assert (control <= with_stream + results["n_true_pixels"]).all()

    # Aggregating per richness is what the detection-rate plot consumes.
    aggregated = aggregate_over_replicates(
        results, metrics=["tpr", "fnr", "fpr", "tnr"], group_by=["richness"]
    )
    assert list(aggregated["richness"]) == [3000, 6000]
    assert (aggregated["tpr_n"] <= 2).all()


def test_evaluate_footprint_realizations_is_reproducible_and_independent(injector):
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    kwargs = {"n_realizations": 2, "channel": 0, "seed": 11}

    first = evaluate_footprint_realizations(
        model, injector, _identity_transform, _param_sets()[:1], **kwargs
    )
    again = evaluate_footprint_realizations(
        model, injector, _identity_transform, _param_sets()[:1], **kwargs
    )

    counts = ["tp", "fp", "tn", "fn"]
    assert first[counts].equals(again[counts]), "same seed must reproduce exactly"
    # Two realizations of the same parameters are different skies.
    assert not first.loc[0, counts].equals(first.loc[1, counts])


def test_evaluate_footprint_realizations_keeps_a_vanished_stream(monkeypatch, injector):
    """A stream too faint to leave any stream-only pixel has nothing to tile
    around. It must still appear as a row (with undefined rates), or the
    faint end of a detection curve would silently lose its failures."""
    import torch

    monkeypatch.setattr(footprint, "tiles_around_stream", lambda *a, **k: [])
    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")

    results = evaluate_footprint_realizations(
        model,
        injector,
        _identity_transform,
        _param_sets()[:1],
        n_realizations=1,
        channel=0,
    )

    assert len(results) == 1
    assert results.loc[0, "n_tiles"] == 0
    assert np.isnan(results.loc[0, "tpr"])
    assert np.isnan(results.loc[0, "fpr_no_stream"])


def test_background_only_sky_is_the_same_sky_without_the_stream(injector):
    sky = injector.inject_stream_full_sky(_param_sets()[1], np.random.default_rng(3))
    empty = background_only_sky(injector, sky)

    valid = sky["valid_mask_full"]
    assert empty["channels"] == sky["channels"]
    for c in range(len(sky["channels"])):
        assert not empty["stream_raw_full"][c].any()
        # Injection is additive: removing the stream leaves the background.
        np.testing.assert_allclose(
            (sky["map_full"][c] - empty["map_full"][c])[valid],
            sky["stream_raw_full"][c][valid],
        )
    assert sky["stream_raw_full"][0].any(), "the stream sky must hold a stream"


def test_plot_detection_rates_flags_partially_vanished_points_and_clips_bars():
    import pandas as pd

    aggregated = pd.DataFrame(
        {
            "richness": [31.0, 36.0],
            "tpr_mean": [0.95, 0.0],
            "tpr_std": [0.2, 0.0],
            "tpr_n": [30, 11],
            "fpr_mean": [0.01, 0.01],
            "fpr_std": [0.2, 0.0],
            "n_true_pixels_n": [30, 30],
        }
    )
    axes = plot_detection_rates(aggregated, "richness")

    texts = [t.get_text() for t in axes[0].texts]
    assert texts == ["11/30"], "only the partially-vanished point is annotated"
    for ax in axes:
        for line in ax.collections:
            segments = getattr(line, "get_segments", list)()
            for segment in segments:
                ys = np.asarray(segment)[:, 1]
                assert ys.min() >= 0.0 and ys.max() <= 1.0, "bars must stay in [0, 1]"
