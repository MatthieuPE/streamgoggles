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
    THRESHOLD_GRID,
    background_only_sky,
    band_snr,
    detection_at_false_alarm_rate,
    detection_metrics,
    evaluate_footprint_realizations,
    plot_confusion_matrix,
    plot_detection_metrics,
    plot_detection_rates,
    plot_stream_detection,
    score_footprint,
    score_streams_on_sky,
    stream_detection,
    stream_frame_coordinates,
    tiles_around_stream,
    track_band_statistics,
)
from streamgoggles.injector import StreamInjector
from streamgoggles.matched_filter import (
    ColorBoxFilter,
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


def test_threshold_grid_is_sorted_and_contains_one_half():
    assert np.all(np.diff(THRESHOLD_GRID) > 0)
    assert 0.5 in THRESHOLD_GRID
    assert THRESHOLD_GRID[0] < 1e-5 and THRESHOLD_GRID[-1] > 1 - 1e-5


def test_score_footprint_threshold_sweep_matches_the_confusion_counts():
    prediction, label, covered = _synthetic_maps()
    prediction[20:40] = np.linspace(0.05, 0.95, 20)  # spread some background
    prediction = np.where(covered, prediction, np.nan)
    thresholds = np.array([0.0, 0.3, 0.5, 0.8, 0.99])
    scores = score_footprint(prediction, label, covered, thresholds=thresholds)

    at_half = list(thresholds).index(0.5)
    assert scores["n_above_stream"][at_half] == scores["tp"]
    assert scores["n_above_background"][at_half] == scores["fp"]
    # Every threshold agrees with a direct count, NaN / uncovered excluded.
    is_stream = label > 0.5
    scored = covered & np.isfinite(prediction)
    for i, t in enumerate(thresholds):
        assert scores["n_above_stream"][i] == np.count_nonzero(
            scored & is_stream & (prediction > t)
        )
        assert scores["n_above_background"][i] == np.count_nonzero(
            scored & ~is_stream & (prediction > t)
        )
    assert np.all(np.diff(scores["n_above_background"]) <= 0)


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


@pytest.fixture(scope="module")
def query_injector(tmp_path_factory):
    """Three trial distances 0.5 apart: what a query-distance model needs."""
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    filters = {
        "good": StreamobsSplineFilter(
            iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
        ),
        "decoy": ColorBoxFilter(
            color_range=(1.2, 1.5), mag_range=(18.0, 24.5), namespace="lsst_yr1"
        ),
    }
    background = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={"seed": 1},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=[16.3, 16.8, 17.3],
        finalize_cfg=None,
        store=BackgroundMapStore(tmp_path_factory.mktemp("query_bgmaps")),
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


def test_evaluate_footprint_realizations_threshold_sweep(injector):
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    results = evaluate_footprint_realizations(
        model,
        injector,
        _identity_transform,
        _param_sets()[:1],
        n_realizations=2,
        channel=0,
        thresholds=THRESHOLD_GRID,
    )
    at_half = int(np.flatnonzero(THRESHOLD_GRID == 0.5)[0])
    for _, row in results.iterrows():
        assert len(row["n_above_stream"]) == len(THRESHOLD_GRID)
        assert row["n_above_stream"][at_half] == row["tp"]
        assert row["n_above_background"][at_half] == row["fp"]
        assert row["n_above_no_stream"][at_half] == row["fp_no_stream"]
        # Counts never exceed the pixels scored, and fall as the threshold rises.
        assert row["n_above_no_stream"][0] <= row["fp_no_stream"] + row["tn_no_stream"]
        assert np.all(np.diff(row["n_above_no_stream"]) <= 0)


def test_score_streams_on_sky_judges_each_stream_on_its_own_track(injector):
    """Two streams on one sky: one row each, each along its own track, and the
    rows feed stream_detection like single-stream realizations."""
    import healpy as hp
    import pandas as pd
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    footprint = np.flatnonzero(injector.background.footprint)
    ra, dec = hp.pix2ang(
        injector.pix.nside, int(footprint[len(footprint) // 2]), lonlat=True
    )
    params = {**_param_sets()[1], "orientation": 30.0}
    full_sky = injector.inject_streams_full_sky(
        [params, params], np.random.default_rng(3), centers=[(ra, dec), (ra, dec + 2)]
    )
    rows = score_streams_on_sky(
        model,
        injector,
        _identity_transform,
        full_sky,
        channel=0,
        thresholds=THRESHOLD_GRID,
        rng=np.random.default_rng(4),
        n_null_bands=20,
    )

    assert [row["stream"] for row in rows] == [0, 1]
    for row in rows:
        assert len(row["n_above_band"]) == len(THRESHOLD_GRID)
        assert row["band_pixels"] > 0
    table = pd.DataFrame([{**row, "richness": params["richness"]} for row in rows])
    detected = stream_detection(table, THRESHOLD_GRID, at=(0.5,))
    assert detected["n_realizations"].iloc[0] == 2


def test_a_query_distance_model_is_scored_against_its_distances_label(
    query_injector,
):
    """A model answering for one queried distance outputs one map; it must be
    scored against the matched filter's label at that distance, exactly the
    label a full-stack model reading that channel would be scored against."""
    import torch

    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )

    channels = [
        (ch["filter"], ch["distance_modulus"])
        for ch in query_injector.inject_stream_full_sky(
            _param_sets()[1], np.random.default_rng(0)
        )["channels"]
    ]
    good_at_query = channels.index(("good", 16.8))
    params = [{**_param_sets()[1], "distance_modulus": 16.8}]

    torch.manual_seed(0)
    query_model = UNet(
        in_channels=QueryDistanceTransform.n_channels,
        out_channels=1,
        base_width=4,
        depth=1,
        head="sigmoid",
    )
    queried = evaluate_footprint_realizations(
        query_model,
        query_injector,
        QueryDistanceTransform(StreamMapTransform(), query_grid=[16.8], query=16.8),
        params,
        n_realizations=2,
        channel=0,
        label_channel=good_at_query,
        no_stream_control=False,
    )
    full_stack_model = UNet(
        in_channels=len(channels),
        out_channels=len(channels),
        base_width=4,
        depth=1,
        head="sigmoid",
    )
    full = evaluate_footprint_realizations(
        full_stack_model,
        query_injector,
        _identity_transform,
        params,
        n_realizations=2,
        channel=good_at_query,
        no_stream_control=False,
    )
    assert (queried["n_tiles"] > 0).all()
    assert list(queried["n_true_pixels"]) == list(full["n_true_pixels"])
    assert (queried["n_true_pixels"] > 0).all()


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

    swept = evaluate_footprint_realizations(
        model,
        injector,
        _identity_transform,
        _param_sets()[:1],
        n_realizations=1,
        channel=0,
        thresholds=THRESHOLD_GRID,
    )
    assert swept.loc[0, "n_above_stream"] is None
    assert swept.loc[0, "n_above_no_stream"] is None


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


# ---------------------------------------------------------------------------
# detection_metrics / plot_detection_metrics
# ---------------------------------------------------------------------------


def _counted_rows():
    """Two SB values, two realizations each, three thresholds."""
    import pandas as pd

    def row(sb, above_stream, above_background, n_true, n_background):
        return {
            "richness": sb,
            "n_above_stream": np.array(above_stream),
            "n_above_background": np.array(above_background),
            "n_true_pixels": n_true,
            "fp": above_background[1],
            "tn": n_background - above_background[1],
        }

    return pd.DataFrame(
        [
            row(32.0, [100, 60, 10], [500, 40, 0], 100, 10_000),
            row(32.0, [100, 40, 0], [300, 20, 0], 100, 10_000),
            row(34.0, [50, 0, 0], [800, 30, 0], 50, 20_000),
            {
                "richness": 34.0,
                "n_above_stream": None,
                "n_above_background": None,
                "n_true_pixels": 0,
                "fp": 0,
                "tn": 0,
            },
        ]
    )


def test_detection_metrics_pools_realizations_before_dividing():
    thresholds = np.array([0.1, 0.5, 0.9])
    m = detection_metrics(_counted_rows(), thresholds).set_index(
        ["richness", "threshold"]
    )

    row = m.loc[(32.0, 0.5)]
    assert (row.S_s, row.S_t, row.B_s, row.B_t) == (100, 200, 60, 20_000)
    assert row.completeness == pytest.approx(0.5)
    assert row.contamination == pytest.approx(60 / 20_000)
    assert row.contrast == pytest.approx(0.5 / (60 / 20_000))


def test_detection_metrics_floors_contamination_and_leaves_contrast_undefined():
    thresholds = np.array([0.1, 0.5, 0.9])
    m = detection_metrics(_counted_rows(), thresholds).set_index(
        ["richness", "threshold"]
    )

    # Nothing flagged in the background: F uses one pixel, never 0.
    strict = m.loc[(32.0, 0.9)]
    assert strict.B_s == 0
    assert strict.contamination == pytest.approx(1 / 20_000)
    assert np.isfinite(strict.contrast)
    # Nothing found in the stream: the contrast is undefined, not zero.
    assert np.isnan(m.loc[(34.0, 0.5)].contrast)
    # The row without counts (a vanished stream) is skipped, not counted as 0.
    assert m.loc[(34.0, 0.1)].S_t == 50


def test_detection_metrics_selects_nearest_thresholds():
    thresholds = np.array([0.1, 0.5, 0.9])
    m = detection_metrics(_counted_rows(), thresholds, at=(0.45, 0.95))
    assert sorted(m["threshold"].unique()) == [0.5, 0.9]


def test_plot_detection_metrics_renders_three_panels_with_hollow_low_counts():
    thresholds = np.array([0.1, 0.5, 0.9])
    m = detection_metrics(_counted_rows(), thresholds)
    axes = plot_detection_metrics(m, training_range=(31, 34))

    assert len(axes) == 3
    assert axes[1].get_yscale() == "log" and axes[2].get_yscale() == "log"
    hollow = [
        line
        for line in axes[0].get_lines()
        if line.get_markerfacecolor() == "white" and len(line.get_xdata())
    ]
    assert hollow, "points with too few stream pixels must be hollow"
    widths = {line.get_label(): line.get_linewidth() for line in axes[0].get_lines()}
    assert widths["threshold 0.5"] > widths["threshold 0.1"]


def test_detection_metrics_matches_the_confusion_counts_end_to_end(injector):
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    results = evaluate_footprint_realizations(
        model,
        injector,
        _identity_transform,
        _param_sets(),
        n_realizations=2,
        channel=0,
        thresholds=THRESHOLD_GRID,
    )
    m = detection_metrics(results, THRESHOLD_GRID, at=(0.5,)).set_index("richness")
    for richness, group in results.groupby("richness"):
        row = m.loc[richness]
        assert row.S_s == group["tp"].sum()
        assert row.S_t == group["tp"].sum() + group["fn"].sum()
        assert row.B_s == group["fp"].sum()
        assert row.B_t == group["fp"].sum() + group["tn"].sum()


# ---------------------------------------------------------------------------
# Track bands and per-stream detection
# ---------------------------------------------------------------------------


def test_stream_frame_coordinates_invert_the_injector_placement():
    """Recover (phi1, phi2) of a placed stream from its ra/dec, with the same
    frame convention as injector.place_stream_in_footprint (gala)."""
    import pandas as pd

    from streamgoggles.injector import place_stream_in_footprint

    nside = 64
    phi1, phi2 = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(-0.6, 0.6, 7))
    stream = pd.DataFrame({"phi1": phi1.ravel(), "phi2": phi2.ravel()})
    footprint = np.zeros(hp.nside2npix(nside), dtype=bool)
    footprint[hp.ang2pix(nside, 20.0, -35.0, lonlat=True)] = True

    placed = place_stream_in_footprint(
        stream, footprint, nside, np.random.default_rng(0), rotation_deg=37.0
    )
    placement = placed.attrs["placement"]
    assert placement["rotation_deg"] == 37.0

    vectors = np.array(
        hp.ang2vec(placed["ra"].to_numpy(), placed["dec"].to_numpy(), lonlat=True)
    )
    got_phi1, got_phi2 = stream_frame_coordinates(vectors, **placement)
    np.testing.assert_allclose(got_phi1, stream["phi1"], atol=1e-6)
    np.testing.assert_allclose(got_phi2, stream["phi2"], atol=1e-6)


def _band_maps(nside=256):
    """A stream band flagged at 0.9 inside a 15-degree covered disc, empty elsewhere."""
    npix = hp.nside2npix(nside)
    placement = {"center_ra": 30.0, "center_dec": -30.0, "rotation_deg": 20.0}
    width, length = 0.5, 8.0
    vectors = np.array(hp.pix2vec(nside, np.arange(npix))).T
    phi1, phi2 = stream_frame_coordinates(vectors, **placement)
    band = (np.abs(phi1) <= length / 2) & (np.abs(phi2) < width)
    covered = np.zeros(npix, dtype=bool)
    centre = hp.ang2vec(placement["center_ra"], placement["center_dec"], lonlat=True)
    covered[hp.query_disc(nside, centre, np.radians(15.0))] = True
    prediction = np.where(covered, 0.0, np.nan)
    prediction[band] = 0.9
    control = np.where(covered, 0.0, np.nan)
    return prediction, control, covered, placement, width, length, band


def test_track_band_statistics_measures_the_band_and_its_sides():
    prediction, control, covered, placement, width, length, band = _band_maps()
    thresholds = np.array([0.5, 0.95])
    stats = track_band_statistics(
        prediction,
        control,
        covered,
        placement,
        width,
        length,
        thresholds,
        np.random.default_rng(0),
        n_null_bands=50,
    )
    assert stats["band_pixels"] == band.sum()
    assert list(stats["n_above_band"]) == [band.sum(), 0]
    assert list(stats["n_above_side"]) == [0, 0]
    assert stats["side_pixels"] > 0
    assert stats["n_null_bands"] == 50
    # An empty background: the flagged band beats every background band.
    assert stats["band_p_value"][0] == pytest.approx(1 / 51)
    assert list(stats["null_density_median"]) == [0.0, 0.0]
    # At 0.95 nothing is flagged anywhere: no evidence at all.
    assert stats["band_p_value"][1] == pytest.approx(1.0)


def test_track_band_statistics_background_as_dense_as_the_stream_gives_no_significance():
    prediction, control, covered, placement, width, length, _ = _band_maps()
    control = np.where(covered, 0.9, np.nan)  # the background is flagged everywhere
    stats = track_band_statistics(
        prediction,
        control,
        covered,
        placement,
        width,
        length,
        np.array([0.5]),
        np.random.default_rng(0),
        n_null_bands=30,
    )
    assert stats["null_density_median"][0] == pytest.approx(1.0)
    assert stats["band_p_value"][0] == pytest.approx(1.0)


def _detection_rows():
    """Four realizations at one SB, thresholds [0.1, 0.5], bands of 200 px."""
    import pandas as pd

    def row(flagged, side, background=(0.0, 0.0), scatter=(0.0, 0.0)):
        return {
            "richness": 33.0,
            "n_above_band": np.array(flagged),
            "n_above_side": np.array(side),
            "band_p_value": np.array([0.01, 0.01]),
            "null_density_mean": np.array(background),
            "null_density_std": np.array(scatter),
            "null_density_median": np.array(background),
            "band_pixels": 200,
            "side_pixels": 200,
        }

    return pd.DataFrame(
        [
            # clean background: SNR = density / (1 pixel / 200) = 60, then 40
            row([60, 40], [10, 4]),
            # SNR 25 then 12, but only 12 pixels at 0.5 (< 20)
            row([25, 12], [5, 2]),
            # dense, but so is the background: SNR (0.4 - 0.3) / 0.1 = 1, then 0.5
            row([80, 70], [70, 60], (0.3, 0.3), (0.1, 0.1)),
            # no tile, so no band
            {**row([0, 0], [0, 0]), "n_above_band": None},
        ]
    )


def _budget_rows(model, flagged_by_threshold, control_by_threshold):
    """Two realizations of one model at one SB, thresholds [0.1, 0.5, 0.9].

    Each realization's stream-free control covers 1000 pixels, so the pooled
    background density at a threshold is (sum of control counts) / 2000.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "model": model,
                "richness": 33.0,
                "n_above_band": np.array(flagged),
                "n_above_no_stream": np.array(control),
                "fp_no_stream": 0,
                "tn_no_stream": 1000,
            }
            for flagged, control in zip(
                flagged_by_threshold, control_by_threshold, strict=True
            )
        ]
    )


def test_detection_at_false_alarm_rate_picks_each_models_own_threshold():
    import pandas as pd

    thresholds = np.array([0.1, 0.5, 0.9])
    # A cautious model: clean already at 0.1, so it is read at 0.1 and both
    # streams (30 and 25 flagged pixels) pass.
    cautious = _budget_rows("cautious", [[30, 10, 0], [25, 8, 0]], [[1, 0, 0]] * 2)
    # A loose model: density 20/2000 = 1e-2 at 0.1 and 2/2000 = 1e-3 at 0.5,
    # so a 1e-3 budget reads it at 0.5, where only one stream keeps 20 pixels.
    loose = _budget_rows("loose", [[90, 40, 5], [60, 15, 1]], [[10, 1, 0]] * 2)
    d = detection_at_false_alarm_rate(
        pd.concat([cautious, loose]), thresholds, targets=(1e-3,), group_by=["model"]
    ).set_index("model")

    assert d.loc["cautious", "threshold"] == 0.1
    assert d.loc["cautious", "detection_fraction"] == 1.0
    assert d.loc["loose", "threshold"] == 0.5
    assert d.loc["loose", "background_density"] == pytest.approx(1e-3)
    assert d.loc["loose", "detection_fraction"] == 0.5
    # At a fixed 0.5 the loose model would look as good as the cautious one
    # read at 0.1 does here; the budget is what separates them.


def test_detection_at_false_alarm_rate_leaves_out_an_unreachable_budget():
    thresholds = np.array([0.1, 0.5, 0.9])
    # Never below 5e-3 even at 0.9: a 1e-3 budget cannot be met on this grid.
    stubborn = _budget_rows("stubborn", [[50, 40, 30]] * 2, [[20, 12, 5]] * 2)
    d = detection_at_false_alarm_rate(
        stubborn, thresholds, targets=(1e-3, 1e-2), group_by=["model"]
    )
    assert list(d["target"]) == [1e-2]


def test_detection_at_false_alarm_rate_needs_the_control():
    import pandas as pd

    with pytest.raises(ValueError, match="n_above_no_stream"):
        detection_at_false_alarm_rate(
            pd.DataFrame({"richness": [33.0], "n_above_band": [np.array([1])]}),
            np.array([0.5]),
        )


def test_band_snr_normalizes_by_area_and_floors_the_noise():
    # The same density (0.1) in bands of different sizes gives the same SNR.
    snr = band_snr(np.array([20, 40]), np.array([200, 400]), 0.02, 0.01)
    assert snr[0] == pytest.approx((0.1 - 0.02) / 0.01)
    assert snr[1] == pytest.approx(snr[0])
    # A clean background: the noise is one pixel, not zero.
    assert band_snr(20, 200, 0.0, 0.0) == pytest.approx(0.1 / (1 / 200))
    # The Poisson noise of the expected background count wins over a smaller
    # measured scatter.
    assert band_snr(20, 200, 0.05, 0.001) == pytest.approx(
        (0.1 - 0.05) / (np.sqrt(0.05 * 200) / 200)
    )
    assert np.isnan(band_snr(0, 0, 0.0, 0.0))


def test_stream_detection_needs_enough_pixels_and_a_high_snr():
    d = stream_detection(_detection_rows(), np.array([0.1, 0.5]), at=(0.1, 0.5))
    d = d.set_index("threshold")

    assert d.loc[0.1, "n_detected"] == 2  # 60 and 25 pixels, high SNR
    assert d.loc[0.5, "n_detected"] == 1  # 12 pixels is below 20
    assert d.loc[0.5, "n_realizations"] == 4
    assert d.loc[0.5, "n_without_band"] == 1
    assert d.loc[0.5, "detection_fraction"] == pytest.approx(0.25)
    assert d.loc[0.5, "detection_low"] <= 0.25 <= d.loc[0.5, "detection_high"]


def test_stream_detection_reports_counts_densities_and_contrasts():
    d = stream_detection(_detection_rows(), np.array([0.1, 0.5]), at=(0.5,)).iloc[0]
    assert d.flagged_in_band == pytest.approx((40 + 12 + 70) / 3)
    assert d.flagged_in_side == pytest.approx((4 + 2 + 60) / 3)
    assert d.band_density == pytest.approx(122 / 600)
    assert d.side_density == pytest.approx(66 / 600)
    assert d.local_contrast == pytest.approx(122 / 66)
    assert d.background_density == pytest.approx(0.3 / 3)
    assert d.background_contrast == pytest.approx((122 / 600) / 0.1)
    assert d.median_snr == pytest.approx(12.0)


def test_stream_detection_needs_the_band_statistics():
    with pytest.raises(ValueError, match="band statistics"):
        stream_detection(
            _detection_rows().drop(columns=["n_above_band"]), np.array([0.5])
        )


def test_plot_stream_detection_renders_intervals():
    d = stream_detection(_detection_rows(), np.array([0.1, 0.5]), at=(0.1, 0.5))
    ax = plot_stream_detection(d, highlight=0.5, training_range=(31, 34))
    assert ax.get_legend_handles_labels()[1] == ["threshold 0.1", "threshold 0.5"]
    assert ax.get_ylabel() == "fraction of streams detected"


def test_stream_detection_end_to_end(injector):
    import torch

    torch.manual_seed(0)
    model = UNet(in_channels=2, out_channels=2, base_width=4, depth=1, head="sigmoid")
    results = evaluate_footprint_realizations(
        model,
        injector,
        _identity_transform,
        _param_sets(),
        n_realizations=2,
        channel=0,
        thresholds=THRESHOLD_GRID,
        n_null_bands=20,
    )
    for _, row in results.iterrows():
        assert len(row["n_above_band"]) == len(THRESHOLD_GRID)
        # nside 64 pixels (~0.9 deg) are wider than the 0.2 deg stream, so the
        # 1-2 sigma side bands may hold no pixel centre; the band must not.
        assert row["band_pixels"] > 0 and row["side_pixels"] >= 0
        assert 0 < row["n_null_bands"] <= 20
        assert np.all((row["band_p_value"] > 0) & (row["band_p_value"] <= 1))
    d = stream_detection(results, THRESHOLD_GRID, at=(0.5,))
    assert list(d["richness"]) == [3000, 6000]
    assert (d["n_realizations"] == 2).all()


# ---------------------------------------------------------------------------
# Real streams on real skies
# ---------------------------------------------------------------------------


def _curved_track(n=400):
    """A gently curving 20-degree track around RA 30, Dec -40."""
    t = np.linspace(-10, 10, n)
    return 30.0 + t, -40.0 + 0.02 * t**2


def test_false_alarm_rates_are_per_group_and_count_ties_against():
    from streamgoggles.evaluation.footprint import false_alarm_map

    prediction = np.array([0.1, 0.5, 0.9, 1.0, 0.2, 0.4, 0.4, 0.8])
    calibration = np.ones(8, dtype=bool)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rate = false_alarm_map(prediction, calibration, groups)
    # Group 0: 0.9 is matched or beaten by 2 of its 4 calibration pixels.
    assert rate[2] == pytest.approx(2 / 4)
    # Group 1 is ranked on its own: 0.8 is its highest, 1 of 4.
    assert rate[7] == pytest.approx(1 / 4)
    # A tie counts against: the two 0.4s are each at "3 of 4 at least as high".
    assert rate[5] == rate[6] == pytest.approx(3 / 4)


def test_false_alarm_rate_is_nan_where_there_is_no_prediction():
    from streamgoggles.evaluation.footprint import false_alarm_map

    prediction = np.array([0.3, np.nan, 0.7])
    rate = false_alarm_map(prediction, np.array([True, False, True]))
    assert np.isnan(rate[1]) and np.isfinite(rate[[0, 2]]).all()


def test_track_band_follows_a_curved_track():
    from streamgoggles.evaluation.footprint import track_band

    nside = 128
    ra, dec = _curved_track()
    band = track_band([(ra, dec)], 0.5, nside)
    # Every track point's pixel is in the band, and its area is about
    # 2 * width * length.
    assert band[hp.ang2pix(nside, ra, dec, lonlat=True)].all()
    area = band.sum() * hp.nside2pixarea(nside, degrees=True)
    assert 16 < area < 28


def test_moving_a_band_keeps_its_shape():
    from streamgoggles.evaluation.footprint import _moved

    ra, dec = _curved_track()
    vectors = np.array(hp.ang2vec(ra, dec, lonlat=True))
    target = np.array(hp.ang2vec(300.0, -20.0, lonlat=True))
    moved = _moved(vectors, target, roll=1.0)
    # Rigid: every pairwise angle is preserved, and the centroid lands there.
    np.testing.assert_allclose(moved @ moved.T, vectors @ vectors.T, atol=1e-9)
    centre = moved.mean(axis=0)
    assert np.dot(centre / np.linalg.norm(centre), target) > 0.9999


def test_a_line_of_flagged_pixels_is_detected_and_noise_is_not():
    """The criterion's two outcomes on one synthetic sky: a stream of flagged
    pixels along a curved track stands out from stream-shaped bands of the
    same sky, and the same track on noise alone does not."""
    from streamgoggles.evaluation.footprint import real_track_statistics, track_band

    nside = 128
    rng = np.random.default_rng(0)
    npix = hp.nside2npix(nside)
    _, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    sky = (dec < -10) & (dec > -70)
    band = track_band([_curved_track()], 0.5, nside)
    calibration = sky & ~track_band([_curved_track()], 3.0, nside)
    noise = sky & (rng.random(npix) < 0.01)

    with_stream = noise | (band & (rng.random(npix) < 0.5))
    found = real_track_statistics(with_stream, sky, band, calibration, rng, 100)
    assert found["n_flagged"] >= 20 and found["snr"] >= 2 and found["p_value"] < 0.02

    alone = real_track_statistics(noise, sky, band, calibration, rng, 100)
    assert alone["snr"] < 2
    assert alone["null_density_mean"] == pytest.approx(0.01, rel=0.3)
