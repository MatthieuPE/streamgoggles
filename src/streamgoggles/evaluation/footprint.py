"""Footprint-level detection evaluation: predict on the sky, score on the sky.

`completeness_purity.evaluate_on_grid` scores the model one window at a
time. That answers "how good is the model on a window it was given", but a
stream search does not hand the model a window around the stream -- it
tiles the survey map, predicts every tile, and reads the result back as a
HEALPix map. This module evaluates that path:

1. `predict_footprint` -- tile a full-sky realization, run the model on
   every tile, stitch input/label/prediction back onto HEALPix.
2. `score_footprint` -- per-pixel confusion counts and the four per-class
   rates over the covered sky.
3. `evaluate_footprint_realizations` -- repeat over many independent
   realizations of each parameter set, one row per realization.
4. `plot_confusion_matrix` / `plot_detection_rates` -- the two views asked
   for: the averaged confusion matrix, and the detection rates as a
   function of a parameter (surface brightness now; distance modulus as a
   second axis later -- the rows already carry every parameter, so grouping
   by two of them needs no change here).

Everything is per-pixel on HEALPix, so a map containing several streams is
scored exactly like a map containing one: the label is the union, and each
pixel is simply stream or not.
"""

import logging
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import pandas as pd

from streamgoggles.evaluation.metrics import confusion_matrix, confusion_rates
from streamgoggles.matched_filter import (
    crop_window,
    finalize_full,
    stitch_windows_to_healpix,
)
from streamgoggles.windows import tile_footprint

if TYPE_CHECKING:
    from streamgoggles.injector import StreamInjector
    from streamgoggles.matched_filter import PixelizationSpec
    from streamgoggles.windows import Window

logger = logging.getLogger(__name__)

_RATE_NAMES = ("tpr", "fnr", "fpr", "tnr")

#: Probability thresholds for a detection curve (found fraction against
#: false-alarm rate as the threshold is swept). Evenly spaced in logit space,
#: so the tail that matters for rare false alarms -- probabilities within
#: ~1e-5 of 1 -- is resolved as finely as the middle. Contains 0.5 exactly,
#: so the curve passes through the default operating point.
THRESHOLD_GRID = 1.0 / (1.0 + np.exp(-np.linspace(-12.0, 12.0, 241)))
THRESHOLD_GRID[120] = 0.5


def _count_above(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Number of `values` strictly above each threshold (same `>` as scoring)."""
    ordered = np.sort(values)
    return len(ordered) - np.searchsorted(ordered, thresholds, side="right")


def tiles_around_stream(
    full_sky: dict,
    channel: int,
    pix: "PixelizationSpec",
    radius_deg: float | None = None,
    stride_fraction: float = 0.5,
) -> list["Window"]:
    """Overlapping tiles covering the neighbourhood of one injected stream.

    Centred on the stream's own stream-only pixels, so the evaluated region
    is the same size whatever the stream's richness. Note what that implies:
    the false-positive rate is measured near a stream rather than over the
    whole footprint. That is a sound controlled comparison across surface
    brightness, but it is not a survey-wide false-positive rate -- tile the
    full footprint (`windows.tile_footprint` on the survey mask) for that,
    which is also what multi-stream populations will need.

    Parameters:
        full_sky: output of `StreamInjector.inject_stream_full_sky`.
        channel: channel whose stream-only map locates the stream.
        pix: PixelizationSpec of the model's input windows.
        radius_deg: neighbourhood radius (degrees); defaults to one window.
        stride_fraction: tile spacing as a fraction of the window size.
            Must be < 1 for tiles to overlap, which is what gives
            `stitch_windows_to_healpix` a choice of window per pixel.

    Returns:
        List of overlapping Windows, possibly empty if the stream left no
        stream-only pixels at all.
    """
    nside = pix.nside
    window_size_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    radius_deg = window_size_deg if radius_deg is None else radius_deg

    stream_pixels = np.flatnonzero(full_sky["stream_raw_full"][channel] > 0)
    if stream_pixels.size == 0:
        return []

    centre = np.mean(np.array(hp.pix2vec(nside, stream_pixels)).T, axis=0)
    centre /= np.linalg.norm(centre)
    region = np.zeros(hp.nside2npix(nside), dtype=bool)
    region[hp.query_disc(nside, centre, np.radians(radius_deg))] = True
    region &= full_sky["valid_mask_full"]

    return tile_footprint(
        region,
        nside,
        tile_size_deg=window_size_deg,
        stride_deg=window_size_deg * stride_fraction,
    )


def predict_footprint(
    model,
    full_sky: dict,
    tiles: list["Window"],
    pix: "PixelizationSpec",
    transform,
    channel: int,
    count_threshold: float,
    device: str = "cpu",
) -> dict:
    """Run the model over tiles of one full-sky realization; stitch to HEALPix.

    Every tile goes through the same crop -> transform -> model path the
    training windows did, and all three outputs (input, label, prediction)
    are stitched by the identical rule, so they are directly comparable.

    The detection label is thresholded AFTER cropping, never before:
    cropping interpolates, so a thresholded {0, 1} map would come back
    fractional.

    Parameters:
        model: trained torch model (put in eval mode here).
        full_sky: output of `StreamInjector.inject_stream_full_sky`.
        tiles: windows to predict on (e.g. from `tiles_around_stream`).
        pix: PixelizationSpec of the model's input windows.
        transform: the same eval transform used for validation (it
            normalizes `map_stack`; augmentation must be off).
        channel: which output channel to keep (e.g. the "good" filter at the
            stream's distance).
        count_threshold: stream-only count above which a pixel is labelled
            a detection (the injector's own).
        device: torch device.

    Returns:
        dict with HEALPix arrays ``input``, ``label``, ``prediction`` (NaN
        where no data) and bool ``covered``.
    """
    import torch

    n_channels = len(full_sky["channels"])
    valid_full = full_sky["valid_mask_full"]
    tile_inputs, tile_labels, tile_preds = [], [], []

    model.eval()
    for tile in tiles:
        crops = [
            crop_window(full_sky["map_full"][c], valid_full, tile, pix)
            for c in range(n_channels)
        ]
        tile_valid = crops[0][1]
        map_stack = np.stack([image for image, _ in crops]).astype(np.float32)
        label_stack = np.stack(
            [
                (
                    crop_window(full_sky["stream_raw_full"][c], valid_full, tile, pix)[
                        0
                    ]
                    > count_threshold
                ).astype(np.float32)
                for c in range(n_channels)
            ]
        )
        prepared = transform(
            {
                "map_stack": map_stack,
                "label_stack": label_stack,
                "valid_mask": tile_valid,
                "params": full_sky["params"],
                "metadata": {"channels": full_sky["channels"]},
            }
        )
        with torch.no_grad():
            batch = torch.as_tensor(prepared["map_stack"]).unsqueeze(0).to(device)
            prediction = model(batch)[0, channel].cpu().numpy()

        # The network has no valid_mask awareness at inference time: what it
        # emits outside the footprint is meaningless, so it never reaches
        # the stitched map.
        tile_inputs.append(np.where(tile_valid, map_stack[channel], np.nan))
        tile_labels.append(np.where(tile_valid, label_stack[channel], np.nan))
        tile_preds.append(np.where(tile_valid, prediction, np.nan))

    nside = pix.nside
    input_hp, covered = stitch_windows_to_healpix(tile_inputs, tiles, pix, nside)
    label_hp, _ = stitch_windows_to_healpix(tile_labels, tiles, pix, nside)
    pred_hp, _ = stitch_windows_to_healpix(tile_preds, tiles, pix, nside)
    return {
        "input": input_hp,
        "label": label_hp,
        "prediction": pred_hp,
        "covered": covered,
    }


def score_footprint(
    prediction: np.ndarray,
    label: np.ndarray,
    covered: np.ndarray,
    threshold: float = 0.5,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Confusion counts and per-class rates over the covered, finite sky.

    Only pixels that are covered AND finite in both maps are scored. That
    restriction is load-bearing: stitched maps hold NaN wherever no window
    reached or the footprint has a hole, and `NaN > threshold` is False, so
    left in they would silently be scored as correctly rejected background.

    Parameters:
        prediction, label: HEALPix maps (e.g. from `predict_footprint`).
        covered: bool HEALPix mask of pixels any window reached.
        threshold: probability above which a pixel counts as detected.
        thresholds: optional array of thresholds (e.g. `THRESHOLD_GRID`).
            If given, also returns ``n_above_stream`` and
            ``n_above_background``: for each threshold, how many true stream
            pixels and how many non-stream pixels have a prediction above it.
            Divided by ``n_true_pixels`` and ``fp + tn``, these are the found
            fraction and false-alarm rate at every threshold, so models can
            be compared at the same false-alarm rate instead of at 0.5. The
            label is binarized at `threshold` as usual.

    Returns:
        dict with ``tp``/``fp``/``tn``/``fn`` counts, the four rates
        (``tpr``/``fnr``/``fpr``/``tnr``, NaN for an empty class),
        ``precision`` (``tp / (tp + fp)``: the fraction of flagged pixels
        that are really stream, NaN if nothing is flagged), and
        ``n_true_pixels`` (``tp + fn``) so an undefined ``tpr`` can be read
        as "nothing to detect" rather than "nothing detected".
    """
    scored = covered & np.isfinite(prediction) & np.isfinite(label)
    counts = confusion_matrix(prediction, label, scored, threshold)
    flagged = counts["tp"] + counts["fp"]
    scores = {
        **counts,
        **confusion_rates(counts),
        "precision": counts["tp"] / flagged if flagged else float("nan"),
        "n_true_pixels": counts["tp"] + counts["fn"],
    }
    if thresholds is not None:
        is_stream = label > threshold
        scores["n_above_stream"] = _count_above(
            prediction[scored & is_stream], thresholds
        )
        scores["n_above_background"] = _count_above(
            prediction[scored & ~is_stream], thresholds
        )
    return scores


def background_only_sky(injector: "StreamInjector", full_sky: dict) -> dict:
    """The same sky as `full_sky`, with the stream taken out.

    Every channel is rebuilt from the injector's background maps alone, and
    the stream-only maps are zero, so the label is empty everywhere.
    Predicting on this with the tiles of `full_sky` measures the no-stream
    false-positive rate on exactly the same patch of sky: any pixel flagged
    here was flagged by the background alone, and the difference from the
    stream sky is what the stream itself added.

    Parameters:
        injector: the StreamInjector that produced `full_sky`.
        full_sky: output of `StreamInjector.inject_stream_full_sky`.

    Returns:
        dict with the same keys as `full_sky`.
    """
    background = injector.background
    maps = [
        finalize_full(
            background.raw_map_full_dict[ch["filter"]][ch["distance_modulus"]],
            background.valid_mask_full,
            injector.finalize_cfg,
        )
        for ch in full_sky["channels"]
    ]
    return {
        **full_sky,
        "map_full": maps,
        "stream_raw_full": [np.zeros_like(m) for m in full_sky["stream_raw_full"]],
    }


def evaluate_footprint_realizations(
    model,
    injector: "StreamInjector",
    transform,
    param_sets: list[dict],
    n_realizations: int,
    channel: int,
    seed: int = 0,
    stride_fraction: float = 0.5,
    radius_deg: float | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
    no_stream_control: bool = True,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score the model on many independent realizations of each parameter set.

    For every parameter set, draws `n_realizations` fresh full-sky
    injections -- different stream population, placement, orientation and
    survey noise each time -- tiles each one, predicts, stitches, and scores.
    One realization of a stream says little (PLAN.md sections 6.14-6.17:
    single realizations produced both a confident "0.000" and a later
    "0.039" for the same configuration), so the unit of evidence here is the
    distribution over realizations.

    Seeds are derived deterministically from `seed`, the parameter-set
    index and the realization index, so a run is reproducible and adding
    realizations never changes the ones already drawn.

    Parameters:
        model: trained torch model.
        injector: StreamInjector (its `pix`/`count_threshold` are used).
        transform: eval transform matching training normalization.
        param_sets: stream parameter dicts, one per point to evaluate.
            Building these as a grid over two parameters (surface
            brightness x distance modulus) needs no change here.
        n_realizations: independent realizations per parameter set.
        channel: output channel to score.
        seed: base seed.
        stride_fraction, radius_deg: tiling, see `tiles_around_stream`.
        threshold: detection probability threshold.
        device: torch device.
        no_stream_control: also predict on `background_only_sky` with the
            same tiles, and record its false positives as ``fp_no_stream``/
            ``tn_no_stream``/``fpr_no_stream``. This is the reference a
            found fraction has to be read against: the fraction of
            background flagged when there is no stream at all.
        thresholds: optional threshold array (e.g. `THRESHOLD_GRID`). If
            given, each row also holds array-valued ``n_above_stream`` and
            ``n_above_background`` (see `score_footprint`), and, with the
            control, ``n_above_no_stream``: counts above each threshold on
            the no-stream sky, out of ``fp_no_stream + tn_no_stream`` pixels.

    Returns:
        DataFrame, one row per realization: every key of the parameter set,
        ``realization``, ``nstars`` (after richness resolution), ``n_tiles``,
        the confusion counts, the four rates, ``precision``,
        ``n_true_pixels``, the no-stream control columns if requested, and
        the per-threshold counts if requested (``None`` where a stream left
        no pixel to tile around).
    """
    control_keys = ("fp_no_stream", "tn_no_stream", "fpr_no_stream")
    rows = []
    for set_index, params in enumerate(param_sets):
        for realization in range(n_realizations):
            rng = np.random.default_rng([seed, set_index, realization])
            full_sky = injector.inject_stream_full_sky(params, rng)
            tiles = tiles_around_stream(
                full_sky, channel, injector.pix, radius_deg, stride_fraction
            )
            row = dict(params)
            row.update(
                realization=realization,
                nstars=full_sky["params"].get("nstars"),
                n_tiles=len(tiles),
            )
            if not tiles:
                # No stream-only pixel at all: nothing to tile around or
                # score. Recorded rather than skipped, so a vanished stream
                # stays visible in the results instead of disappearing.
                row.update(dict.fromkeys(("tp", "fp", "tn", "fn", "n_true_pixels"), 0))
                row.update(dict.fromkeys((*_RATE_NAMES, "precision"), float("nan")))
                if no_stream_control:
                    row.update(dict.fromkeys(control_keys, float("nan")))
                if thresholds is not None:
                    row.update(
                        dict.fromkeys(
                            (
                                "n_above_stream",
                                "n_above_background",
                                "n_above_no_stream",
                            )
                            if no_stream_control
                            else ("n_above_stream", "n_above_background"),
                            None,
                        )
                    )
                rows.append(row)
                continue

            maps = predict_footprint(
                model,
                full_sky,
                tiles,
                injector.pix,
                transform,
                channel,
                injector.count_threshold,
                device,
            )
            row.update(
                score_footprint(
                    maps["prediction"],
                    maps["label"],
                    maps["covered"],
                    threshold,
                    thresholds,
                )
            )
            if no_stream_control:
                control = predict_footprint(
                    model,
                    background_only_sky(injector, full_sky),
                    tiles,
                    injector.pix,
                    transform,
                    channel,
                    injector.count_threshold,
                    device,
                )
                scores = score_footprint(
                    control["prediction"],
                    control["label"],
                    control["covered"],
                    threshold,
                    thresholds,
                )
                row.update(
                    fp_no_stream=scores["fp"],
                    tn_no_stream=scores["tn"],
                    fpr_no_stream=scores["fpr"],
                )
                if thresholds is not None:
                    # The control has no stream: every pixel is background.
                    row["n_above_no_stream"] = scores["n_above_background"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_matrix(rates: dict, ax=None, title: str | None = None):
    """Plot the row-normalized 2x2 confusion matrix.

    Rows are the TRUE class, columns the PREDICTED class, and each row sums
    to one, so a cell reads as "of the pixels that truly are X, this fraction
    was called Y" -- comparable across maps whose stream and background
    pixel counts differ by orders of magnitude, which raw counts are not.

    Parameters:
        rates: dict with ``tpr``/``fnr``/``fpr``/``tnr`` (e.g. means over
            realizations).
        ax: matplotlib Axes (created if None).
        title: optional title.

    Returns:
        The Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.8))

    matrix = np.array([[rates["tpr"], rates["fnr"]], [rates["fpr"], rates["tnr"]]])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    cell_names = [["found", "missed"], ["false alarm", "correct reject"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            text = "n/a" if np.isnan(value) else f"{value:.3f}"
            ax.text(
                j,
                i,
                f"{text}\n{cell_names[i][j]}",
                ha="center",
                va="center",
                color="white" if (not np.isnan(value) and value > 0.5) else "black",
                fontsize=10,
            )
    ax.set_xticks([0, 1], ["stream", "no stream"])
    ax.set_yticks([0, 1], ["stream", "no stream"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    return ax


def plot_detection_rates(
    aggregated: pd.DataFrame,
    param: str,
    axes=None,
    training_range: tuple[float, float] | None = None,
):
    """Plot the found fraction and the background false-alarm rate against a parameter.

    Two panels sharing the x axis, because the two numbers only mean
    something together -- finding half the stream pixels is worthless if the
    background is flagged just as often -- but live on very different
    scales (a found fraction spans 0-1, a useful false-alarm rate is ~1%):

    - top: ``tpr``, the fraction of true stream pixels found. The missed
      fraction is its complement and is not drawn.
    - bottom: ``fpr``, the fraction of non-stream pixels flagged as stream,
      on the sky with the stream; and, if present, ``fpr_no_stream``, the
      same tiles with the stream removed. The gap between the two is what
      the stream itself adds; the no-stream line is the floor set by the
      background alone.

    Mean +/- std over realizations; std bars are clipped to [0, 1]. Where
    only some realizations kept a true pixel, the found point is annotated
    "k/n"; where none did, it is marked as "label vanished" rather than
    drawn as zero.

    Parameters:
        aggregated: output of
            `completeness_purity.aggregate_over_replicates(results,
            metrics=["tpr", "fpr", ...], group_by=[param])`. Include
            ``fpr_no_stream`` for the control line and ``n_true_pixels``
            for the k/n annotations.
        param: column to use as the x axis.
        axes: two matplotlib Axes, top and bottom (created if None).
        training_range: optional (min, max) of the parameter the model was
            trained on, shaded so extrapolation is visible at a glance.

    Returns:
        The two Axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(
            2, 1, figsize=(6.5, 6.0), sharex=True, height_ratios=(3, 2)
        )
    ax_found, ax_false = axes

    data = aggregated.sort_values(param)
    x = data[param].to_numpy(dtype=float)

    def _errorbar(ax, rate, **kwargs):
        mean = data[f"{rate}_mean"].to_numpy(dtype=float)
        std = np.nan_to_num(data[f"{rate}_std"].to_numpy(dtype=float))
        defined = np.isfinite(mean)
        # A fraction lives in [0, 1]; a symmetric std bar past either end
        # would draw an impossible value, so the bars are clipped there.
        lower = mean - np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0) - mean
        ax.errorbar(
            x[defined],
            mean[defined],
            yerr=[lower[defined], upper[defined]],
            capsize=4,
            **kwargs,
        )

    for ax in axes:
        if training_range is not None:
            ax.axvspan(*training_range, color="0.9", zorder=0, label="training range")

    _errorbar(ax_found, "tpr", marker="o", label="true stream pixels found")

    # Where only some realizations kept a true pixel, the rates average over
    # those alone: say how many, so a "0.0 found" over 11 of 30 skies is not
    # read as a verdict on all 30.
    if "tpr_n" in data and "n_true_pixels_n" in data:
        with_pixels = data["tpr_n"].to_numpy()
        total = data["n_true_pixels_n"].to_numpy()
        tpr_mean = data["tpr_mean"].to_numpy(dtype=float)
        for xi, yi, k, n in zip(x, tpr_mean, with_pixels, total, strict=True):
            if 0 < k < n:
                ax_found.annotate(
                    f"{k}/{n}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=8,
                    color="0.3",
                )

    vanished = ~np.isfinite(data["tpr_mean"].to_numpy(dtype=float))
    if vanished.any():
        ax_found.scatter(
            x[vanished],
            np.full(vanished.sum(), 0.5),
            marker="x",
            color="0.4",
            s=70,
            label="no true pixels (label vanished)",
            zorder=3,
        )
    ax_found.set_ylabel("fraction of true\nstream pixels found")
    ax_found.set_ylim(-0.05, 1.05)
    ax_found.legend(fontsize=8, loc="best")

    _errorbar(ax_false, "fpr", marker="s", color="C3", label="with the stream injected")
    if "fpr_no_stream_mean" in data:
        _errorbar(
            ax_false,
            "fpr_no_stream",
            marker="^",
            color="0.35",
            linestyle="--",
            label="same tiles, no stream",
        )
    ax_false.set_ylim(bottom=0.0)
    ax_false.set_ylabel("fraction of background\npixels flagged as stream")
    ax_false.set_xlabel(param)
    ax_false.legend(fontsize=8, loc="best")
    return axes
