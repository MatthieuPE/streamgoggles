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
from streamgoggles.matched_filter import crop_window, stitch_windows_to_healpix
from streamgoggles.windows import tile_footprint

if TYPE_CHECKING:
    from streamgoggles.injector import StreamInjector
    from streamgoggles.matched_filter import PixelizationSpec
    from streamgoggles.windows import Window

logger = logging.getLogger(__name__)

_RATE_NAMES = ("tpr", "fnr", "fpr", "tnr")


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

    Returns:
        dict with ``tp``/``fp``/``tn``/``fn`` counts, the four rates
        (``tpr``/``fnr``/``fpr``/``tnr``, NaN for an empty class), and
        ``n_true_pixels`` (``tp + fn``) so an undefined ``tpr`` can be read
        as "nothing to detect" rather than "nothing detected".
    """
    scored = covered & np.isfinite(prediction) & np.isfinite(label)
    counts = confusion_matrix(prediction, label, scored, threshold)
    return {
        **counts,
        **confusion_rates(counts),
        "n_true_pixels": counts["tp"] + counts["fn"],
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

    Returns:
        DataFrame, one row per realization: every key of the parameter set,
        ``realization``, ``nstars`` (after richness resolution), ``n_tiles``,
        the confusion counts and the four rates, and ``n_true_pixels``.
    """
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
                row.update(dict.fromkeys(_RATE_NAMES, float("nan")))
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
                    maps["prediction"], maps["label"], maps["covered"], threshold
                )
            )
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
    ax=None,
    training_range: tuple[float, float] | None = None,
):
    """Plot found/missed fractions of true stream pixels against a parameter.

    Shows mean +/- std over realizations of ``tpr`` (true stream pixels
    found) and ``fnr`` (true stream pixels missed). Points where no
    realization had any true pixel are marked separately rather than drawn
    as zero: past that point the *label* has vanished (nothing exceeds the
    detection threshold), which is a different statement from "the model
    stopped detecting", and the two must not read the same.

    Parameters:
        aggregated: output of
            `completeness_purity.aggregate_over_replicates(results,
            metrics=["tpr", "fnr", ...], group_by=[param])`.
        param: column to use as the x axis.
        ax: matplotlib Axes (created if None).
        training_range: optional (min, max) of the parameter the model was
            trained on, shaded so extrapolation is visible at a glance.

    Returns:
        The Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))

    data = aggregated.sort_values(param)
    x = data[param].to_numpy(dtype=float)

    if training_range is not None:
        ax.axvspan(*training_range, color="0.9", zorder=0, label="training range")

    for rate, label, marker in (
        ("tpr", "found (true positive rate)", "o"),
        ("fnr", "missed (false negative rate)", "s"),
    ):
        mean = data[f"{rate}_mean"].to_numpy(dtype=float)
        std = np.nan_to_num(data[f"{rate}_std"].to_numpy(dtype=float))
        defined = np.isfinite(mean)
        ax.errorbar(
            x[defined],
            mean[defined],
            yerr=std[defined],
            marker=marker,
            capsize=4,
            label=label,
        )

    vanished = ~np.isfinite(data["tpr_mean"].to_numpy(dtype=float))
    if vanished.any():
        ax.scatter(
            x[vanished],
            np.full(vanished.sum(), 0.5),
            marker="x",
            color="0.4",
            s=70,
            label="no true pixels (label vanished)",
            zorder=3,
        )

    ax.set_xlabel(param)
    ax.set_ylabel("fraction of true stream pixels")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    return ax
