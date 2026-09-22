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
5. `detection_metrics` / `plot_detection_metrics` -- completeness,
   contamination and contrast at any threshold, from the per-threshold
   counts: the area-independent view used to compare models
   (docs: "Experiments and tests").
6. `stream_detection` / `plot_stream_detection` -- per stream rather than
   per pixel: the fraction of injected streams showing a line of flagged
   pixels along their track, denser than stream-shaped bands of background
   (`track_band_statistics`).

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
    label_channel: int | None = None,
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
        label_channel: which channel of the sky's stack the input and label
            come from, when it differs from the model's output channel -- a
            model that answers for one queried distance outputs one map
            (``channel=0``) for the matched filter's channel at that distance
            (`datasets.transforms.QueryDistanceTransform`). Defaults to
            ``channel``.

    Returns:
        dict with HEALPix arrays ``input``, ``label``, ``prediction`` (NaN
        where no data) and bool ``covered``.
    """
    if label_channel is None:
        label_channel = channel
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
        tile_inputs.append(np.where(tile_valid, map_stack[label_channel], np.nan))
        tile_labels.append(np.where(tile_valid, label_stack[label_channel], np.nan))
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


def stream_frame_coordinates(
    vectors: np.ndarray, center_ra: float, center_dec: float, rotation_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Positions in a stream's own frame, (phi1 along the track, phi2 across).

    The frame is the one `injector.place_stream_in_footprint` uses: a great
    circle through (center_ra, center_dec), its phi1 axis leaving the centre
    at position angle ``rotation_deg`` (east of north), phi1 = phi2 = 0 at the
    centre.

    Parameters:
        vectors: (N, 3) unit vectors (e.g. ``hp.pix2vec`` of HEALPix pixels).
        center_ra, center_dec, rotation_deg: the placement, in degrees
            (``inject_stream_full_sky(...)["placement"]``).

    Returns:
        (phi1, phi2) in degrees, each of shape (N,).
    """
    ra, dec = np.radians(center_ra), np.radians(center_dec)
    origin = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    north = np.array(
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
    )
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    theta = np.radians(rotation_deg)
    along = np.cos(theta) * north + np.sin(theta) * east
    pole = np.cross(origin, along)
    phi2 = np.degrees(np.arcsin(np.clip(vectors @ pole, -1.0, 1.0)))
    phi1 = np.degrees(np.arctan2(vectors @ along, vectors @ origin))
    return phi1, phi2


def track_band_statistics(
    prediction: np.ndarray,
    control_prediction: np.ndarray,
    covered: np.ndarray,
    placement: dict,
    width: float,
    length: float,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    n_null_bands: int = 200,
    min_band_coverage: float = 0.9,
) -> dict:
    """Flagged pixels along a stream's track, against stream-shaped background bands.

    A stream is found in a map by a line of flagged pixels denser than the
    background's false alarms, not by any single pixel. For one realization:

    - **band**: pixels within 1 width-sigma of the injected track
      (``|phi2| < width``, ``|phi1| <= length / 2``) on the map with the
      stream;
    - **side bands**: pixels between 1 and 2 sigma (``width <= |phi2| <
      2 width``), just beside the track. They are not pure background: about
      27% of a Gaussian stream's stars fall there, and the prediction can be
      wider than the stream;
    - **background bands**: the same 1-sigma band shape placed at
      ``n_null_bands`` random positions and orientations on the map with the
      stream removed (``control_prediction``), keeping only placements with
      at least ``min_band_coverage`` of the band's area scored. Their flagged
      densities are the distribution the background alone produces in a
      region of the stream's shape, clustering of false alarms included.

    Densities (flagged pixels / band pixels) are compared rather than counts,
    since bands cut by the footprint have fewer pixels.

    Parameters:
        prediction, control_prediction: stitched HEALPix probability maps with
            and without the stream (NaN where not scored).
        covered: bool HEALPix mask of pixels any tile reached.
        placement: ``{"center_ra", "center_dec", "rotation_deg"}``.
        width: the stream's cross-track Gaussian sigma, degrees.
        length: the stream's length, degrees.
        thresholds: probability thresholds (e.g. `THRESHOLD_GRID`).
        rng: generator for the background-band placements.
        n_null_bands: number of background bands.
        min_band_coverage: minimum scored fraction of a background band.

    Returns:
        dict with, per threshold (arrays over ``thresholds``):
        ``n_above_band`` and ``n_above_side`` (flagged counts in the band and
        side bands), ``null_density_mean``, ``null_density_std``,
        ``null_density_median`` and ``null_density_high`` (mean, standard
        deviation, median and 97.7th percentile of the background-band
        densities) and
        ``band_p_value`` (fraction of background bands at least as dense as
        the stream band, with a +1 correction); and the scalars
        ``band_pixels``, ``side_pixels``, ``n_null_bands``.
    """
    nside = hp.npix2nside(len(prediction))
    scored = covered & np.isfinite(prediction)
    pixels = np.flatnonzero(scored)
    vectors = np.array(hp.pix2vec(nside, pixels)).T
    phi1, phi2 = stream_frame_coordinates(vectors, **placement)
    along = np.abs(phi1) <= length / 2
    in_band = along & (np.abs(phi2) < width)
    in_side = along & (np.abs(phi2) >= width) & (np.abs(phi2) < 2 * width)
    values = prediction[pixels]
    n_above_band = _count_above(values[in_band], thresholds)
    n_above_side = _count_above(values[in_side], thresholds)

    band_area = 2 * width * length
    expected_pixels = band_area / hp.nside2pixarea(nside, degrees=True)
    control_scored = covered & np.isfinite(control_prediction)
    control_pixels = np.flatnonzero(control_scored)
    control_vectors = np.array(hp.pix2vec(nside, control_pixels)).T
    control_values = control_prediction[control_pixels]
    reach = np.cos(np.radians(np.hypot(length / 2, width)))
    densities = []
    attempts = 0
    while len(densities) < n_null_bands and attempts < 50 * n_null_bands:
        attempts += 1
        centre = int(rng.integers(len(control_pixels)))
        near = np.flatnonzero(control_vectors @ control_vectors[centre] >= reach)
        centre_ra, centre_dec = hp.pix2ang(nside, control_pixels[centre], lonlat=True)
        p1, p2 = stream_frame_coordinates(
            control_vectors[near],
            float(centre_ra),
            float(centre_dec),
            float(rng.uniform(0.0, 360.0)),
        )
        members = near[(np.abs(p1) <= length / 2) & (np.abs(p2) < width)]
        if len(members) < min_band_coverage * expected_pixels:
            continue
        densities.append(
            _count_above(control_values[members], thresholds) / len(members)
        )
    null = np.array(densities) if densities else np.full((0, len(thresholds)), np.nan)

    band_density = n_above_band / max(int(in_band.sum()), 1)
    at_least_as_dense = (null >= band_density[None, :]).sum(axis=0)
    empty = np.full(len(thresholds), np.nan)
    return {
        "band_pixels": int(in_band.sum()),
        "side_pixels": int(in_side.sum()),
        "n_null_bands": len(null),
        "n_above_band": n_above_band,
        "n_above_side": n_above_side,
        "null_density_mean": null.mean(axis=0) if len(null) else empty,
        "null_density_std": null.std(axis=0) if len(null) else empty,
        "null_density_median": np.median(null, axis=0) if len(null) else empty,
        "null_density_high": np.percentile(null, 97.7, axis=0) if len(null) else empty,
        "band_p_value": (1 + at_least_as_dense) / (len(null) + 1),
    }


_BAND_KEYS = (
    "band_pixels",
    "side_pixels",
    "n_null_bands",
    "n_above_band",
    "n_above_side",
    "null_density_mean",
    "null_density_std",
    "null_density_median",
    "null_density_high",
    "band_p_value",
)


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
    n_null_bands: int = 200,
    label_channel: int | None = None,
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
            With both ``thresholds`` and the control, and a stream with a
            ``width`` and ``length`` (uniform morphology), each row also holds
            `track_band_statistics`: flagged pixels within 1 sigma of the
            track and between 1 and 2 sigma, and the densities of
            ``n_null_bands`` stream-shaped bands on the no-stream sky
            (used by `stream_detection`).
        n_null_bands: number of background bands per realization.
        label_channel: the sky channel the label comes from when the model's
            output ``channel`` is not a channel of the sky (see
            `predict_footprint`); also the channel tiles are placed around.

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
                full_sky,
                channel if label_channel is None else label_channel,
                injector.pix,
                radius_deg,
                stride_fraction,
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
                                *_BAND_KEYS,
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
                label_channel,
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
                    label_channel,
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
                    if "width" in params and "length" in params:
                        row.update(
                            track_band_statistics(
                                maps["prediction"],
                                control["prediction"],
                                maps["covered"],
                                full_sky["placement"],
                                float(params["width"]),
                                float(params["length"]),
                                thresholds,
                                np.random.default_rng(
                                    [seed, set_index, realization, 1]
                                ),
                                n_null_bands,
                            )
                        )
                    else:
                        row.update(dict.fromkeys(_BAND_KEYS, None))
            rows.append(row)
    return pd.DataFrame(rows)


def score_streams_on_sky(
    model,
    injector: "StreamInjector",
    transform,
    full_sky: dict,
    channel: int,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    n_null_bands: int = 200,
    stride_fraction: float = 0.5,
    radius_deg: float | None = None,
    device: str = "cpu",
) -> list[dict]:
    """Track-band statistics for each stream of a multi-stream sky.

    The multi-stream counterpart of one realization of
    `evaluate_footprint_realizations`, for skies built by
    `StreamInjector.inject_streams_full_sky`: the area around all the streams
    is tiled, predicted and stitched once, the same tiles are predicted once
    more on the sky with every stream removed, and each stream is then judged
    along its own track against stream-shaped bands on that stream-free sky.
    The result feeds `stream_detection` and `detection_at_false_alarm_rate`
    like any other realization.

    Parameters:
        model: trained torch model.
        injector: the StreamInjector that built ``full_sky``.
        transform: eval transform matching the model's normalization.
        full_sky: output of `inject_streams_full_sky`.
        thresholds: probability thresholds to count flagged pixels at.
        rng: draws the background bands' positions.
        n_null_bands: background bands per stream.
        channel, stride_fraction, radius_deg, device: as
            `evaluate_footprint_realizations`.

    Returns:
        One dict per stream, in the sky's order: ``stream`` (its index), the
        stream's ``nstars``, the stream-free control's ``n_above_no_stream``,
        ``fp_no_stream`` and ``tn_no_stream``, and `track_band_statistics`
        for its track. Empty if no stream left a pixel to tile around.
    """
    tiles = tiles_around_stream(
        full_sky, channel, injector.pix, radius_deg, stride_fraction
    )
    if not tiles:
        return []
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
        control["prediction"], control["label"], control["covered"], 0.5, thresholds
    )
    rows = []
    for index, (params, placement) in enumerate(
        zip(full_sky["params"], full_sky["placement"], strict=True)
    ):
        row = {
            "stream": index,
            "nstars": params.get("nstars"),
            "n_above_no_stream": scores["n_above_background"],
            "fp_no_stream": scores["fp"],
            "tn_no_stream": scores["tn"],
        }
        row.update(
            track_band_statistics(
                maps["prediction"],
                control["prediction"],
                maps["covered"],
                placement,
                float(params["width"]),
                float(params["length"]),
                thresholds,
                rng,
                n_null_bands,
            )
        )
        rows.append(row)
    return rows


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
    # those alone: say how many, so a "0.0 found" over 11 of 30 realizations is not
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


def detection_metrics(
    results: pd.DataFrame,
    thresholds: np.ndarray,
    group_by: list[str] | tuple[str, ...] = ("richness",),
    at: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Completeness, contamination and contrast at each threshold.

    With counts summed over all realizations of a group, at threshold t
    (a pixel is classified as stream when its probability is above t):

    - ``S_t``: true stream pixels; ``S_s``: those classified as stream;
    - ``B_t``: true background pixels; ``B_s``: those classified as stream;
    - ``completeness`` C = S_s / S_t: the fraction of the stream found;
    - ``contamination`` F = B_s / B_t: the fraction of background flagged,
      computed with at least one flagged pixel, so it is never zero and the
      contrast is never overstated;
    - ``contrast`` C / F: how many times more likely a stream pixel is to be
      flagged than a background pixel. NaN when no stream pixel is found
      (undefined, not zero).

    None of these depends on how much sky was scored, unlike the purity
    S_s / (S_s + B_s) = 1 / (1 + (B_t / S_t) / (C / F)).

    Parameters:
        results: output of `evaluate_footprint_realizations` called with
            ``thresholds`` (rows whose stream left no pixel to tile around
            carry no counts and are skipped).
        thresholds: the same threshold array passed there (e.g.
            `THRESHOLD_GRID`).
        group_by: columns defining a group (one row per group and threshold).
        at: optional thresholds to keep; each is matched to the nearest value
            of ``thresholds``. Default: every threshold.

    Returns:
        DataFrame with the group columns, ``threshold``, ``S_s``, ``S_t``,
        ``B_s``, ``B_t``, ``completeness``, ``contamination``, ``contrast``.
    """
    group_by = list(group_by)
    counted = results[results["n_above_stream"].notna()]
    indices = (
        range(len(thresholds))
        if at is None
        else sorted({int(np.argmin(np.abs(thresholds - t))) for t in at})
    )
    rows = []
    for keys, group in counted.groupby(group_by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        stream_above = np.sum(np.stack(list(group["n_above_stream"])), axis=0)
        background_above = np.sum(np.stack(list(group["n_above_background"])), axis=0)
        n_stream = int(group["n_true_pixels"].sum())
        n_background = int((group["fp"] + group["tn"]).sum())
        for k in indices:
            completeness = stream_above[k] / n_stream if n_stream else np.nan
            contamination = max(int(background_above[k]), 1) / n_background
            rows.append(
                {
                    **dict(zip(group_by, keys, strict=True)),
                    "threshold": float(thresholds[k]),
                    "S_s": int(stream_above[k]),
                    "S_t": n_stream,
                    "B_s": int(background_above[k]),
                    "B_t": n_background,
                    "completeness": completeness,
                    "contamination": contamination,
                    "contrast": completeness / contamination
                    if completeness > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_detection_metrics(
    metrics: pd.DataFrame,
    param: str = "richness",
    axes=None,
    min_stream_pixels: int = 20,
    highlight: float | None = 0.5,
    training_range: tuple[float, float] | None = None,
):
    """Plot completeness, contamination and contrast against a parameter.

    One panel per metric, one line per threshold. Points where fewer than
    ``min_stream_pixels`` stream pixels are found are drawn hollow: too few
    counts for the completeness or the contrast to be a measurement.

    Parameters:
        metrics: output of `detection_metrics`.
        param: column for the x axis.
        axes: three matplotlib Axes (created if None).
        min_stream_pixels: below this S_s, a point is drawn hollow.
        highlight: threshold drawn thicker (the reference one), or None.
        training_range: optional (min, max) of ``param`` used in training,
            shaded.

    Returns:
        The three Axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = (
        ("completeness", r"completeness $C = S_s/S_t$", False),
        ("contamination", r"contamination $F = B_s/B_t$", True),
        ("contrast", r"contrast $C/F$", True),
    )
    thresholds = sorted(metrics["threshold"].unique())
    cmap = plt.get_cmap("viridis")
    for j, threshold in enumerate(thresholds):
        data = metrics[metrics["threshold"] == threshold].sort_values(param)
        color = cmap(j / max(len(thresholds) - 1, 1))
        emphasis = highlight is not None and np.isclose(threshold, highlight)
        for ax, (column, _, _) in zip(axes, panels, strict=True):
            x = data[param].to_numpy(dtype=float)
            y = data[column].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                color=color,
                lw=2.6 if emphasis else 1.4,
                ms=6 if emphasis else 4,
                label=f"threshold {threshold:.3g}",
            )
            if column != "contamination":
                hollow = data["S_s"].to_numpy() < min_stream_pixels
                ax.plot(
                    x[hollow],
                    y[hollow],
                    ls="none",
                    marker="o",
                    ms=7 if emphasis else 5,
                    mfc="white",
                    mec=color,
                    zorder=5,
                )
    for ax, (_, label, log) in zip(axes, panels, strict=True):
        if training_range is not None:
            ax.axvspan(*training_range, color="0.93", zorder=0)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(param)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both" if log else "major")
    axes[0].legend(fontsize=8)
    return axes


def band_snr(
    flagged: np.ndarray,
    band_pixels: np.ndarray,
    background_mean: np.ndarray,
    background_std: np.ndarray,
) -> np.ndarray:
    """Signal-to-noise of flagged pixels along a track, normalized by area.

    ``SNR = (rho_band - <rho_bg>) / sigma_bg``, with ``rho_band = flagged /
    band_pixels`` the flagged density within 1 sigma of the track, and
    ``<rho_bg>``, ``sigma_bg`` the mean and scatter of the flagged density in
    background bands of the same shape (`track_band_statistics`). The scatter
    is floored at the Poisson noise of the expected background count, and at
    one pixel when the background is clean: ``sqrt(max(<rho_bg> N, 1)) / N``
    for a band of N pixels, so an empty background never gives an infinite
    SNR.

    All arguments broadcast; returns the SNR (NaN where ``band_pixels`` is 0).
    """
    flagged = np.asarray(flagged, dtype=float)
    n = np.asarray(band_pixels, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = flagged / n
        floor = np.sqrt(np.maximum(background_mean * n, 1.0)) / n
        noise = np.maximum(background_std, floor)
        snr = (density - background_mean) / noise
    return np.where(n > 0, snr, np.nan)


def stream_detection(
    results: pd.DataFrame,
    thresholds: np.ndarray,
    at: tuple[float, ...] = (0.5,),
    min_pixels: int = 20,
    min_snr: float = 2.0,
    group_by: list[str] | tuple[str, ...] = ("richness",),
) -> pd.DataFrame:
    """Fraction of injected streams detected, per group and threshold.

    A search finds a stream as a line of flagged pixels denser than the
    background's false alarms, so detection is judged along the track
    (`track_band_statistics`). One realization is **detected** at threshold t
    if, within 1 width-sigma of its track:

    - at least ``min_pixels`` pixels are flagged (a real search would not
      follow up a handful of pixels), and
    - the flagged density stands out from the background: `band_snr` >=
      ``min_snr``, comparing densities (flagged pixels per pixel of area) in
      the band and in stream-shaped bands placed on the same sky without the
      stream.

    Realizations without band statistics (no tile, or a stream without
    width/length) count as not detected, reported in ``n_without_band``.

    Parameters:
        results: output of `evaluate_footprint_realizations` with
            ``thresholds`` and ``no_stream_control=True``.
        thresholds: the threshold array passed there.
        at: thresholds to report (nearest grid values).
        min_pixels: minimum flagged pixels within 1 sigma of the track.
        min_snr: minimum `band_snr`.
        group_by: columns defining a group.

    Returns:
        DataFrame with the group columns, ``threshold``, ``n_realizations``,
        ``n_without_band``, ``n_detected``, ``detection_fraction`` and its
        Wilson 68% interval ``detection_low``/``detection_high``;
        ``median_snr``; mean counts ``flagged_in_band`` (within 1 sigma),
        ``flagged_in_side`` (1-2 sigma), ``band_pixels``, ``side_pixels``;
        pooled densities ``band_density``, ``side_density``, and the mean
        ``background_density`` of the background bands; ``local_contrast``
        (band / side density) and ``background_contrast`` (band / background
        density), NaN when the denominator is 0; and ``median_p_value``, the
        median fraction of background bands at least as dense as the stream
        band.

    Raises:
        ValueError if the band statistics are missing.
    """
    if "n_above_band" not in results:
        raise ValueError(
            "stream_detection needs track band statistics: run "
            "evaluate_footprint_realizations with thresholds and no_stream_control=True"
        )
    group_by = list(group_by)
    indices = sorted({int(np.argmin(np.abs(thresholds - t))) for t in at})
    rows = []
    for keys, group in results.groupby(group_by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        banded = group[group["n_above_band"].notna()]
        n = len(group)

        def column(name, k, frame=banded):
            return np.array([row[k] for row in frame[name]], dtype=float)

        for k in indices:
            flagged = column("n_above_band", k)
            side = column("n_above_side", k)
            background_mean = column("null_density_mean", k)
            band_pixels = banded["band_pixels"].to_numpy(dtype=float)
            side_pixels = banded["side_pixels"].to_numpy(dtype=float)
            snr = band_snr(
                flagged, band_pixels, background_mean, column("null_density_std", k)
            )
            detected = (flagged >= min_pixels) & (snr >= min_snr)
            n_detected = int(detected.sum())
            low, high = _wilson_interval(n_detected, n)
            band_density = (
                flagged.sum() / band_pixels.sum() if band_pixels.sum() else np.nan
            )
            side_density = (
                side.sum() / side_pixels.sum() if side_pixels.sum() else np.nan
            )
            background_density = (
                float(np.nanmean(background_mean)) if len(background_mean) else np.nan
            )
            rows.append(
                {
                    **dict(zip(group_by, keys, strict=True)),
                    "threshold": float(thresholds[k]),
                    "n_realizations": n,
                    "n_without_band": n - len(banded),
                    "n_detected": n_detected,
                    "detection_fraction": n_detected / n if n else np.nan,
                    "detection_low": low,
                    "detection_high": high,
                    "median_snr": float(np.nanmedian(snr)) if len(snr) else np.nan,
                    "flagged_in_band": float(flagged.mean())
                    if len(flagged)
                    else np.nan,
                    "flagged_in_side": float(side.mean()) if len(side) else np.nan,
                    "band_pixels": float(band_pixels.mean())
                    if len(band_pixels)
                    else np.nan,
                    "side_pixels": float(side_pixels.mean())
                    if len(side_pixels)
                    else np.nan,
                    "band_density": band_density,
                    "side_density": side_density,
                    "background_density": background_density,
                    "local_contrast": band_density / side_density
                    if side_density
                    else np.nan,
                    "background_contrast": band_density / background_density
                    if background_density
                    else np.nan,
                    "median_p_value": float(np.median(column("band_p_value", k)))
                    if len(banded)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def detection_at_false_alarm_rate(
    results: pd.DataFrame,
    thresholds: np.ndarray,
    targets: tuple[float, ...] = (1e-4, 1e-3),
    group_by: list[str] | tuple[str, ...] = ("richness",),
    min_pixels: int = 20,
) -> pd.DataFrame:
    """Fraction of streams detected when the threshold is set by a false-alarm budget.

    A fixed probability threshold puts different models at different
    false-alarm rates -- two models can flag stream-free sky at rates an order
    of magnitude apart at 0.5 -- so comparing their detections there mixes
    what each model can see with how cautious it happens to be. Here each
    group gets its own threshold: the lowest on ``thresholds`` at which the
    fraction of stream-free sky it flags (the no-stream control of
    `evaluate_footprint_realizations`, pooled over the group's realizations)
    is at most the target. Detection is then read at that threshold.

    One realization counts as detected if at least ``min_pixels`` pixels are
    flagged within 1 sigma of its track. The SNR condition of
    `stream_detection` is not applied: it compares the band with the
    background, and choosing the threshold by background density already
    fixes that comparison for every group. Realizations without band
    statistics count as not detected, as in `stream_detection`.

    Group by whatever defines one model: ``("richness",)`` for a single model,
    or add a seed or configuration column to give each trained model its own
    threshold before averaging them.

    Parameters:
        results: output of `evaluate_footprint_realizations` with
            ``thresholds`` and ``no_stream_control=True``.
        thresholds: the threshold array passed there, in increasing order.
        targets: false-alarm budgets, as fractions of stream-free pixels.
        group_by: columns defining a group.
        min_pixels: minimum flagged pixels within 1 sigma of the track.

    Returns:
        DataFrame with the group columns, ``target``, the chosen ``threshold``
        and the ``background_density`` it achieves, ``n_realizations``,
        ``n_detected``, ``detection_fraction`` and its Wilson 68% interval
        ``detection_low``/``detection_high``. A group that never gets down to
        a target on the grid has no row for it: report how many groups reach
        each target rather than averaging over a silently shrinking set.

    Raises:
        ValueError if the band statistics or the no-stream control are missing.
    """
    missing = {"n_above_band", "n_above_no_stream"} - set(results.columns)
    if missing:
        raise ValueError(
            f"detection_at_false_alarm_rate needs {sorted(missing)}: run "
            "evaluate_footprint_realizations with thresholds and no_stream_control=True"
        )
    group_by = list(group_by)
    rows = []
    for keys, group in results.groupby(group_by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        control = np.sum(np.stack(list(group["n_above_no_stream"])), axis=0)
        control_pixels = float((group["fp_no_stream"] + group["tn_no_stream"]).sum())
        density = control / control_pixels
        banded = group[group["n_above_band"].notna()]
        flagged = (
            np.stack(list(banded["n_above_band"]))
            if len(banded)
            else np.empty((0, len(thresholds)))
        )
        n = len(group)
        for target in targets:
            reachable = np.flatnonzero(density <= target)
            if not reachable.size:
                continue
            k = int(reachable[0])
            n_detected = int((flagged[:, k] >= min_pixels).sum())
            low, high = _wilson_interval(n_detected, n)
            rows.append(
                {
                    **dict(zip(group_by, keys, strict=True)),
                    "target": target,
                    "threshold": float(thresholds[k]),
                    "background_density": float(density[k]),
                    "n_realizations": n,
                    "n_detected": n_detected,
                    "detection_fraction": n_detected / n if n else np.nan,
                    "detection_low": low,
                    "detection_high": high,
                }
            )
    return pd.DataFrame(rows)


def _wilson_interval(
    successes: int, trials: int, z: float = 1.0
) -> tuple[float, float]:
    """Wilson score interval for a binomial fraction (z=1: ~68%)."""
    if trials == 0:
        return np.nan, np.nan
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(centre - half, 0.0), min(centre + half, 1.0)


def plot_stream_detection(
    detection: pd.DataFrame,
    param: str = "richness",
    ax=None,
    highlight: float | None = 0.5,
    training_range: tuple[float, float] | None = None,
):
    """Plot the fraction of streams detected against a parameter.

    One line per threshold, with Wilson 68% intervals (the number of
    realizations per point is small, so the interval matters).

    Parameters:
        detection: output of `stream_detection`.
        param: column for the x axis.
        ax: matplotlib Axes (created if None).
        highlight: threshold drawn thicker, or None.
        training_range: optional (min, max) of ``param`` used in training.

    Returns:
        The Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    thresholds = sorted(detection["threshold"].unique())
    cmap = plt.get_cmap("viridis")
    for j, threshold in enumerate(thresholds):
        data = detection[detection["threshold"] == threshold].sort_values(param)
        emphasis = highlight is not None and np.isclose(threshold, highlight)
        y = data["detection_fraction"].to_numpy(dtype=float)
        ax.errorbar(
            data[param].to_numpy(dtype=float),
            y,
            yerr=[
                np.clip(y - data["detection_low"].to_numpy(dtype=float), 0, None),
                np.clip(data["detection_high"].to_numpy(dtype=float) - y, 0, None),
            ],
            marker="o",
            capsize=3,
            color=cmap(j / max(len(thresholds) - 1, 1)),
            lw=2.6 if emphasis else 1.4,
            label=f"threshold {threshold:.3g}",
        )
    if training_range is not None:
        ax.axvspan(*training_range, color="0.93", zorder=0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(param)
    ax.set_ylabel("fraction of streams detected")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
