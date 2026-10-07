"""Train on the real DES Y6 background; map the network's output over DES.

Training uses the real sky with every known stream, globular cluster and dwarf
masked (`des_yr6_background`), so an injected stream is the only stream a
window contains. Everything else is the configuration the experiments
adopted: the query-distance U-Net (depth 2, width 12, sigmoid head) reading
three matched-filter maps around the queried distance, the fixed decoy box and
the three distances; batch Dice, batch 8, background fraction 0.05; training
streams drawn over the DES ranges with age and metallicity drawn too; the
matched filter at 13 Gyr, Z = 0.0002 with the DES Y6 error model at one sigma
(docs/source/experiments/matched_filter_errors.md); six quick (4800-window)
trainings averaged into one map.

**Two folds.** There is one real sky. A model predicting on pixels it trained
on has been taught their background, which flatters its false-alarm rate --
and that rate is what sets a detection threshold. So the footprint is split
into 20-degree stripes of right ascension (`objects_overlap.spatial_fold`),
six models train on each fold, and every pixel of the final map is predicted
by the six that never trained on it. The known streams are masked in both
folds' training, so finding them is never recall.

**Inference** runs on `des_yr6_inference`: the same stars, with the known
streams left in and only Sagittarius, the clusters and the dwarfs masked. For
each queried distance modulus (15 to 19 by 0.5) it writes one HEALPix map of
the ensemble's output, and a figure of it.

Run from the repository root:
  python scripts/experiments/real_des/run.py train --fold 0 [--seeds 42 ...]
  python scripts/experiments/real_des/run.py train --fold 1
  python scripts/experiments/real_des/run.py infer
  python scripts/experiments/real_des/run.py figures   # one map per distance, and a GIF
"""

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "real_des"
MODELS = OUT / "models"
MAPS = OUT / "maps"
FIGURES = OUT / "figures"
DATA = Path("~/Documents/data/DES_yr6").expanduser()
TRAINING_CATALOGUE = DATA / "des_yr6_background.parquet"
INFERENCE_CATALOGUE = DATA / "des_yr6_inference.parquet"

SURVEY, RELEASE = "des", "yr6"
NAMESPACE = f"{SURVEY}_{RELEASE}"
CLIPPING = {"g": {"min": 16.0, "max": 24.5}, "r": {"min": 16.0, "max": 24.5}}
# Two folds: each pixel is predicted by the ensemble of the other one.
N_FOLDS, STRIPE_DEG = 2, 20.0
SEEDS = [42, 43, 44, 45, 46, 47]
WINDOWS = 4800
TRAINING_SET = "population"
STRIDE_FRACTION = 0.5


def stream_parameters_module():
    """The stream-parameters experiment, whose training loop, grids and
    ensemble this reuses unchanged, so the two differ only in the sky."""
    path = REPO / "scripts" / "experiments" / "stream_parameters" / "run.py"
    spec = importlib.util.spec_from_file_location("sp_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filters_config():
    """The matched filter as calibrated for DES, and the decoy box."""
    from streamgoggles.matched_filter import DES_YR6_ERROR_MODEL

    return {
        "good": {
            "type": "isochrone",
            "reference_isochrone": {"age": 13.0, "z": 0.0002},
            "error_model": dict(DES_YR6_ERROR_MODEL),
            "error_multiplier": [1.0, 1.0],
        },
        "decoy": {"type": "box", "color_range": (1.2, 1.5), "mag_range": (18.0, 24.5)},
    }


def build_sky(catalogue, fold=None):
    """Background and injector on a real catalogue, one fold of it or all.

    The background maps are cached on disk, keyed by the catalogue, the fold
    and the filters, so each is built once.
    """
    import numpy as np

    from streamgoggles.background import Background
    from streamgoggles.background_sources import PreparedCatalogBackgroundSource
    from streamgoggles.injector import StreamInjector
    from streamgoggles.matched_filter import PixelizationSpec, build_matched_filters
    from streamgoggles.storage import BackgroundMapStore
    from streamgoggles.stream_sources import StreamObsSource

    sp = stream_parameters_module()
    pix = PixelizationSpec(nside=512, image_size_pix=(96, 96))
    config = filters_config()
    filters = build_matched_filters(config, namespace=NAMESPACE)
    source_cfg = {"path": str(catalogue)}
    if fold is not None:
        source_cfg["fold"] = {
            "index": int(fold),
            "n_folds": N_FOLDS,
            "stripe_deg": STRIPE_DEG,
            "nside": pix.nside,
        }
    background = Background.load_or_cache(
        source=PreparedCatalogBackgroundSource(),
        source_cfg=source_cfg,
        study_region=None,
        cuts=[],
        clipping=CLIPPING,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=sp.CHANNEL_DISTANCES,
        finalize_cfg={"enabled": False},
        store=BackgroundMapStore(OUT / "background_maps"),
        pix=pix,
        survey=SURVEY,
        release=RELEASE,
        filter_configs=config,
    )
    # Nothing downstream reads the catalogue, and at 30-odd million stars it
    # would be copied into every data-loading worker. The maps hold integer
    # counts, exact in float32; the interpolation that reads them still runs
    # in float64, so windows are unchanged at half the memory.
    background.catalog = None
    for per_distance in background.raw_map_full_dict.values():
        for dm, counts in per_distance.items():
            per_distance[dm] = counts.astype(np.float32)
    injector = StreamInjector(
        background=background,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=CLIPPING,
        pix=pix,
        survey=SURVEY,
        release=RELEASE,
        bands=("g", "r"),
        richness_kind="surface_brightness",
        finalize_cfg={"enabled": False},
        label_policy="stream_detection",
        count_threshold=1.0,
        min_stream_length_deg=3.0,
    )
    return background, injector, pix


def model_path(fold, seed):
    return MODELS / f"fold{fold}_w{WINDOWS}_seed{seed}"


def train(fold, seeds):
    """Six quick trainings on one fold's sky, saved as they finish."""
    import torch

    sp = stream_parameters_module()
    MODELS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    background, injector, _ = build_sky(TRAINING_CATALOGUE, fold)
    print(
        f"fold {fold}: {background.valid_mask_full.sum():,} valid pixels, "
        f"sky built in {time.time() - start:.0f}s",
        flush=True,
    )
    for seed in seeds:
        stem = model_path(fold, seed)
        if stem.with_suffix(".pt").exists():
            print(f"{stem.name}: already trained", flush=True)
            continue
        start = time.time()
        model, _, result = sp.train(seed, WINDOWS, background, injector, TRAINING_SET)
        torch.save(model.state_dict(), stem.with_suffix(".pt"))
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "seed": seed,
                    "windows": WINDOWS,
                    "training_set": TRAINING_SET,
                    "filters": filters_config(),
                    "catalogue": str(TRAINING_CATALOGUE),
                    "train_losses": result["train_losses"],
                    "val_losses": result["val_losses"],
                    "train_s": time.time() - start,
                },
                indent=2,
            )
        )
        print(f"{stem.name}: trained in {time.time() - start:.0f}s", flush=True)


def load_ensemble(fold, seeds):
    """The models trained on one fold, averaged into one predictor."""
    import torch

    from streamgoggles.datasets.transforms import QueryDistanceTransform
    from streamgoggles.models.unet import UNet

    sp = stream_parameters_module()
    models = []
    for seed in seeds:
        model = UNet(
            in_channels=QueryDistanceTransform.n_channels, out_channels=1, **sp.MODEL
        )
        model.load_state_dict(torch.load(model_path(fold, seed).with_suffix(".pt")))
        model.eval()
        models.append(model)
    return sp.Ensemble(models)


def infer(seeds):
    """One map per queried distance, each pixel from the fold that did not train on it."""
    import healpy as hp
    import numpy as np
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
        WindowNormalizer,
    )
    from streamgoggles.matched_filter import WindowProjection, stitch_windows_to_healpix
    from streamgoggles.objects_overlap import get_footprint, spatial_fold
    from streamgoggles.windows import tile_footprint

    sp = stream_parameters_module()
    MAPS.mkdir(parents=True, exist_ok=True)
    # One torch thread: two OpenMP runtimes end up loaded in this environment,
    # and torch's first multithreaded convolution then segfaults (see
    # configure_torch_threads). Training sets this; inference must too.
    configure_torch_threads(num_workers=0)
    background, _, pix = build_sky(INFERENCE_CATALOGUE)
    valid = background.valid_mask_full
    usable, _, _ = get_footprint("des_yr6_inference", nside=pix.nside)
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        usable & valid,
        pix.nside,
        tile_size_deg=window_deg,
        stride_deg=window_deg * STRIDE_FRACTION,
    )
    ensembles = {fold: load_ensemble(fold, seeds) for fold in range(N_FOLDS)}
    channels = [
        {"filter": name, "distance_modulus": dm}
        for dm in sp.CHANNEL_DISTANCES
        for name in filters_config()
    ]
    transforms = {
        query: QueryDistanceTransform(
            StreamMapTransform(normalizer=WindowNormalizer(), augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=query,
        )
        for query in sp.QUERY_GRID
    }
    print(
        f"{len(tiles)} tiles over {usable.sum() * hp.nside2pixarea(pix.nside, degrees=True):,.0f} deg^2",
        flush=True,
    )

    # One crop per tile, shared by every queried distance and both folds.
    predictions = {(q, f): [] for q in sp.QUERY_GRID for f in range(N_FOLDS)}
    start = time.time()
    for number, tile in enumerate(tiles, start=1):
        projection = WindowProjection.for_window(tile, pix, valid)
        valid_at = valid[projection.pixnums]
        map_stack = np.stack(
            [
                projection.image(
                    np.where(
                        valid_at,
                        background.raw_map_full_dict[c["filter"]][
                            c["distance_modulus"]
                        ][projection.pixnums],
                        0.0,
                    )
                )
                for c in channels
            ]
        ).astype(np.float32)
        sample = {
            "map_stack": map_stack,
            "label_stack": np.zeros_like(map_stack),
            "valid_mask": projection.valid,
            "params": {},
            "metadata": {"channels": channels},
        }
        for query, transform in transforms.items():
            batch = torch.as_tensor(transform(sample)["map_stack"]).unsqueeze(0)
            with torch.no_grad():
                for fold, ensemble in ensembles.items():
                    output = ensemble(batch)[0, 0].numpy()
                    predictions[(query, fold)].append(
                        np.where(projection.valid, output, np.nan)
                    )
        if number % 25 == 0:
            print(
                f"  {number}/{len(tiles)} tiles, {time.time() - start:.0f}s", flush=True
            )

    # Each pixel from the ensemble that never trained on its fold -- and, as
    # a diagnostic, from the one that did. Both are already computed on every
    # tile, so the second costs nothing, and comparing them on stream-free
    # sky measures what training on a sky does to a model's output there.
    ra, _ = hp.pix2ang(pix.nside, np.arange(hp.nside2npix(pix.nside)), lonlat=True)
    pixel_fold = spatial_fold(ra, STRIPE_DEG, N_FOLDS)
    stream_free, _, _ = get_footprint("des_yr6_background", nside=pix.nside)
    (MAPS / "in_fold").mkdir(exist_ok=True)
    summary = {}
    for query in sp.QUERY_GRID:
        stitched = {
            fold: stitch_windows_to_healpix(
                predictions[(query, fold)], tiles, pix, pix.nside
            )[0]
            for fold in range(N_FOLDS)
        }
        out_of_fold = np.full(hp.nside2npix(pix.nside), np.nan)
        in_fold = np.full(hp.nside2npix(pix.nside), np.nan)
        for fold in range(N_FOLDS):
            mine = pixel_fold == fold
            # The product: this fold's pixels from the ensemble trained on
            # the other one, which never saw them.
            out_of_fold[mine] = stitched[1 - fold][mine]
            # The diagnostic: from the ensemble that trained on them.
            in_fold[mine] = stitched[fold][mine]
        for image in (out_of_fold, in_fold):
            image[~(usable & valid)] = np.nan
        for image, path in (
            (out_of_fold, MAPS / f"prediction_dm{query:.1f}.fits"),
            (in_fold, MAPS / "in_fold" / f"prediction_dm{query:.1f}.fits"),
        ):
            hp.write_map(
                path,
                np.where(np.isfinite(image), image, hp.UNSEEN).astype(np.float32),
                dtype=np.float32,
                overwrite=True,
                coord="C",
                column_names=["PROBABILITY"],
            )
        entry = {"covered_pixels": int(np.isfinite(out_of_fold).sum())}
        for name, image in (("out_of_fold", out_of_fold), ("in_fold", in_fold)):
            quiet = image[stream_free & np.isfinite(image)]
            entry[name] = {
                "stream_free_pixels": int(quiet.size),
                "stream_free_median": float(np.median(quiet)),
                "stream_free_p99_9": float(np.percentile(quiet, 99.9)),
                "stream_free_above_0.5": float((quiet > 0.5).mean()),
            }
        summary[f"{query:.1f}"] = entry
        print(
            f"m-M {query:.1f}: stream-free sky above 0.5 -- out of fold "
            f"{entry['out_of_fold']['stream_free_above_0.5']:.2e}, in fold "
            f"{entry['in_fold']['stream_free_above_0.5']:.2e}",
            flush=True,
        )
    (MAPS / "summary.json").write_text(json.dumps(summary, indent=2))


# The calibration sky: where false-alarm rates and null bands are measured.
# The training mask removes every known stream, cluster and dwarf, but the
# first maps show real structure still in what it leaves, at radii measured
# from the maps' flagged-pixel profiles (docs: "What the stream-free sky still
# holds"). Each radius is where the excess reaches the far-field level.
CALIBRATION_DISCS = {
    # name: (ra, dec, radius_deg)
    "LMC periphery": (80.894, -69.756, 20.0),  # 52% at 12-15 deg, 3% by 18-21
    "SMC periphery": (13.187, -72.829, 12.0),  # 23% at 9-12 deg, 1% by 12-15
}
SAGITTARIUS_EXTENT_DEG = 9.0  # from its track; 8% at 8-9 deg, background by 9-10
DWARF_HALF_LIGHT_RADII = 12.0  # Sculptor's excess reaches 2 deg, Fornax's 2
CLUSTER_RADIUS_DEG = 2.0  # NGC 1904, NGC 1261: 72-86% within 1 deg, ~4% at 1-2
CALIBRATION_MASK = OUT / "calibration_mask_nside512.fits.gz"


def object_mask(nside=512, max_dwarf_mv=None, max_radius_deg=None):
    """The compact objects whose outskirts the training mask leaves: every
    dwarf galaxy near the footprint to DWARF_HALF_LIGHT_RADII half-light
    radii, and every globular cluster within CLUSTER_RADIUS_DEG of it to that
    radius. Part of the calibration mask; a line search masks it too, since
    every line through such an object is bright. With ``max_dwarf_mv``, only
    the dwarfs brighter than that absolute magnitude: the ultra-faint ones
    make no bursts of lines, and some lie on streams (Tucana III's own
    progenitor, Tucana II beside Indus). With ``max_radius_deg``, no dwarf is
    masked farther than that: Fornax's and Sculptor's stars stand out to
    1.5 degrees in the matched filter, where 12 half-light radii reach 4.0
    and 2.2 (docs: line_model/des2018_known)."""
    import healpy as hp
    import numpy as np

    from streamgoggles.objects_overlap import (
        get_dwarf,
        get_footprint,
        get_GC,
        mask_objects,
    )

    covered, _, _ = get_footprint("des_yr6", nside=nside)

    def near_footprint(catalogue, reach_deg):
        near = []
        for row in catalogue:
            vector = hp.ang2vec(float(row["ra"]), float(row["dec"]), lonlat=True)
            disc = hp.query_disc(nside, vector, np.radians(reach_deg))
            near.append(bool(covered[disc].any()))
        return np.array(near)

    dwarfs = get_dwarf()
    if max_dwarf_mv is not None:
        bright = np.asarray(dwarfs["M_V"].filled(np.nan), float) < max_dwarf_mv
        dwarfs = dwarfs[bright]
    if max_radius_deg is not None:
        dwarfs = dwarfs.copy()
        dwarfs["rhalf"] = np.minimum(
            np.asarray(dwarfs["rhalf"].filled(np.nan), float),
            60.0 * max_radius_deg / DWARF_HALF_LIGHT_RADII,
        )
    dwarf_mask, _ = mask_objects(
        dwarfs,
        near_footprint(dwarfs, 2.0),
        nside=nside,
        radius_factor=DWARF_HALF_LIGHT_RADII,
    )
    clusters = get_GC()
    cluster_mask, _ = mask_objects(
        clusters,
        near_footprint(clusters, CLUSTER_RADIUS_DEG),
        nside=nside,
        radius_factor=0.0,
        min_radius_deg=CLUSTER_RADIUS_DEG,
    )
    return dwarf_mask | cluster_mask


def calibration_mask(nside=512):
    """The stream-free sky with the unmasked structure the maps revealed removed.

    Built on the training mask, so every known stream, cluster and dwarf is
    already out; this removes, in addition, the Magellanic Clouds'
    peripheries, Sagittarius out to 9 degrees from its track, every dwarf to
    12 half-light radii, and every globular cluster whose centre lies within
    2 degrees of the footprint to 2 degrees -- including the bright ones whose
    centres sit in Gold's foreground holes, which the training mask never
    selected.
    """
    import healpy as hp
    import numpy as np

    from streamgoggles.objects_overlap import get_footprint, mask_streams

    usable, _, _ = get_footprint("des_yr6_background", nside=nside)
    removed = np.zeros_like(usable)
    for ra, dec, radius in CALIBRATION_DISCS.values():
        vector = hp.ang2vec(ra, dec, lonlat=True)
        removed[hp.query_disc(nside, vector, np.radians(radius))] = True
    sagittarius, _ = mask_streams(
        {"Sagittarius": SAGITTARIUS_EXTENT_DEG},
        nside=nside,
        wide_stream_deg=0.0,
        wide_stream_factor=1.0,
        tracks={},
    )
    removed |= sagittarius
    removed |= object_mask(nside)
    calibration = usable & ~removed
    hp.write_map(
        CALIBRATION_MASK,
        calibration.astype(np.uint8),
        dtype=np.uint8,
        overwrite=True,
        coord="C",
        column_names=["CALIBRATION"],
    )
    area = hp.nside2pixarea(nside, degrees=True)
    print(
        f"calibration sky: {calibration.sum() * area:,.0f} deg^2 of the "
        f"{usable.sum() * area:,.0f} the training mask leaves",
        flush=True,
    )
    return calibration


# The detection criterion of the simulated experiments, on real tracks.
TARGET_FALSE_ALARM_RATE = 1e-3
MIN_FLAGGED_PIXELS = 20
MIN_SNR = 2.0
N_NULL_BANDS = 200
DETECTION_SEED = 2026
DETECTIONS = OUT / "detections.csv"


def detection_tracks():
    """The track each stream's detection band follows: the paper's own.

    {stream: [(ra, dec)]}, the arcs of Shipp et al. (2018) Table 1
    (`objects_overlap.des2018_arc`). The masks use every galstreams reference
    where that is the cautious choice; a detection band must follow the stream
    where DES found it. The galstreams tracks used before put Aliqa Uma's band
    0.6-1.45 degrees off the stream and Molonglo's 2 degrees
    (notebooks/des2018_reproduction.ipynb).
    """
    from streamgoggles.objects_overlap import DES2018_STREAM_WIDTHS, des2018_arc

    return {name: [des2018_arc(name, n=400)] for name in DES2018_STREAM_WIDTHS}


def detect():
    """Each DES 2018 stream, judged along its DES track at every distance.

    Flagged pixels are those whose false-alarm rate -- against the calibration
    sky of their own fold -- is at most 1e-3. A stream is detected at a
    distance when at least 20 pixels within one width of its track are
    flagged and their density stands out from 200 copies of the band moved
    onto calibration sky by an S/N of at least 2: the criterion of the
    simulated experiments. It is reported at the queried distance nearest the
    stream's own, and at every other, since a detection that peaks at the
    stream's distance is one that belongs to the stream.
    """
    import healpy as hp
    import numpy as np
    import pandas as pd

    from streamgoggles.evaluation.footprint import (
        false_alarm_map,
        real_track_statistics,
        track_band,
    )
    from streamgoggles.objects_overlap import get_footprint, spatial_fold

    sp = stream_parameters_module()
    nside = 512
    calibration = (
        hp.read_map(CALIBRATION_MASK).astype(bool)
        if CALIBRATION_MASK.exists()
        else calibration_mask(nside)
    )
    inference, _, _ = get_footprint("des_yr6_inference", nside=nside)
    ra, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    folds = spatial_fold(ra, STRIPE_DEG, N_FOLDS)
    chosen = detection_tracks()
    bands = {
        name: track_band(chosen[name], width, nside) & inference
        for name, (width, _, _, _) in sp.DES_STREAMS.items()
    }
    rng = np.random.default_rng(DETECTION_SEED)
    rows = []
    for query in sp.QUERY_GRID:
        prediction = hp.read_map(MAPS / f"prediction_dm{query:.1f}.fits")
        prediction[prediction == hp.UNSEEN] = np.nan
        rate = false_alarm_map(prediction, calibration, folds)
        scored = np.isfinite(rate)
        flagged = scored & (rate <= TARGET_FALSE_ALARM_RATE)
        for name, (width, length, distance, sb) in sp.DES_STREAMS.items():
            stats = real_track_statistics(
                flagged, scored, bands[name], calibration, rng, N_NULL_BANDS
            )
            rows.append(
                {
                    "stream": name,
                    "distance_modulus": distance,
                    "width": width,
                    "surface_brightness": sb,
                    "query": query,
                    **stats,
                    "detected": bool(
                        stats["n_flagged"] >= MIN_FLAGGED_PIXELS
                        and stats["snr"] >= MIN_SNR
                    ),
                }
            )
        print(f"m-M {query:.1f}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(DETECTIONS, index=False)
    nearest = table.loc[
        table.groupby("stream")
        .apply(lambda g: (g["query"] - g["distance_modulus"]).abs().idxmin())
        .to_numpy()
    ].sort_values("snr", ascending=False)
    pd.set_option("display.width", 200)
    print(
        nearest[
            [
                "stream",
                "distance_modulus",
                "query",
                "band_pixels",
                "n_flagged",
                "null_density_mean",
                "snr",
                "p_value",
                "detected",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\ndetected at the nearest distance: {int(nearest.detected.sum())} of {len(nearest)}"
    )
    summary = input_significance(nearest)
    summary.to_csv(STREAM_SUMMARY, index=False)
    figure_detections(table, summary)
    return table


STREAM_SUMMARY = OUT / "stream_summary.csv"
DOC_FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "real_des"


def input_significance(nearest):
    """How visible each stream is in the matched-filter counts the model reads.

    The classic matched-filter test, on the same band: counts within one
    width of the track against the mean in side bands two to four widths
    away (off the stream's wings), at the queried distance nearest the
    stream's own. It separates a stream the network misses from one the
    input does not contain.
    """
    import numpy as np

    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.objects_overlap import get_footprint

    sp = stream_parameters_module()
    background, _, _ = build_sky(INFERENCE_CATALOGUE)
    inference, _, _ = get_footprint("des_yr6_inference", nside=512)
    valid = background.valid_mask_full & inference
    chosen = detection_tracks()
    rows = []
    for row in nearest.itertuples():
        width = sp.DES_STREAMS[row.stream][0]
        tracks = chosen[row.stream]
        band = track_band(tracks, width, 512) & valid
        side = (
            track_band(tracks, 4 * width, 512)
            & ~track_band(tracks, 2 * width, 512)
            & valid
        )
        counts = background.raw_map_full_dict["good"][row.query]
        expected = counts[side].mean() * band.sum()
        rows.append(
            {
                "stream": row.stream,
                "distance_modulus": row.distance_modulus,
                "query": row.query,
                "input_contrast": counts[band].mean() / counts[side].mean() - 1,
                "input_snr": (counts[band].sum() - expected) / np.sqrt(expected),
                "network_snr": row.snr,
                "n_flagged": row.n_flagged,
                "detected": row.detected,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows).sort_values("network_snr", ascending=False)


def figure_detections(table, summary):
    """S/N against queried distance per stream; and network against input."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    order = summary.stream.tolist()
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, order, strict=False):
        rows = table[table.stream == name].sort_values("query")
        found = bool(summary.set_index("stream").loc[name, "detected"])
        ax.plot(
            rows["query"],
            rows["snr"],
            marker="o",
            ms=4,
            lw=1.6,
            color="#2a78d6" if found else "#8a8986",
        )
        ax.axvline(rows["distance_modulus"].iloc[0], color="#eb6834", lw=1.2, ls="--")
        ax.axhline(MIN_SNR, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=2)
        ax.set_title(f"{name}{'' if not found else '  (detected)'}", fontsize=9.5)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(order) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("queried m−M")
    for ax in axes[:, 0]:
        ax.set_ylabel("S/N along the track")
    fig.suptitle(
        "Each DES 2018 stream along its DES track, at every queried distance — "
        "dashed: its catalogued distance; dotted: S/N = 2",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / "snr_by_distance.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(
        summary.input_snr,
        np.maximum(summary.network_snr, 0.3),
        c=summary.distance_modulus,
        cmap="Blues",
        vmin=15.0,
        vmax=19.0,
        s=70,
        edgecolors="#1a1a19",
        linewidths=0.6,
    )
    for row in summary.itertuples():
        on_floor = row.network_snr < 0.3
        # Missed streams all sit on the floor, a few tenths apart in x:
        # slanted labels keep them apart.
        ax.annotate(
            row.stream,
            (row.input_snr, max(row.network_snr, 0.3)),
            fontsize=8,
            xytext=(4, 6) if on_floor else (5, 3),
            textcoords="offset points",
            rotation=50 if on_floor else 0,
        )
    ax.set_yscale("log")
    ax.axhline(MIN_SNR, color="0.6", lw=0.8, ls=":")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("S/N of the stream in the matched-filter counts (input)")
    ax.set_ylabel("S/N of the network's flagged pixels (output; 0 drawn at 0.3)")
    fig.colorbar(points, ax=ax, label="catalogued m−M")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / "input_vs_network.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def figures():
    """One map per queried distance, and an animation through them.

    Each shows the out-of-fold ensemble output over the DES footprint on one
    shared scale, with the DES 2018 streams' tracks outlined for orientation
    -- along their DES tracks, the ones the training mask used.
    """
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import skyproj
    from PIL import Image

    from streamgoggles.objects_overlap import DES2018_STREAM_WIDTHS, stream_tracks

    sp = stream_parameters_module()
    FIGURES.mkdir(parents=True, exist_ok=True)
    tracks = {name: stream_tracks(name) for name in DES2018_STREAM_WIDTHS}
    frames = []
    for query in sp.QUERY_GRID:
        path = MAPS / f"prediction_dm{query:.1f}.fits"
        if not path.exists():
            continue
        values = hp.read_map(path)
        values = hp.ud_grade(values, nside_out=256)  # legible at page size
        kpc = 10 ** ((query + 5) / 5) / 1000
        _, ax = plt.subplots(figsize=(10, 6.6))
        sky = skyproj.DESSkyproj(ax=ax)
        sky.draw_hpxmap(values, zoom=False, cmap="Blues", vmin=0.0, vmax=1.0)
        sky.draw_colorbar(label="network output (out of fold)", fontsize=9)
        for by_reference in tracks.values():
            for ra, dec in by_reference.values():
                jumps = np.flatnonzero(np.abs(np.diff(ra)) > 180) + 1
                for piece_ra, piece_dec in zip(
                    np.split(ra, jumps), np.split(dec, jumps), strict=True
                ):
                    sky.ax.plot(
                        piece_ra, piece_dec, color="#eb6834", lw=0.8, alpha=0.55
                    )
        sky.ax.set_title(
            f"m−M = {query:.1f}  ({kpc:.0f} kpc) — DES 2018 tracks in orange",
            fontsize=11,
            pad=24,
        )
        out = FIGURES / f"prediction_dm{query:.1f}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        frames.append(out)
        print(f"{out.name}", flush=True)
    if frames:
        images = [Image.open(frame).convert("RGB") for frame in frames]
        size = images[0].size
        images = [image.resize(size) for image in images]
        gif = FIGURES / "prediction.gif"
        images[0].save(
            gif, save_all=True, append_images=images[1:], duration=900, loop=0
        )
        print(f"{gif.name}: {len(images)} frames", flush=True)


# Recovery efficiency: each DES 2018 stream, simulated with its own Table 1
# parameters and injected into the real inference sky many times, scored by
# exactly the test the real streams get. It separates "this stream is hard"
# from "this place is hard": the same stream at 20 random places, and at 4
# places parallel to its own track (3 and 5 degrees either side).
RECOVERY = OUT / "recovery.csv"
N_RANDOM_PLACEMENTS = 50
HOME_OFFSETS_DEG = (-5.0, -3.0, 3.0, 5.0)
MIN_ON_CALIBRATION = 0.9  # of an injected band's pixels, as for the null bands
RECOVERY_SEED = 2027
RECOVERY_MODELS = OUT / "recovery_models.csv"
RECOVERY_CLEANED = OUT / "recovery_without_false_alarm_regions.csv"
MODEL_MAPS = MAPS / "models"


def great_circle(center_ra, center_dec, rotation_deg, length_deg, n=300):
    """(ra, dec) of the arc of `length_deg` centred on a point, along a position angle."""
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
    along = np.linspace(-length_deg / 2, length_deg / 2, n)
    points = center.directional_offset_by(
        np.where(along < 0, rotation_deg + 180.0, rotation_deg) * u.deg,
        np.abs(along) * u.deg,
    )
    return points.ra.deg, points.dec.deg


def recovery_placements(name, calibration, usable, rng, n_random):
    """[(kind, offset, center_ra, center_dec, rotation, band fraction on calibration sky)].

    "home": parallel to the paper's track, offset across it; "random": a random
    calibration pixel and orientation. A placement is kept when at least
    MIN_ON_CALIBRATION of its band lies on calibration sky; random ones are
    redrawn until n_random are kept.
    """
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.objects_overlap import DES2018_STREAM_WIDTHS, des2018_arc

    sp = stream_parameters_module()
    width, length = sp.DES_STREAMS[name][:2]

    def on_calibration(ra, dec, rotation):
        band = track_band([great_circle(ra, dec, rotation, length)], width, 512)
        return (band & calibration).sum() / max(band.sum(), 1)

    placements = []
    arc_ra, arc_dec = des2018_arc(name, n=301)
    middle = SkyCoord(ra=arc_ra[150] * u.deg, dec=arc_dec[150] * u.deg)
    ahead = SkyCoord(ra=arc_ra[151] * u.deg, dec=arc_dec[151] * u.deg)
    angle = middle.position_angle(ahead)
    for offset in HOME_OFFSETS_DEG:
        center = middle.directional_offset_by(angle + 90 * u.deg, offset * u.deg)
        # the arc's own direction, carried to the offset centre
        rotation = center.position_angle(
            ahead.directional_offset_by(angle + 90 * u.deg, offset * u.deg)
        ).deg
        ra, dec = center.ra.deg, center.dec.deg
        fraction = on_calibration(ra, dec, rotation)
        placements.append(("home", offset, ra, dec, rotation, fraction))
    candidates = np.flatnonzero(calibration & usable)
    tries = 0
    while (
        sum(p[0] == "random" for p in placements) < n_random and tries < 50 * n_random
    ):
        tries += 1
        ra, dec = hp_pix2ang(int(rng.choice(candidates)))
        rotation = float(rng.uniform(0, 360))
        fraction = on_calibration(ra, dec, rotation)
        if fraction >= MIN_ON_CALIBRATION:
            placements.append(("random", np.nan, ra, dec, rotation, fraction))
    assert DES2018_STREAM_WIDTHS[name] == width
    return placements


def hp_pix2ang(pixel, nside=512):
    import healpy as hp

    ra, dec = hp.pix2ang(nside, pixel, lonlat=True)
    return float(ra), float(dec)


# Regions of the calibration sky where an out-of-fold ensemble answers at
# full confidence (> 0.9, at any queried distance): the Fornax galaxy cluster,
# the Eridanus group (NGC 1407, NGC 1332), an excess on Tucana III's eastward
# extension, and two small survey artefacts -- all in fold 0, all false alarms
# of the fold-1 ensemble (docs: "Recovery efficiency"). Taking them out of the
# calibration sky is a DIAGNOSTIC of what they cost, not a calibration: it is
# chosen from the model's own output.
FALSE_ALARM_REGIONS = OUT / "false_alarm_regions_nside512.fits.gz"
FALSE_ALARM_LEVEL = 0.9
FALSE_ALARM_MARGIN_DEG = 1.0


def false_alarm_regions(nside=512):
    """The saturated calibration pixels of the out-of-fold maps, grown by 1 degree."""
    import healpy as hp
    import numpy as np

    sp = stream_parameters_module()
    calibration = hp.read_map(CALIBRATION_MASK).astype(bool)
    hot = np.zeros(hp.nside2npix(nside), bool)
    for query in sp.QUERY_GRID:
        m = hp.read_map(MAPS / f"prediction_dm{query:.1f}.fits")
        hot |= calibration & (m != hp.UNSEEN) & (m > FALSE_ALARM_LEVEL)
    regions = np.zeros_like(hot)
    for pixel in np.flatnonzero(hot):
        regions[
            hp.query_disc(
                nside, hp.pix2vec(nside, pixel), np.radians(FALSE_ALARM_MARGIN_DEG)
            )
        ] = True
    hp.write_map(
        FALSE_ALARM_REGIONS,
        regions.astype(np.float32),
        dtype=np.float32,
        overwrite=True,
        coord="C",
    )
    area = hp.nside2pixarea(nside, degrees=True)
    print(
        f"{hot.sum()} saturated calibration pixels; with a {FALSE_ALARM_MARGIN_DEG:g} deg "
        f"margin {(regions & calibration).sum() * area:.0f} deg2 of calibration sky",
        flush=True,
    )
    return regions


def infer_models(seeds):
    """Each model's own map, over the whole inference sky, at the queried
    distances the DES 2018 streams use: what `recovery --per-model` calibrates
    a single model's false-alarm rate on. Same tiles and inputs as `infer`.
    """
    import healpy as hp
    import numpy as np
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
        WindowNormalizer,
    )
    from streamgoggles.matched_filter import WindowProjection, stitch_windows_to_healpix
    from streamgoggles.objects_overlap import get_footprint
    from streamgoggles.windows import tile_footprint

    sp = stream_parameters_module()
    configure_torch_threads(num_workers=0)
    MODEL_MAPS.mkdir(parents=True, exist_ok=True)
    background, _, pix = build_sky(INFERENCE_CATALOGUE)
    valid = background.valid_mask_full
    usable, _, _ = get_footprint("des_yr6_inference", nside=pix.nside)
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        usable & valid,
        pix.nside,
        tile_size_deg=window_deg,
        stride_deg=window_deg * STRIDE_FRACTION,
    )
    models = {
        f"fold{fold}_seed{seed}": model
        for fold in range(N_FOLDS)
        for seed, model in zip(seeds, load_ensemble(fold, seeds).models, strict=True)
    }
    channels = [
        {"filter": name, "distance_modulus": dm}
        for dm in sp.CHANNEL_DISTANCES
        for name in filters_config()
    ]
    queries = sorted(
        {
            min(sp.QUERY_GRID, key=lambda q: abs(q - d))
            for (_, _, d, _) in sp.DES_STREAMS.values()
        }
    )
    transforms = {
        query: QueryDistanceTransform(
            StreamMapTransform(normalizer=WindowNormalizer(), augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=query,
        )
        for query in queries
    }
    images = {(key, q): [] for key in models for q in queries}
    start = time.time()
    for number, tile in enumerate(tiles, start=1):
        projection = WindowProjection.for_window(tile, pix, valid)
        valid_at = valid[projection.pixnums]
        map_stack = np.stack(
            [
                projection.image(
                    np.where(
                        valid_at,
                        background.raw_map_full_dict[c["filter"]][
                            c["distance_modulus"]
                        ][projection.pixnums],
                        0.0,
                    )
                )
                for c in channels
            ]
        ).astype(np.float32)
        sample = {
            "map_stack": map_stack,
            "label_stack": np.zeros_like(map_stack),
            "valid_mask": projection.valid,
            "params": {},
            "metadata": {"channels": channels},
        }
        for query, transform in transforms.items():
            batch = torch.as_tensor(transform(sample)["map_stack"]).unsqueeze(0)
            with torch.no_grad():
                for key, model in models.items():
                    images[(key, query)].append(
                        np.where(projection.valid, model(batch)[0, 0].numpy(), np.nan)
                    )
        if number % 100 == 0:
            print(
                f"  {number}/{len(tiles)} tiles, {time.time() - start:.0f}s", flush=True
            )
    for (key, query), stack in images.items():
        image = stitch_windows_to_healpix(stack, tiles, pix, pix.nside)[0]
        image[~(usable & valid)] = np.nan
        hp.write_map(
            MODEL_MAPS / f"{key}_dm{query:.1f}.fits",
            np.where(np.isfinite(image), image, hp.UNSEEN).astype(np.float32),
            dtype=np.float32,
            overwrite=True,
            coord="C",
        )
    print(f"{len(images)} maps in {time.time() - start:.0f}s", flush=True)


def recovery(
    seeds,
    n_random=N_RANDOM_PLACEMENTS,
    streams=None,
    per_model=False,
    without_false_alarms=False,
    sb_offset=0.0,
    output=None,
):
    """Recovery efficiency of each DES 2018 stream on the real inference sky.

    Each placement injects one simulated copy of the stream -- Table 1 width,
    length, distance, surface brightness and population -- into the real
    inference maps, re-predicts the tiles covering its band with the ensemble
    of the fold that did not train on each pixel, and scores it as `detect`
    scores the real streams: false-alarm rates against the original maps'
    calibration pixels of the same fold, >= MIN_FLAGGED_PIXELS flagged within
    one width, S/N >= MIN_SNR against null bands, at the queried distance
    nearest the stream's own. The input's own S/N is recorded too: the
    injected stars in the band over the square root of the background's, in
    the matched-filter channel at that distance.

    Each fold's ensemble is also scored alone on every band, calibrated on its
    own output (``*_model{fold}`` columns). With ``per_model``, each of the
    twelve models is scored alone instead, calibrated on its own sky map
    (`infer_models`), into RECOVERY_MODELS; the placements and injections are
    the same. With ``without_false_alarms``, the calibration sky loses
    `false_alarm_regions` (a diagnostic), into RECOVERY_CLEANED. With
    ``sb_offset``, every copy is that many mag/arcsec^2 fainter, random
    placements only, into ``output`` (`detection_limit`).
    """
    import healpy as hp
    import numpy as np
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
        WindowNormalizer,
    )
    from streamgoggles.evaluation.footprint import real_track_statistics, track_band
    from streamgoggles.matched_filter import (
        WindowProjection,
        stitch_windows_to_healpix,
        window_to_healpix_indices,
    )
    from streamgoggles.objects_overlap import (
        DES2018_POPULATIONS,
        get_footprint,
        spatial_fold,
    )
    from streamgoggles.windows import tile_footprint

    sp = stream_parameters_module()
    configure_torch_threads(num_workers=0)
    background, injector, pix = build_sky(INFERENCE_CATALOGUE)
    nside = pix.nside
    valid = background.valid_mask_full
    usable, _, _ = get_footprint("des_yr6_inference", nside=nside)
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        usable & valid,
        nside,
        tile_size_deg=window_deg,
        stride_deg=window_deg * STRIDE_FRACTION,
    )
    tile_pixels = [window_to_healpix_indices(t, pix, nside)[0] for t in tiles]
    calibration = hp.read_map(CALIBRATION_MASK).astype(bool)
    if without_false_alarms:
        calibration &= ~false_alarm_regions(nside)
    ra_all, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    folds = spatial_fold(ra_all, STRIPE_DEG, N_FOLDS)
    ensembles = {fold: load_ensemble(fold, seeds) for fold in range(N_FOLDS)}
    channels = [
        {"filter": name, "distance_modulus": dm}
        for dm in sp.CHANNEL_DISTANCES
        for name in filters_config()
    ]

    # The original maps: the flagged sky every injection is compared against.
    original, reference, flagged = {}, {}, {}
    for query in sp.QUERY_GRID:
        m = hp.read_map(MAPS / f"prediction_dm{query:.1f}.fits")
        m[m == hp.UNSEEN] = np.nan
        original[query] = m
        reference[query] = {
            f: np.sort(m[np.isfinite(m) & calibration & (folds == f)])
            for f in range(N_FOLDS)
        }

    def rate(query, values, pixel_folds):
        out = np.empty(values.size)
        for f in range(N_FOLDS):
            ref = reference[query][f]
            mine = pixel_folds == f
            out[mine] = (
                ref.size - np.searchsorted(ref, values[mine], side="left")
            ) / ref.size
        return out

    for query in sp.QUERY_GRID:
        r = np.full(original[query].size, np.nan)
        finite = np.isfinite(original[query])
        r[finite] = rate(query, original[query][finite], folds[finite])
        flagged[query] = finite & (r <= TARGET_FALSE_ALARM_RATE)

    # The diagnostic: each fold's ensemble -- or, per_model, each model --
    # alone, on the whole band. An ensemble's sky is its out-of-fold output on
    # the other fold and its in-fold output on its own; a model's is its own
    # map (infer_models). Each is calibrated per fold on that same output.
    members = {f: ensembles[f].models for f in range(N_FOLDS)}
    if per_model:
        scorers = [
            (f"fold{f}_seed{seed}", f, i)
            for f in range(N_FOLDS)
            for i, seed in enumerate(seeds)
        ]
    else:
        scorers = [(f"model{f}", f, None) for f in range(N_FOLDS)]
    queries = sorted(
        {
            min(sp.QUERY_GRID, key=lambda q: abs(q - d))
            for (_, _, d, _) in sp.DES_STREAMS.values()
        }
    )
    single = {}
    for key, k, i in scorers:
        for query in queries:
            if i is None:
                m = hp.read_map(MAPS / "in_fold" / f"prediction_dm{query:.1f}.fits")
                m[m == hp.UNSEEN] = np.nan
                everywhere = np.where(folds == k, m, original[query])
            else:
                everywhere = hp.read_map(MODEL_MAPS / f"{key}_dm{query:.1f}.fits")
                everywhere[everywhere == hp.UNSEEN] = np.nan
            refs = {
                f: np.sort(
                    everywhere[np.isfinite(everywhere) & calibration & (folds == f)]
                )
                for f in range(N_FOLDS)
            }
            finite = np.isfinite(everywhere)
            r = np.full(everywhere.size, np.nan)
            for f in range(N_FOLDS):
                mine = finite & (folds == f)
                r[mine] = (
                    refs[f].size
                    - np.searchsorted(refs[f], everywhere[mine], side="left")
                ) / refs[f].size
            single[(key, query)] = (
                everywhere,
                refs,
                finite & (r <= TARGET_FALSE_ALARM_RATE),
            )

    if output is None:
        output = (
            RECOVERY_MODELS
            if per_model
            else (RECOVERY_CLEANED if without_false_alarms else RECOVERY)
        )
    done = pd.read_csv(output) if output.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    names = streams or list(sp.DES_STREAMS)
    for index, name in enumerate(sp.DES_STREAMS):
        if name not in names or (len(done) and (done["stream"] == name).any()):
            continue
        width, length, distance, sb = sp.DES_STREAMS[name]
        age, z = DES2018_POPULATIONS[name]
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
        transform = QueryDistanceTransform(
            StreamMapTransform(normalizer=WindowNormalizer(), augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=query,
        )
        good = next(
            i
            for i, c in enumerate(channels)
            if c["filter"] == "good" and c["distance_modulus"] == query
        )
        rng = np.random.default_rng([RECOVERY_SEED, index])
        start = time.time()
        for number, (kind, offset, ra, dec, rotation, on_cal) in enumerate(
            recovery_placements(name, calibration, usable & valid, rng, n_random)
        ):
            if sb_offset and kind != "random":
                continue
            params = {
                "morphology": "uniform",
                "richness": sb + sb_offset,
                "width": width,
                "length": length,
                "distance_modulus": distance,
                "age": age,
                "z": z,
                "isochrone_model": sp.ISOCHRONE_FAMILY,
                "orientation": rotation,
            }
            sky = injector.inject_streams_full_sky(
                [params],
                np.random.default_rng([RECOVERY_SEED, index, number]),
                centers=[(ra, dec)],
            )
            band = track_band([great_circle(ra, dec, rotation, length)], width, nside)
            band &= usable & valid
            covering = [i for i, px in enumerate(tile_pixels) if band[px].any()]
            images = {f: [] for f in range(N_FOLDS)}
            model_images = {key: [] for key, _, i in scorers if i is not None}
            for i in covering:
                projection = WindowProjection.for_window(tiles[i], pix, valid)
                valid_at = valid[projection.pixnums]
                stack = np.stack(
                    [
                        projection.image(
                            np.where(valid_at, full[projection.pixnums], 0.0)
                        )
                        for full in sky["map_full"]
                    ]
                ).astype(np.float32)
                sample = {
                    "map_stack": stack,
                    "label_stack": np.zeros_like(stack),
                    "valid_mask": projection.valid,
                    "params": {},
                    "metadata": {"channels": channels},
                }
                batch = torch.as_tensor(transform(sample)["map_stack"]).unsqueeze(0)
                with torch.no_grad():
                    for f in range(N_FOLDS):
                        outputs = [m(batch)[0, 0].numpy() for m in members[f]]
                        # the ensemble is the plain mean of its models
                        images[f].append(
                            np.where(projection.valid, np.mean(outputs, axis=0), np.nan)
                        )
                        for key, k, i in scorers:
                            if i is not None and k == f:
                                model_images[key].append(
                                    np.where(projection.valid, outputs[i], np.nan)
                                )
            stitched = {
                f: stitch_windows_to_healpix(
                    images[f], [tiles[i] for i in covering], pix, nside
                )[0]
                for f in range(N_FOLDS)
            }
            at = np.flatnonzero(band)
            # each pixel from the ensemble that did not train on its fold
            values = np.where(folds[at] == 0, stitched[1][at], stitched[0][at])
            finite = np.isfinite(values)
            now = flagged[query].copy()
            now[at] = False
            now[at[finite]] = (
                rate(query, values[finite], folds[at[finite]])
                <= TARGET_FALSE_ALARM_RATE
            )
            scored = np.isfinite(original[query])
            scored[at] = finite
            stats = real_track_statistics(
                now, scored, band, calibration, rng, N_NULL_BANDS
            )
            by_model = {}
            for n_scorer, (key, k, i) in enumerate(scorers):
                everywhere, refs, base = single[(key, query)]
                if i is None:
                    mine = stitched[k][at]
                else:
                    mine = stitch_windows_to_healpix(
                        model_images[key], [tiles[j] for j in covering], pix, nside
                    )[0][at]
                ok = np.isfinite(mine)
                hit = np.zeros(at.size, bool)
                for f in range(N_FOLDS):
                    sel = ok & (folds[at] == f)
                    hit[sel] = (
                        refs[f].size - np.searchsorted(refs[f], mine[sel], side="left")
                    ) / refs[f].size <= TARGET_FALSE_ALARM_RATE
                flags = base.copy()
                flags[at] = hit
                seen = np.isfinite(everywhere)
                seen[at] = ok
                k_stats = real_track_statistics(
                    flags,
                    seen,
                    band,
                    calibration,
                    np.random.default_rng([RECOVERY_SEED, index, number, n_scorer]),
                    N_NULL_BANDS,
                )
                by_model[f"n_flagged_{key}"] = k_stats["n_flagged"]
                by_model[f"snr_{key}"] = k_stats["snr"]
                by_model[f"detected_{key}"] = bool(
                    k_stats["n_flagged"] >= MIN_FLAGGED_PIXELS
                    and k_stats["snr"] >= MIN_SNR
                )
            stream_stars = sky["stream_raw_full"][good][band].sum()
            background_stars = background.raw_map_full_dict["good"][query][band].sum()
            lon, lat = hp.Rotator(coord=["C", "G"])(ra, dec, lonlat=True)
            rows.append(
                {
                    "stream": name,
                    "kind": kind,
                    "offset_deg": offset,
                    "center_ra": ra,
                    "center_dec": dec,
                    "rotation_deg": rotation,
                    "l": float(lon),
                    "b": float(lat),
                    "fold": int(np.round(folds[at].mean())),
                    "on_calibration": on_cal,
                    "distance_modulus": distance,
                    "query": query,
                    "width": width,
                    "length": length,
                    "surface_brightness": sb + sb_offset,
                    "sb_offset": sb_offset,
                    "nstars": sky["params"][0].get("nstars"),
                    "background_per_pixel": background_stars / max(band.sum(), 1),
                    "input_snr": stream_stars / np.sqrt(max(background_stars, 1.0)),
                    **stats,
                    **by_model,
                    "detected": bool(
                        stats["n_flagged"] >= MIN_FLAGGED_PIXELS
                        and stats["snr"] >= MIN_SNR
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(output, index=False)
        mine = pd.DataFrame(rows).query("stream == @name")
        print(
            f"{name}: recovered {mine.detected.sum()} of {len(mine)} "
            f"(home {mine[mine.kind == 'home'].detected.sum()}/"
            f"{(mine.kind == 'home').sum()}), {time.time() - start:.0f}s",
            flush=True,
        )
    return pd.DataFrame(rows)


def fold_figure():
    """Where each fold is: the training sky, and the calibration sky with the
    regions that saturate an out-of-fold ensemble. Also prints their areas."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import skyproj
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from streamgoggles.objects_overlap import get_footprint, spatial_fold

    nside = 512
    training, _, _ = get_footprint("des_yr6_background", nside=nside)
    inference, _, _ = get_footprint("des_yr6_inference", nside=nside)
    calibration = hp.read_map(CALIBRATION_MASK).astype(bool)
    regions = (
        hp.read_map(FALSE_ALARM_REGIONS).astype(bool)
        if FALSE_ALARM_REGIONS.exists()
        else false_alarm_regions(nside)
    )
    ra, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    folds = spatial_fold(ra, STRIPE_DEG, N_FOLDS)
    area = hp.nside2pixarea(nside, degrees=True)
    colours = ["#c9c9c9", "#1f3b73", "#8fb4e3", "#e07b39"]
    cmap = ListedColormap(colours)

    panels = []
    first = np.full(folds.size, np.nan)
    first[inference] = 0
    first[training & (folds == 0)] = 1
    first[training & (folds == 1)] = 2
    panels.append(
        (
            first,
            "the training sky of each fold",
            ["inference only (streams, never trained on)", "fold 0", "fold 1"],
        )
    )
    second = np.full(folds.size, np.nan)
    second[inference] = 0
    second[calibration & (folds == 0)] = 1
    second[calibration & (folds == 1)] = 2
    second[inference & regions] = 3
    panels.append(
        (
            second,
            "the calibration sky of each fold",
            ["not calibration", "fold 0", "fold 1", "saturated false alarms (+1 deg)"],
        )
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    fig.subplots_adjust(hspace=0.3)
    for ax, (values, title, labels) in zip(axes, panels, strict=True):
        sky = skyproj.DESSkyproj(ax=ax)
        sky.draw_hpxmap(values, zoom=False, cmap=cmap, vmin=-0.5, vmax=3.5)
        sky.ax.set_title(title, fontsize=11, pad=24)
        sky.ax.legend(
            handles=[Patch(color=c, label=t) for c, t in zip(colours, labels)],
            loc="lower left",
            fontsize=8,
            frameon=False,
        )
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOC_FIGURES / "folds.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # The three skies, nested: calibration within training within inference.
    nested = np.full(folds.size, np.nan)
    nested[inference] = 0
    nested[training] = 1
    nested[calibration] = 2
    nested[calibration & regions] = 3
    colours3 = ["#d9d9d9", "#8fb4e3", "#1f3b73", "#e07b39"]
    labels3 = [
        "inference only: the known streams, never trained on",
        "training, not calibration: structure left in the training sky",
        "calibration: stream-free sky, where false alarms are measured",
        "saturated false alarms (diagnostic, +1 deg)",
    ]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    sky = skyproj.DESSkyproj(ax=ax)
    sky.draw_hpxmap(
        nested, zoom=False, cmap=ListedColormap(colours3), vmin=-0.5, vmax=3.5
    )
    sky.ax.set_title("the three skies", fontsize=11, pad=24)
    sky.ax.legend(
        handles=[Patch(color=c, label=t) for c, t in zip(colours3, labels3)],
        loc="lower left",
        fontsize=8,
        frameon=False,
    )
    fig.savefig(DOC_FIGURES / "skies.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    for fold in range(N_FOLDS):
        mine = folds == fold
        print(
            f"fold {fold}: training {(training & mine).sum() * area:,.0f} deg2, "
            f"calibration {(calibration & mine).sum() * area:,.0f} deg2 "
            f"({(training & mine & ~calibration).sum() / (training & mine).sum():.0%}"
            f" of the training sky is not calibration), saturated regions "
            f"{(regions & calibration & mine).sum() * area:.0f} deg2",
            flush=True,
        )


DETECTION_LIMIT = OUT / "detection_limit"
SB_OFFSETS = (0.0, 0.5, 1.0, 1.5, 2.0)


def detection_limit(seeds, n_random=20):
    """Recovery against input S/N: each DES 2018 stream's copies made fainter
    by SB_OFFSETS mag/arcsec^2, on the calibration sky without the saturated
    false-alarm regions (the masks now adopted)."""
    DETECTION_LIMIT.mkdir(parents=True, exist_ok=True)
    for offset in SB_OFFSETS:
        recovery(
            seeds,
            n_random,
            without_false_alarms=True,
            sb_offset=offset,
            output=DETECTION_LIMIT / f"offset_{offset:.1f}.csv",
        )


# The real streams' input S/N with our selection, at their own distance, from
# the density profile across each (notebooks/des2018_reproduction.ipynb,
# docs: real_des/des2018_reproduction) -- the same quantity as a copy's
# input S/N: the stream's stars within one width over the square root of the
# background's there.
REAL_INPUT_SNR = {
    "ATLAS": 23.1, "Elqui": 15.7, "Jhelum": 12.0, "Phoenix": 12.9,
    "Tucana III": 11.1, "Indus": 11.0, "Chenab": 11.7, "Turranburra": 3.9,
    "Aliqa Uma": 7.9, "Wambelong": 5.2, "Willka Yaku": 8.7, "Turbio": 5.1,
    "Molonglo": 1.1, "Ravi": -0.2,
}  # fmt: skip


def detection_limit_figure():
    """Recovery of dimmed copies against their input S/N, near and far, with
    the real streams placed on the same axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = pd.concat(
        [pd.read_csv(f) for f in sorted(DETECTION_LIMIT.glob("offset_*.csv"))],
        ignore_index=True,
    )
    data = data[data.kind == "random"]
    real = pd.read_csv(STREAM_SUMMARY).set_index("stream")
    edges = np.array([0, 2, 4, 6, 8, 10, 13, 17, 22, 30, 45, 100])
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = (
        ("m−M < 16.5", data.distance_modulus < 16.5, "#8fb4e3"),
        ("m−M ≥ 16.5", data.distance_modulus >= 16.5, "#1f3b73"),
    )
    summary = {}
    for label, mask, colour in groups:
        rows = data[mask]
        bins = pd.cut(rows.input_snr, edges)
        rate = rows.groupby(bins, observed=True).detected.agg(["mean", "size"])
        rate = rate[rate["size"] >= 5]
        centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
        ax.plot(
            centres, rate["mean"], "o-", color=colour, lw=2, label=f"copies, {label}"
        )
        summary[label] = rate
    for name, snr in REAL_INPUT_SNR.items():
        found = bool(real.loc[name, "detected"])
        near = real.loc[name, "distance_modulus"] < 16.5
        y = 1.04 if found else -0.04
        ax.plot(
            max(snr, 0.6),
            y,
            "D",
            ms=6,
            color="#1f3b73" if not near else "#8fb4e3",
            markeredgecolor="black" if found else "#b0b0b0",
        )
        ax.annotate(
            name,
            (max(snr, 0.6), y),
            xytext=(0, 7 if found else -12),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            rotation=35,
        )
    ax.set_xscale("log")
    ax.set_xlabel("S/N of the stream in the matched-filter input")
    ax.set_ylabel("fraction of copies recovered")
    ax.set_ylim(-0.2, 1.25)
    ax.axhline(0, color="#e6e6e6", lw=0.8)
    ax.axhline(1, color="#e6e6e6", lw=0.8)
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            color="#1f3b73",
            markeredgecolor="black",
            label="real stream, found (top)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            color="#1f3b73",
            markeredgecolor="#b0b0b0",
            label="real stream, missed (bottom)",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOC_FIGURES / "detection_limit.png", dpi=120, bbox_inches="tight")
    # what the copies predict for the real streams: each stream's chance of
    # detection at its real input S/N, from copies at the same distance class
    expected = 0.0
    for name, snr in REAL_INPUT_SNR.items():
        near = real.loc[name, "distance_modulus"] < 16.5
        rate = summary["m−M < 16.5" if near else "m−M ≥ 16.5"]
        chance = next(
            (
                r["mean"]
                for b, r in rate.iterrows()
                if b.left < max(snr, 0.1) <= b.right
            ),
            0.0,
        )
        expected += chance
        print(
            f"{name:12s} input S/N {snr:5.1f}  expected {chance:.2f}  found {bool(real.loc[name, 'detected'])}"
        )
    print(f"expected detections {expected:.1f}, found {int(real.detected.sum())}")
    plt.close(fig)
    for label, rate in summary.items():
        print(label)
        print(rate.round(2).to_string())


def recovery_figures():
    """Recovery per stream and fold; the simulated copies' input against the real streams'."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = pd.read_csv(RECOVERY)
    random = data[data.kind == "random"]
    real = pd.read_csv(STREAM_SUMMARY).set_index("stream")
    order = (
        random.groupby("stream").distance_modulus.first().sort_values().index.tolist()
    )
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    def wilson(k, n, z=1.0):
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(order))
    styles = {
        "all": {"color": "#1f3b73", "marker": "o", "ms": 7, "offset": 0.0},
        0: {"color": "#6f9fd8", "marker": "v", "ms": 6, "offset": -0.22},
        1: {"color": "#6f9fd8", "marker": "^", "ms": 6, "offset": 0.22},
    }
    labels = {"all": "all 50 places", 0: "fold 0 places", 1: "fold 1 places"}
    for key, style in styles.items():
        rows = random if key == "all" else random[random.fold == key]
        stats = rows.groupby("stream").detected.agg(["sum", "size"]).reindex(order)
        rate = stats["sum"] / stats["size"]
        low, high = wilson(stats["sum"].to_numpy(), stats["size"].to_numpy())
        ax.errorbar(
            x + style["offset"],
            rate,
            yerr=[np.clip(rate - low, 0, None), np.clip(high - rate, 0, None)],
            fmt=style["marker"],
            color=style["color"],
            ms=style["ms"],
            lw=1,
            capsize=0,
            label=labels[key],
        )
    for i, name in enumerate(order):
        found = bool(real.loc[name, "detected"])
        ax.text(
            i,
            1.08,
            "✔" if found else "✘",
            ha="center",
            fontsize=11,
            color="#1f3b73" if found else "#b0b0b0",
        )
    ax.text(-0.9, 1.08, "real:", ha="right", fontsize=9, color="#3a3a38")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{n}\n{random[random.stream == n].distance_modulus.iloc[0]:g}"
            for n in order
        ],
        fontsize=8,
        rotation=30,
        ha="right",
    )
    ax.set_ylim(-0.03, 1.15)
    ax.set_xlim(-1.2, len(order) - 0.5)
    ax.set_ylabel("fraction recovered")
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    fig.savefig(DOC_FIGURES / "recovery_by_stream.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    summary = (
        random.groupby("stream")
        .input_snr.describe(percentiles=[0.1, 0.5, 0.9])
        .reindex(order)
    )
    ax.vlines(
        x,
        summary["10%"],
        summary["90%"],
        color="#6f9fd8",
        lw=5,
        label="simulated copies (10-90%)",
    )
    ax.plot(
        x, summary["50%"], "o", color="#1f3b73", ms=6, label="simulated copies, median"
    )
    ax.plot(
        x,
        real.loc[order, "input_snr"],
        "D",
        color="#e07b39",
        ms=6,
        label="the real stream (side bands)",
    )
    ax.axhline(0, color="#b0b0b0", lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-10, -5, 0, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["−10", "−5", "0", "5", "10", "20", "50", "100"])
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("S/N in the matched-filter input")
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(DOC_FIGURES / "recovery_input.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "step",
        choices=[
            "train",
            "infer",
            "figures",
            "calibration",
            "detect",
            "recovery",
            "recovery-figures",
            "infer-models",
            "folds",
            "detection-limit",
            "detection-limit-figure",
        ],
    )
    parser.add_argument("--streams", nargs="+", help="recovery: only these streams")
    parser.add_argument("--placements", type=int, default=N_RANDOM_PLACEMENTS)
    parser.add_argument(
        "--per-model", action="store_true", help="recovery: score each model alone"
    )
    parser.add_argument(
        "--without-false-alarms",
        action="store_true",
        help="recovery: calibrate without the saturated false-alarm regions",
    )
    parser.add_argument("--fold", type=int, choices=list(range(N_FOLDS)))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    arguments = parser.parse_args()
    if arguments.step == "train":
        if arguments.fold is None:
            parser.error("train needs --fold")
        train(arguments.fold, arguments.seeds)
    elif arguments.step == "infer":
        infer(arguments.seeds)
    elif arguments.step == "calibration":
        calibration_mask()
    elif arguments.step == "detect":
        detect()
    elif arguments.step == "recovery":
        recovery(
            arguments.seeds,
            arguments.placements,
            arguments.streams,
            arguments.per_model,
            arguments.without_false_alarms,
        )
        if not (arguments.per_model or arguments.without_false_alarms):
            recovery_figures()
    elif arguments.step == "infer-models":
        infer_models(arguments.seeds)
    elif arguments.step == "folds":
        fold_figure()
    elif arguments.step == "detection-limit":
        detection_limit(arguments.seeds)
        detection_limit_figure()
    elif arguments.step == "detection-limit-figure":
        detection_limit_figure()
    elif arguments.step == "recovery-figures":
        recovery_figures()
    else:
        figures()
