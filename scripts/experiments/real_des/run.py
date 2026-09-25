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

    from streamgoggles.objects_overlap import (
        get_dwarf,
        get_footprint,
        get_GC,
        mask_objects,
        mask_streams,
    )

    usable, _, _ = get_footprint("des_yr6_background", nside=nside)
    covered, _, _ = get_footprint("des_yr6", nside=nside)
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

    def near_footprint(catalogue, reach_deg):
        near = []
        for row in catalogue:
            vector = hp.ang2vec(float(row["ra"]), float(row["dec"]), lonlat=True)
            disc = hp.query_disc(nside, vector, np.radians(reach_deg))
            near.append(bool(covered[disc].any()))
        return np.array(near)

    dwarfs = get_dwarf()
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
    removed |= dwarf_mask | cluster_mask
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
    """One track per stream for its detection band: the DES measurement.

    The masks use every reference where that is the cautious choice; a
    detection band must follow the stream as DES measured it. Chenab's mask
    covers the whole Orphan-Chenab stream, but its band is Chenab's DES
    segment, or "detecting Chenab" would mean detecting Orphan. Where
    galstreams has no DES track, the reference whose length matches DES's
    (ATLAS: Li et al. 2021, 23.6 deg against 22.6).
    """
    from streamgoggles.objects_overlap import STREAM_TRACKS

    return {
        **STREAM_TRACKS,
        "Chenab": ("Orphan-Chenab.shipp2019",),
        "ATLAS": ("AAU-ATLAS.li2021",),
        "Aliqa Uma": ("AAU-AliqaUma.li2021",),
        "Molonglo": ("Molonglo.grillmair2017b",),
    }


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
    from streamgoggles.objects_overlap import get_footprint, spatial_fold, stream_tracks

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
        name: track_band(
            list(stream_tracks(name, tracks=chosen).values()), width, nside
        )
        & inference
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
    from streamgoggles.objects_overlap import get_footprint, stream_tracks

    sp = stream_parameters_module()
    background, _, _ = build_sky(INFERENCE_CATALOGUE)
    inference, _, _ = get_footprint("des_yr6_inference", nside=512)
    valid = background.valid_mask_full & inference
    chosen = detection_tracks()
    rows = []
    for row in nearest.itertuples():
        width = sp.DES_STREAMS[row.stream][0]
        tracks = list(stream_tracks(row.stream, tracks=chosen).values())
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "step", choices=["train", "infer", "figures", "calibration", "detect"]
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
    else:
        figures()
