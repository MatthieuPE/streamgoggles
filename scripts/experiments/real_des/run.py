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
    parser.add_argument("step", choices=["train", "infer", "figures"])
    parser.add_argument("--fold", type=int, choices=list(range(N_FOLDS)))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    arguments = parser.parse_args()
    if arguments.step == "train":
        if arguments.fold is None:
            parser.error("train needs --fold")
        train(arguments.fold, arguments.seeds)
    elif arguments.step == "infer":
        infer(arguments.seeds)
    else:
        figures()
