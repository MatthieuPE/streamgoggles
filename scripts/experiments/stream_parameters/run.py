"""Stream parameters: detection across the range of the DES 2018 streams.

The model answers for one queried trial distance ``dm``. Its input, per window:

  0. matched-filter map at dm - 0.5
  1. matched-filter map at dm
  2. matched-filter map at dm + 0.5
  3. fixed decoy box map (colour 1.2-1.5, g 18-24.5; the same at every distance)
  4-6. constant maps holding the distances of maps 0-2, scaled (dm - 17) / 2

Its output is one map: the stream stars the matched filter at dm selects,
thresholded at one star per pixel (`datasets.transforms.QueryDistanceTransform`).
At inference the query slides over 15, 15.5, ... 19.

Training draws each stream's parameters from ranges that bracket the DES 2018
streams (Shipp et al. 2018) without matching them, so the model is not tuned
to those streams:

  distance modulus   uniform 15-19           (DES streams 15.6-18.5)
  width              log-uniform 0.1-1.5 deg (0.16-1.16)
  length             uniform 4-30 deg        (4.8-29.2)
  surface brightness uniform 32-34.5         (31.9-34.3)
  distance gradient  uniform +-0.2 mag/deg, at most 1.5 mag end to end
  age, metallicity   fixed 12.5 Gyr, Z = 0.0002

The rest is the configuration selected by the hyperparameter experiment (batch
Dice, batch 8, background fraction 0.05, U-Net depth 2 width 12), on DES year 6
at RA 0, Dec -50. Models are the quick tier (4800 windows) unless --windows
says otherwise.

Run from the repository root:
  python scripts/experiments/stream_parameters/run.py [--seeds 42 43] [--windows 4800]
"""

import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "stream_parameters"
MODELS = OUT / "models"
RESULTS = OUT / "results.pkl"

SURVEY, RELEASE = "des", "yr6"
REGION = {"center_ra": 0.0, "center_dec": -50.0, "width_deg": 25.0, "height_deg": 18.0}
CLIPPING = {"g": {"min": 16.0, "max": 24.5}, "r": {"min": 16.0, "max": 24.5}}
FILTERS = {
    "good": {"type": "isochrone", "reference_isochrone": {"age": 12.5, "z": 0.0002}},
    "decoy": {"type": "box", "color_range": (1.2, 1.5), "mag_range": (18.0, 24.5)},
}
STEP = 0.5
QUERY_GRID = [15.0 + STEP * i for i in range(9)]  # 15 ... 19
CHANNEL_DISTANCES = [14.5 + STEP * i for i in range(11)]  # 14.5 ... 19.5
BACKGROUND_SEED, INDEPENDENT_BACKGROUND_SEED = 20260915, 777
TRAINING = {
    "epochs": 40,
    "batch_size": 8,
    "lr": 2e-3,
    "background_fraction": 0.05,
    "loss_name": "batch_dice",
}
MODEL = {"depth": 2, "base_width": 12, "head": "sigmoid"}
N_VALIDATION = 40

# Known-location evaluation: a grid over the parameters that matter most,
# each point at a queried distance equal to the stream's own.
EVAL_DISTANCES = [15.0, 16.0, 17.0, 18.0, 19.0]
EVAL_WIDTHS = [0.2, 0.6, 1.2]
EVAL_SB = [32.0, 33.0, 34.0]
EVAL_LENGTH = 15.0
N_REALIZATIONS = 20
N_NULL_BANDS = 200
EVAL_SEED = 2026


def training_parameters():
    from streamgoggles.config import DistributionType, ParameterSpec

    def fixed(name, value):
        return ParameterSpec(name=name, dist_type=DistributionType.FIXED, value=value)

    def uniform(name, low, high, kind=DistributionType.UNIFORM):
        return ParameterSpec(name=name, dist_type=kind, min_val=low, max_val=high)

    return {
        "morphology": fixed("morphology", "uniform"),
        "richness": uniform("richness", 32.0, 34.5),
        "width": uniform("width", 0.1, 1.5, DistributionType.LOG_UNIFORM),
        "length": uniform("length", 4.0, 30.0),
        "distance_modulus": uniform("distance_modulus", 15.0, 19.0),
        "distance_gradient": uniform("distance_gradient", -0.2, 0.2),
        "max_distance_change": fixed("max_distance_change", 1.5),
        "age": fixed("age", 12.5),
        "z": fixed("z", 0.0002),
    }


def build_sky(background_seed):
    """Background, filters and injector on DES year 6 at the study region."""
    from streamgoggles.background import Background
    from streamgoggles.background_sources import (
        StreamObsLightBackgroundSource,
        StudyRegion,
    )
    from streamgoggles.injector import StreamInjector
    from streamgoggles.matched_filter import PixelizationSpec, build_matched_filters
    from streamgoggles.storage import BackgroundMapStore
    from streamgoggles.stream_sources import StreamObsSource

    namespace = f"{SURVEY}_{RELEASE}"
    pix = PixelizationSpec(nside=512, image_size_pix=(96, 96))
    filters = build_matched_filters(FILTERS, namespace=namespace)
    background = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={"seed": background_seed},
        study_region=StudyRegion(**REGION),
        cuts=[],
        clipping=CLIPPING,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=CHANNEL_DISTANCES,
        finalize_cfg={"enabled": False},
        store=BackgroundMapStore(tempfile.mkdtemp(prefix="stream_parameters_bg_")),
        pix=pix,
        survey=SURVEY,
        release=RELEASE,
        filter_configs=FILTERS,
    )
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
    return background, injector


def train(seed, windows, background, injector):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from streamgoggles.config import StreamConfig
    from streamgoggles.datasets.stream_map_dataset import (
        StreamMapDataset,
        TransformedDataset,
        configure_torch_threads,
        default_num_workers,
        stream_map_collate_fn,
    )
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        RobustNormalizer,
        StreamMapTransform,
    )
    from streamgoggles.models.losses import get_loss
    from streamgoggles.models.unet import UNet
    from streamgoggles.storage import SimulationStore
    from streamgoggles.training.plain_runner import PlainTrainer

    config = StreamConfig(
        params=training_parameters(),
        background_fraction=TRAINING["background_fraction"],
        persist=False,
        richness_kind="surface_brightness",
    )

    def dataset(rng_seed, steps):
        return StreamMapDataset(
            config=config,
            background=background,
            injector=injector,
            store=SimulationStore(tempfile.mkdtemp(prefix="stream_parameters_sims_")),
            eval_mode=False,
            steps_per_epoch=steps,
            rng=np.random.default_rng(rng_seed),
        )

    train_dataset = dataset(seed, windows // TRAINING["epochs"])
    # Normalization: every (distance, filter) map with its own statistics.
    fit = [train_dataset[i] for i in range(8)]
    normalizer = RobustNormalizer()
    normalizer.fit(
        np.concatenate([s["map_stack"] for s in fit], axis=1),
        np.concatenate([s["valid_mask"] for s in fit], axis=0),
    )

    def query_view(augment, rng_seed):
        return QueryDistanceTransform(
            StreamMapTransform(
                normalizer=normalizer,
                augment=augment,
                rng=np.random.default_rng(rng_seed),
            ),
            query_grid=QUERY_GRID,
            step=STEP,
            rng=np.random.default_rng(rng_seed + 1),
        )

    # Validation: a fixed set of windows and queries, the same every epoch.
    validation_source = dataset(seed + 1000, N_VALIDATION)
    validation_view = query_view(False, seed + 2000)
    validation = [validation_view(validation_source[i]) for i in range(N_VALIDATION)]

    # Head bias at the class prior of the queried label.
    fit_view = query_view(False, seed + 3000)
    positive = np.mean(
        [fit_view(s)["label_stack"][0][s["valid_mask"]].mean() for s in fit]
    )
    positive = float(np.clip(positive, 1e-3, 1 - 1e-3))

    torch.manual_seed(seed)
    model = UNet(
        in_channels=QueryDistanceTransform.n_channels,
        out_channels=1,
        **MODEL,
    )
    with torch.no_grad():
        model.head_conv.bias.fill_(float(np.log(positive / (1 - positive))))
        model.head_conv.weight.zero_()

    num_workers = default_num_workers()
    configure_torch_threads(num_workers=num_workers)
    loader = {"collate_fn": stream_map_collate_fn, "num_workers": num_workers}
    if num_workers > 0:
        loader["persistent_workers"] = True
    train_dl = DataLoader(
        TransformedDataset(train_dataset, query_view(True, seed + 1)),
        batch_size=TRAINING["batch_size"],
        **loader,
    )
    val_dl = DataLoader(
        validation,
        batch_size=TRAINING["batch_size"],
        collate_fn=stream_map_collate_fn,
    )
    trainer = PlainTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=TRAINING["lr"]),
        loss_fn=get_loss(TRAINING["loss_name"]),
        device="cpu",
    )
    result = trainer.train(
        train_dl,
        val_dl=val_dl,
        epochs=TRAINING["epochs"],
    )
    return model, normalizer, result


def main(seeds, windows):
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import numpy as np
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        RobustNormalizer,
        StreamMapTransform,
    )
    from streamgoggles.evaluation.footprint import (
        THRESHOLD_GRID,
        evaluate_footprint_realizations,
    )
    from streamgoggles.models.unet import UNet

    MODELS.mkdir(parents=True, exist_ok=True)
    background, injector = build_sky(BACKGROUND_SEED)
    _, evaluation_injector = build_sky(INDEPENDENT_BACKGROUND_SEED)
    channels = [
        {"filter": name, "distance_modulus": d}
        for d in CHANNEL_DISTANCES
        for name in FILTERS
    ]

    frames = [pd.read_pickle(RESULTS)] if RESULTS.exists() else []
    done = (
        set(
            map(
                tuple,
                frames[0][["seed", "windows", "eval_set"]].drop_duplicates().to_numpy(),
            )
        )
        if frames
        else set()
    )

    for seed in seeds:
        stem = f"w{windows}_seed{seed}"
        weights, meta = MODELS / f"{stem}.pt", MODELS / f"{stem}.json"
        if weights.exists():
            saved = json.loads(meta.read_text())
            model = UNet(
                in_channels=QueryDistanceTransform.n_channels, out_channels=1, **MODEL
            )
            model.load_state_dict(torch.load(weights))
            normalizer = RobustNormalizer()
            normalizer.mean = np.array(saved["normalizer_mean"], dtype=np.float32)
            normalizer.std = np.array(saved["normalizer_std"], dtype=np.float32)
        else:
            start = time.time()
            model, normalizer, result = train(seed, windows, background, injector)
            torch.save(model.state_dict(), weights)
            meta.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "windows": windows,
                        "training_parameters": {
                            k: str(v) for k, v in training_parameters().items()
                        },
                        "normalizer_mean": [float(v) for v in normalizer.mean],
                        "normalizer_std": [float(v) for v in normalizer.std],
                        "train_losses": result["train_losses"],
                        "val_losses": result["val_losses"],
                        "train_s": time.time() - start,
                    }
                )
            )
            print(f"{stem}: trained in {time.time() - start:.0f}s", flush=True)
        model.eval()
        configure_torch_threads(num_workers=0)

        for distance in EVAL_DISTANCES:
            for width in EVAL_WIDTHS:
                for sb in EVAL_SB:
                    eval_set = f"dm{distance:g}_w{width:g}_sb{sb:g}"
                    if (seed, windows, eval_set) in done:
                        continue
                    transform = QueryDistanceTransform(
                        StreamMapTransform(normalizer=normalizer, augment=False),
                        query_grid=QUERY_GRID,
                        step=STEP,
                        query=distance,
                    )
                    label_channel = QueryDistanceTransform.channel_index(
                        channels, "good", distance
                    )
                    params = {
                        "morphology": "uniform",
                        "richness": sb,
                        "width": width,
                        "length": EVAL_LENGTH,
                        "distance_modulus": distance,
                        "age": 12.5,
                        "z": 0.0002,
                    }
                    start = time.time()
                    with torch.no_grad():
                        scored = evaluate_footprint_realizations(
                            model,
                            evaluation_injector,
                            transform,
                            [params],
                            n_realizations=N_REALIZATIONS,
                            channel=0,
                            label_channel=label_channel,
                            seed=EVAL_SEED,
                            thresholds=THRESHOLD_GRID,
                            n_null_bands=N_NULL_BANDS,
                        )
                    frames.append(
                        scored.assign(seed=seed, windows=windows, eval_set=eval_set)
                    )
                    pd.concat(frames, ignore_index=True).to_pickle(RESULTS)
                    print(
                        f"{stem} {eval_set}: scored in {time.time() - start:.0f}s",
                        flush=True,
                    )
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--windows", type=int, default=4800)
    arguments = parser.parse_args()
    main(arguments.seeds, arguments.windows)
