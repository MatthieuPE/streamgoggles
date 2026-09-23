"""Stream parameters: detection across the range of the DES 2018 streams.

The model answers for one queried trial distance ``dm``. Its input, per window:

  0. matched-filter map at dm - 0.5
  1. matched-filter map at dm
  2. matched-filter map at dm + 0.5
  3. fixed decoy box map (colour 1.2-1.5, g 18-24.5; the same at every distance)
  4-6. constant maps holding the distances of maps 0-2, scaled (dm - 17) / 2

Each map is standardized from its own window: (x - mean) / std over the
window's valid pixels, channel by channel, with nothing fitted on other
windows or on the background. Its output is one map: the stream stars the
matched filter at dm selects,
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
  age, metallicity   fixed 12 Gyr, Z = 0.0002

The rest is the configuration selected by the hyperparameter experiment (batch
Dice, batch 8, background fraction 0.05, U-Net depth 2 width 12), on DES year 6
at RA 0, Dec -50. Models are the quick tier (4800 windows) unless --windows
says otherwise.

--training-set replaces those ranges, to ask what aiming at this population
bought: "wide" widens the geometry well past the DES streams, "population"
draws age and metallicity instead of fixing them.

--mode chooses what the trained models are scored on: the parameter "grid",
the "des" streams with their own parameters, or an "isochrone" scan where the
injected population differs from the one the matched filter assumes.

--ensemble scores the seeds' averaged output maps as one predictor instead of
scoring each training on its own.

Run from the repository root:
  python scripts/experiments/stream_parameters/run.py [--seeds 42 43]
      [--windows 4800] [--mode grid|des|isochrone] [--ensemble]
      [--training-set des|wide|population] [--realizations 20]
"""

import argparse
import json
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "stream_parameters"
MODELS = OUT / "models"
RESULTS = OUT / "results.pkl"

SURVEY, RELEASE = "des", "yr6"
REGION = {"center_ra": 0.0, "center_dec": -50.0, "width_deg": 25.0, "height_deg": 18.0}
CLIPPING = {"g": {"min": 16.0, "max": 24.5}, "r": {"min": 16.0, "max": 24.5}}
# ugali picks an isochrone by nearest neighbour among the files on disk,
# without saying so. des/marigo2017 held four of them when the models below
# were trained -- 10 and 12 Gyr, Z = 0.0001 and 0.0002 -- so the 12.5 Gyr this
# experiment asked for was always the 12.0 Gyr file, and is named honestly
# here. The full grid (126 ages x 91 metallicities) has since been installed,
# which leaves those four files untouched and the filter identical vertex for
# vertex; keeping 12.0 here is what makes that true, since asking for 12.5 now
# would resolve to a file no trained model ever saw.
POPULATION = {"age": 12.0, "z": 0.0002}
FILTERS = {
    "good": {"type": "isochrone", "reference_isochrone": dict(POPULATION)},
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

# The DES streams of Shipp et al. (2018), Tables 1 and 2: width (deg), length
# (deg), distance modulus, surface brightness (mag/arcsec^2). Palca has no
# published width or surface brightness and is left out. An evaluation set
# only -- training never sees these values (it draws from the ranges above),
# so the model is not tuned to them. Their stellar populations are not
# modelled: every injected stream keeps this experiment's isochrone
# (12 Gyr, Z = 0.0002).
DES_STREAMS = {
    "Tucana III": (0.18, 4.8, 17.0, 32.0),
    "ATLAS": (0.24, 22.6, 16.8, 33.0),
    "Molonglo": (0.32, 7.4, 16.8, 33.0),
    "Phoenix": (0.16, 13.6, 16.4, 32.6),
    "Indus": (0.83, 20.3, 16.1, 31.9),
    "Jhelum": (1.16, 29.2, 15.6, 33.3),
    "Ravi": (0.72, 16.6, 16.8, 33.4),
    "Chenab": (0.71, 18.5, 18.0, 34.1),
    "Elqui": (0.54, 9.4, 18.5, 34.3),
    "Aliqa Uma": (0.26, 10.0, 17.3, 33.8),
    "Turbio": (0.25, 15.0, 16.1, 32.6),
    "Willka Yaku": (0.21, 6.4, 17.7, 32.9),
    "Turranburra": (0.60, 16.9, 17.2, 34.0),
    "Wambelong": (0.40, 14.2, 15.9, 33.7),
}
# A stream longer than the study region leaves too few placements for the
# background bands the detection test needs, so a longer one is evaluated on a
# segment of this length. Conservative: more track can only help.
MAX_EVAL_LENGTH = 15.0

# Isochrone mismatch: the stream's population differs from the one the matched
# filter assumes, which is also the only population the model was trained on.
# Scanned in the filter's own family, so age and metallicity are the only
# things that differ. Bressan2012 at the filter's own values is scanned as
# well: that point differs only by the isochrone family, which measures that
# systematic separately instead of leaving it mixed in.
ISOCHRONE_MODEL = "Marigo2017"
ISOCHRONE_OTHER_FAMILY = "Bressan2012"
ISOCHRONE_AGES = [9.0, 10.5, 12.0, 13.5]
ISOCHRONE_Z = [0.0001, 0.0002, 0.0005, 0.001]
ISOCHRONE_CORNERS = [(9.0, 0.001), (13.5, 0.0001)]
# Two operating points: one with margin to lose (width 0.6 at SB 34, which the
# grid recovers 93% of) and one already on the edge (width 0.2, 33%), since a
# point the models recover 100% of can only show that nothing broke.
ISOCHRONE_GEOMETRY = [(0.6, 34.0), (0.2, 34.0)]
ISOCHRONE_LENGTH, ISOCHRONE_DISTANCE = 15.0, 17.0


def isochrone_populations():
    """(age, Z, isochrone family) of the mismatch scan.

    Age scanned at the filter's metallicity, metallicity at the filter's age,
    two corners with both wrong, and the filter's own pair in both families.
    """
    pairs = [(age, POPULATION["z"]) for age in ISOCHRONE_AGES]
    pairs += [
        (POPULATION["age"], z)
        for z in ISOCHRONE_Z
        if (POPULATION["age"], z) not in pairs
    ]
    pairs += [pair for pair in ISOCHRONE_CORNERS if pair not in pairs]
    return [(age, z, ISOCHRONE_MODEL) for age, z in pairs] + [
        (POPULATION["age"], POPULATION["z"], ISOCHRONE_OTHER_FAMILY)
    ]


# Three training sets, each a set of ranges the stream parameters are drawn
# from. "des" brackets the DES 2018 streams, and is what every result before
# this section used. The other two ask what it costs to stop aiming at that
# population:
#   "wide"       the same fixed isochrone, but geometry ranges well past the
#                DES streams on every side (a model that is not told where to
#                look in parameter space)
#   "population" the DES ranges, with age and metallicity drawn instead of
#                fixed, so the model sees streams the matched filter's single
#                isochrone does not describe
TRAINING_SETS = {
    "des": {},
    "wide": {
        "richness": (31.0, 36.0),
        "width": (0.05, 3.0, "log"),
        "length": (3.0, 30.0),
        "distance_gradient": (-0.4, 0.4),
    },
    "population": {
        "age": (9.0, 13.5),
        "z": (0.0001, 0.001, "log"),
    },
}


def training_parameters(training_set="des"):
    from streamgoggles.config import DistributionType, ParameterSpec

    def fixed(name, value):
        return ParameterSpec(name=name, dist_type=DistributionType.FIXED, value=value)

    def uniform(name, low, high, kind=DistributionType.UNIFORM):
        return ParameterSpec(name=name, dist_type=kind, min_val=low, max_val=high)

    def override(spec):
        """The training set's own value for this parameter, if it has one: a
        (low, high[, "log"]) range, or a single value held fixed."""
        changed = TRAINING_SETS[training_set].get(spec.name)
        if changed is None:
            return spec
        if not isinstance(changed, tuple):
            return fixed(spec.name, changed)
        low, high, *log = changed
        kind = DistributionType.LOG_UNIFORM if log else DistributionType.UNIFORM
        return uniform(spec.name, low, high, kind)

    return {
        name: override(spec)
        for name, spec in {
            "morphology": fixed("morphology", "uniform"),
            "richness": uniform("richness", 32.0, 34.5),
            "width": uniform("width", 0.1, 1.5, DistributionType.LOG_UNIFORM),
            "length": uniform("length", 4.0, 30.0),
            "distance_modulus": uniform("distance_modulus", 15.0, 19.0),
            "distance_gradient": uniform("distance_gradient", -0.2, 0.2),
            "max_distance_change": fixed("max_distance_change", 1.5),
            "age": fixed("age", POPULATION["age"]),
            "z": fixed("z", POPULATION["z"]),
            "isochrone_model": fixed("isochrone_model", "Marigo2017"),
        }.items()
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


def train(seed, windows, background, injector, training_set="des"):
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
        StreamMapTransform,
        WindowNormalizer,
    )
    from streamgoggles.models.losses import get_loss
    from streamgoggles.models.unet import UNet
    from streamgoggles.storage import SimulationStore
    from streamgoggles.training.plain_runner import PlainTrainer

    config = StreamConfig(
        params=training_parameters(training_set),
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
    # Each window's maps are standardized from that window alone (see the
    # docs page): nothing is fitted, so the same holds on real data.
    normalizer = WindowNormalizer()

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
    fit = [train_dataset[i] for i in range(8)]
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


class Ensemble:
    """Several trainings scored as one predictor: their output maps averaged.

    Every model of this experiment standardizes its input from the window
    itself (`WindowNormalizer`), so they all take exactly the same input and
    the average is over their output probabilities alone. A survey search
    deploying six models would read this one map, not six.
    """

    def __init__(self, models):
        import torch

        self.models = models
        self.torch = torch

    def eval(self):
        for model in self.models:
            model.eval()
        return self

    def __call__(self, x):
        return self.torch.stack([model(x) for model in self.models]).mean(dim=0)


def evaluation_points(mode):
    """(name, stream parameters) to evaluate, for one of the three modes.

    "grid"      distance x width x surface brightness, at the experiment's own
                isochrone
    "des"       the DES 2018 streams with their own parameters
    "isochrone" fixed geometry, population (age, Z) scanned away from the one
                the matched filter assumes
    """

    def point(name, width, length, distance_modulus, sb, **population):
        return name, {
            "morphology": "uniform",
            "richness": sb,
            "width": width,
            "length": length,
            "distance_modulus": distance_modulus,
            "age": POPULATION["age"],
            "z": POPULATION["z"],
            "isochrone_model": "Marigo2017",
            **population,
        }

    if mode == "des":
        return [
            point(name, width, min(length, MAX_EVAL_LENGTH), distance, sb)
            for name, (width, length, distance, sb) in DES_STREAMS.items()
        ]
    if mode == "isochrone":
        return [
            point(
                f"age{age:g}_z{z:g}_{model}_w{width:g}_sb{sb:g}",
                width=width,
                length=ISOCHRONE_LENGTH,
                distance_modulus=ISOCHRONE_DISTANCE,
                sb=sb,
                age=age,
                z=z,
                isochrone_model=model,
            )
            for age, z, model in isochrone_populations()
            for width, sb in ISOCHRONE_GEOMETRY
        ]
    return [
        point(f"dm{distance:g}_w{width:g}_sb{sb:g}", width, EVAL_LENGTH, distance, sb)
        for distance in EVAL_DISTANCES
        for width in EVAL_WIDTHS
        for sb in EVAL_SB
    ]


def main(
    seeds,
    windows,
    mode="grid",
    ensemble=False,
    realizations=N_REALIZATIONS,
    training_set="des",
):
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
        WindowNormalizer,
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

    results_file = (
        OUT
        / {
            "grid": "results.pkl",
            "des": "des_results.pkl",
            "isochrone": "isochrone_results.pkl",
        }[mode]
    )
    suffix = ("" if training_set == "des" else f"_{training_set}") + (
        "_ensemble" if ensemble else ""
    )
    results_file = results_file.with_name(
        results_file.stem + suffix + results_file.suffix
    )
    frames = [pd.read_pickle(results_file)] if results_file.exists() else []
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

    def model_stem(seed):
        name = "ensemble" if seed == -1 else f"seed{seed}"
        prefix = "" if training_set == "des" else f"{training_set}_"
        return f"{prefix}w{windows}_{name}"

    def load_or_train(seed):
        stem = model_stem(seed)
        weights, meta = MODELS / f"{stem}.pt", MODELS / f"{stem}.json"
        if weights.exists():
            saved = json.loads(meta.read_text())
            model = UNet(
                in_channels=QueryDistanceTransform.n_channels, out_channels=1, **MODEL
            )
            model.load_state_dict(torch.load(weights))
            print(f"{stem}: loaded ({saved['train_s']:.0f}s of training)", flush=True)
        else:
            start = time.time()
            model, _, result = train(seed, windows, background, injector, training_set)
            torch.save(model.state_dict(), weights)
            meta.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "windows": windows,
                        "training_set": training_set,
                        "training_parameters": {
                            k: str(v)
                            for k, v in training_parameters(training_set).items()
                        },
                        "normalization": "per window and channel (WindowNormalizer)",
                        "train_losses": result["train_losses"],
                        "val_losses": result["val_losses"],
                        "train_s": time.time() - start,
                    }
                )
            )
            print(f"{stem}: trained in {time.time() - start:.0f}s", flush=True)
        return model

    # One entry per training, or a single entry averaging all of them. The
    # ensemble is labelled seed -1 so the two never mix in a results file.
    if ensemble:
        to_score = [(-1, lambda: Ensemble([load_or_train(s) for s in seeds]))]
    else:
        to_score = [(seed, partial(load_or_train, seed)) for seed in seeds]

    for seed, build in to_score:
        stem = model_stem(seed)
        if all(
            (seed, windows, eval_set) in done for eval_set, _ in evaluation_points(mode)
        ):
            print(f"{stem}: already scored", flush=True)
            continue
        model = build()
        model.eval()
        configure_torch_threads(num_workers=0)

        for eval_set, params in evaluation_points(mode):
            if (seed, windows, eval_set) in done:
                continue
            # A stream is queried at the grid point nearest its own
            # distance, which is what a search would do.
            query = min(QUERY_GRID, key=lambda q: abs(q - params["distance_modulus"]))
            transform = QueryDistanceTransform(
                StreamMapTransform(normalizer=WindowNormalizer(), augment=False),
                query_grid=QUERY_GRID,
                step=STEP,
                query=query,
            )
            label_channel = QueryDistanceTransform.channel_index(
                channels, "good", query
            )
            start = time.time()
            with torch.no_grad():
                scored = evaluate_footprint_realizations(
                    model,
                    evaluation_injector,
                    transform,
                    [params],
                    n_realizations=realizations,
                    channel=0,
                    label_channel=label_channel,
                    seed=EVAL_SEED,
                    thresholds=THRESHOLD_GRID,
                    n_null_bands=N_NULL_BANDS,
                )
            frames.append(
                scored.assign(
                    seed=seed,
                    windows=windows,
                    eval_set=eval_set,
                    queried_distance=query,
                )
            )
            pd.concat(frames, ignore_index=True).to_pickle(results_file)
            print(
                f"{stem} {eval_set}: scored in {time.time() - start:.0f}s",
                flush=True,
            )
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--windows", type=int, default=4800)
    parser.add_argument(
        "--mode",
        choices=["grid", "des", "isochrone"],
        default="grid",
        help="which evaluation points to score the trained models on",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="average the seeds' output maps and score that one predictor",
    )
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument(
        "--training-set",
        choices=list(TRAINING_SETS),
        default="des",
        help="which ranges the training streams are drawn from",
    )
    arguments = parser.parse_args()
    main(
        arguments.seeds,
        arguments.windows,
        arguments.mode,
        arguments.ensemble,
        arguments.realizations,
        arguments.training_set,
    )
