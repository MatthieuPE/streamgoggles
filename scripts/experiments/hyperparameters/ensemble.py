"""Does averaging several trainings beat a single one?

Training-to-training variation dominates this experiment: the same
configuration detects between 15% and 90% of SB 33.5 streams depending on the
seed. A survey search deploys one model, so this scores an *ensemble*: the
per-pixel probabilities of several trained models, averaged, thresholded once.

Every model of a configuration is loaded from data/experiments/hyperparameters/
models/ (no retraining) and scored on the same streams as every other result
(same EVAL_SEED, surface brightnesses and realizations as run.py).

Each model was trained with its own input normalization, fitted on its own
first training windows. The ensemble feeds every model the input in *its* own
normalization, converting from the shared one it is given:
    x_i = (x_shared * std_shared + mean_shared - mean_i) / std_i

Which trainings are averaged is a named member set. "nested" averages the
first N seeds for N = 1, 2, 3, 4, 6 -- the original design, where "how many"
and "which" are confounded. "halves" averages two disjoint triples and all six,
so the gain from averaging can be told apart from the luck of which trainings
went in.

Results go to one file per configuration and number of streams, so runs never
overwrite each other or mix sample sizes; the original 4800-window run at 500
streams keeps its name, ensemble_results.pkl.

Run from the repository root:
  python scripts/experiments/hyperparameters/ensemble.py [configuration]
      [--sets nested|halves|singles] [--streams N]
"""

import argparse
import json
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "experiments" / "hyperparameters"
MODELS = DATA / "models"
RESULTS = DATA / "ensemble_results.pkl"
NOTEBOOK = REPO / "notebooks" / "train_model.ipynb"

DEFAULT_CONFIGURATION = "w4800_sb32-34.5_d2_b12_lr0.002_bs8"
SEEDS = [42, 43, 44, 45, 46, 47]
MEMBER_SETS = {
    # 4 is the equal-cost comparison against one training on 4 x 4800 windows.
    "nested": [tuple(SEEDS[:n]) for n in (1, 2, 3, 4, 6)],
    "halves": [tuple(SEEDS[:3]), tuple(SEEDS[3:]), tuple(SEEDS)],
    # Each training on its own, on as many streams as the averages: a model meant
    # for deployment is a single prediction, so its absolute rate needs a few
    # hundred streams per point, not the 20 shared streams of run.py.
    "singles": [(seed,) for seed in SEEDS],
}
EVAL_SB = [32.0, 33.0, 33.5, 34.0, 34.5, 35.0]
# Streams injected PER surface brightness, so this many times the 6 entries of
# EVAL_SB. An ensemble is one prediction with no seeds to pool over, unlike a
# configuration curve (6 trainings x 20 = 120 per point), so this number alone
# sets its sampling interval: +-11 points at 20, +-5 at 100, +-2 at 500. The
# streams are simulated on demand and never seen in training, so the only limit
# is time: about 0.45 s per stream for one model, 0.85 s for six averaged.
N_REALIZATIONS = 500
N_NULL_BANDS = 200
EVAL_SEED = 2026
INDEPENDENT_BACKGROUND_SEED = 777
SETUP_CELLS = ["95eb95f8", "ac5b36bc"]
BACKGROUND_CELL, INJECTOR_CELL = "06d604ea", "d4da0c1e"
TRAIN_CELLS = ["a680f20b", "b2785c24", "d0449892", "c1aae991"]


def results_path(configuration, n_streams):
    """One file per configuration and sample size."""
    if configuration == DEFAULT_CONFIGURATION and n_streams == N_REALIZATIONS:
        return RESULTS
    return DATA / f"ensemble_results_{configuration}_{n_streams}streams.pkl"


def members_label(seeds):
    return ",".join(str(seed) for seed in seeds)


def main(configuration, sets="nested", n_streams=N_REALIZATIONS):
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.evaluation.footprint import (
        THRESHOLD_GRID,
        evaluate_footprint_realizations,
        stream_detection,
    )

    class Ensemble(nn.Module):
        """Average the per-pixel probabilities of several trained models."""

        def __init__(self, models, means, stds, shared_mean, shared_std):
            super().__init__()
            self.models = nn.ModuleList(models)
            shape = (1, -1, 1, 1)
            self.register_buffer(
                "shared_mean", torch.tensor(shared_mean).reshape(shape)
            )
            self.register_buffer("shared_std", torch.tensor(shared_std).reshape(shape))
            self.means = [torch.tensor(m).reshape(shape) for m in means]
            self.stds = [torch.tensor(s).reshape(shape) for s in stds]

        def forward(self, x):
            raw = x * self.shared_std + self.shared_mean
            outputs = [
                model((raw - mean) / std)
                for model, mean, std in zip(
                    self.models, self.means, self.stds, strict=True
                )
            ]
            return torch.stack(outputs).mean(dim=0)

    cells = {c.get("id"): c for c in json.loads(NOTEBOOK.read_text())["cells"]}

    def run_cell(cell_id, namespace):
        source = "".join(
            line
            for line in cells[cell_id]["source"]
            if not line.lstrip().startswith("%")
        )
        exec(source, namespace)  # noqa: S102 -- trusted notebook in this repository

    settings = json.loads(
        (MODELS / f"{configuration}_seed{SEEDS[0]}.json").read_text()
    )["configuration"]
    g = {"__name__": "notebook"}
    for cell_id in SETUP_CELLS:
        run_cell(cell_id, g)
    g["train_cfg"] = {
        **g["train_cfg"],
        "steps_per_epoch": settings["steps_per_epoch"],
        "lr": settings["lr"],
        "batch_size": settings["batch_size"],
    }
    g["stream_param_cfg"] = {**g["stream_param_cfg"], "richness": None}
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("run", Path(__file__).parent / "run.py")
    run = module_from_spec(spec)
    spec.loader.exec_module(run)
    run.pin_experiment_sky(g)
    # Rebuild the decoy channel these models were trained with, from their own
    # saved configuration: feeding a model a decoy it never saw would score
    # noise. Models saved before the decoy became a setting used the shifted
    # box (PLAN.md 6.32).
    g["filters_cfg"] = {
        **g["filters_cfg"],
        "decoy": run.DECOYS[settings.get("decoy", "shifted")],
    }
    g["stream_param_cfg"]["richness"] = run.TRAINING_SB[settings["training_sb"]]
    g["model_cfg"] = {
        **g["model_cfg"],
        "depth": settings["depth"],
        "base_width": settings["base_width"],
    }
    g["background_cfg"] = {
        **g["background_cfg"],
        "source_kwargs": {"seed": INDEPENDENT_BACKGROUND_SEED},
    }
    run_cell(BACKGROUND_CELL, g)
    run_cell(INJECTOR_CELL, g)
    configure_torch_threads(num_workers=0)

    # One model per seed, each with its own normalization.
    models, means, stds = [], [], []
    for seed in SEEDS:
        g["SEED"] = seed
        for cell_id in TRAIN_CELLS:
            run_cell(cell_id, g)
        saved = json.loads((MODELS / f"{configuration}_seed{seed}.json").read_text())
        g["model"].load_state_dict(
            torch.load(MODELS / f"{configuration}_seed{seed}.pt")
        )
        g["model"].eval()
        models.append(g["model"])
        means.append(np.array(saved["normalizer_mean"], dtype=np.float32))
        stds.append(np.array(saved["normalizer_std"], dtype=np.float32))
        g.pop("model")

    # The transform handed to the evaluation uses the first model's normalization.
    g["normalizer"].mean, g["normalizer"].std = means[0], stds[0]
    stream = g["stream_param_cfg"]
    base = {
        k: stream[k]
        for k in ("morphology", "width", "length", "distance_modulus", "age", "z")
    }
    param_sets = [dict(base, richness=sb) for sb in EVAL_SB]
    channel = g["injector"].filter_names.index("good")

    # Resumable: a member set already scored with this many streams for every
    # surface brightness is kept and skipped, so an interrupted run (a sleeping
    # laptop, a closed session) loses at most the set in progress. The file is
    # specific to this configuration and sample size, so it never mixes them,
    # and sets scored by an earlier run with other --sets are kept too.
    results_file = results_path(configuration, n_streams)
    frames = []
    if results_file.exists():
        previous = pd.read_pickle(results_file)
        previous = previous[previous.configuration == configuration]
        if "members" not in previous:
            # Written before member sets existed: always the first N seeds.
            previous = previous.assign(
                members=previous["ensemble_size"].map(
                    lambda n: members_label(SEEDS[: int(n)])
                )
            )
        for _, group in previous.groupby("members"):
            counts = group.groupby("richness").size()
            if set(counts.index) == set(EVAL_SB) and (counts == n_streams).all():
                frames.append(group)
    done = {frame["members"].iloc[0] for frame in frames}
    if done:
        print(f"already scored with {n_streams} streams per point: {sorted(done)}")

    for seeds in MEMBER_SETS[sets]:
        label = members_label(seeds)
        if label in done:
            continue
        members = [SEEDS.index(seed) for seed in seeds]
        ensemble = Ensemble(
            [models[i] for i in members],
            [means[i] for i in members],
            [stds[i] for i in members],
            means[0],
            stds[0],
        )
        ensemble.eval()
        start = time.time()
        with torch.no_grad():
            scored = evaluate_footprint_realizations(
                ensemble,
                g["injector"],
                g["eval_transform"],
                param_sets,
                n_realizations=n_streams,
                channel=channel,
                seed=EVAL_SEED,
                thresholds=THRESHOLD_GRID,
                n_null_bands=N_NULL_BANDS,
            )
        frames.append(
            scored.assign(
                configuration=configuration, ensemble_size=len(seeds), members=label
            )
        )
        pd.concat(frames, ignore_index=True).to_pickle(results_file)
        print(f"ensemble of {label}: scored in {time.time() - start:.0f}s", flush=True)

    results = pd.concat(frames, ignore_index=True)
    detection = stream_detection(
        results, THRESHOLD_GRID, at=(0.1, 0.5), group_by=["members", "richness"]
    )
    pd.set_option("display.width", 220)
    print(f"\n== {configuration}: streams detected, ensemble of N trainings")
    print(
        detection.pivot_table(
            index=["members", "threshold"],
            columns="richness",
            values="detection_fraction",
        ).to_string(float_format=lambda v: f"{v:.2f}")
    )
    print("\n== flagged pixels within 1 sigma / background density, threshold 0.5")
    at_half = detection[np.isclose(detection.threshold, 0.5)]
    print(
        at_half.pivot_table(
            index="members",
            columns="richness",
            values=["flagged_in_band", "background_density"],
        ).to_string(float_format=lambda v: f"{v:.3g}")
    )
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configuration", nargs="?", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--sets", choices=sorted(MEMBER_SETS), default="nested")
    parser.add_argument("--streams", type=int, default=N_REALIZATIONS)
    arguments = parser.parse_args()
    main(arguments.configuration, arguments.sets, arguments.streams)
