"""Does averaging several trainings beat a single one?

Training-to-training variation dominates this experiment: the same
configuration detects between 15% and 90% of SB 33.5 streams depending on the
seed. A survey search deploys one model, so this scores an *ensemble*: the
per-pixel probabilities of several trained models, averaged, thresholded once.

Every model of a configuration is loaded from data/experiments/hyperparameters/
models/ (no retraining) and scored on the same skies as every other result
(same EVAL_SEED, surface brightnesses and realizations as run.py).

Each model was trained with its own input normalization, fitted on its own
first training windows. The ensemble feeds every model the input in *its* own
normalization, converting from the shared one it is given:
    x_i = (x_shared * std_shared + mean_shared - mean_i) / std_i

Run from the repository root:
  python scripts/experiments/hyperparameters/ensemble.py [configuration_name]
"""

import json
import sys
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
# 4 is the equal-cost comparison against one training on 4 x 4800 windows.
ENSEMBLE_SIZES = [1, 2, 3, 4, 6]
EVAL_SB = [32.0, 33.0, 33.5, 34.0, 34.5, 35.0]
N_REALIZATIONS = 20
N_NULL_BANDS = 200
EVAL_SEED = 2026
INDEPENDENT_BACKGROUND_SEED = 777
SETUP_CELLS = ["95eb95f8", "ac5b36bc"]
BACKGROUND_CELL, INJECTOR_CELL = "06d604ea", "d4da0c1e"
TRAIN_CELLS = ["a680f20b", "b2785c24", "d0449892", "c1aae991"]


def main(configuration):
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

    frames = []
    for size in ENSEMBLE_SIZES:
        members = list(range(size))
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
                n_realizations=N_REALIZATIONS,
                channel=channel,
                seed=EVAL_SEED,
                thresholds=THRESHOLD_GRID,
                n_null_bands=N_NULL_BANDS,
            )
        frames.append(scored.assign(configuration=configuration, ensemble_size=size))
        pd.concat(frames, ignore_index=True).to_pickle(RESULTS)
        print(f"ensemble of {size}: scored in {time.time() - start:.0f}s", flush=True)

    results = pd.concat(frames, ignore_index=True)
    detection = stream_detection(
        results, THRESHOLD_GRID, at=(0.1, 0.5), group_by=["ensemble_size", "richness"]
    )
    pd.set_option("display.width", 220)
    print(f"\n== {configuration}: streams detected, ensemble of N trainings")
    print(
        detection.pivot_table(
            index=["ensemble_size", "threshold"],
            columns="richness",
            values="detection_fraction",
        ).to_string(float_format=lambda v: f"{v:.2f}")
    )
    print("\n== flagged pixels within 1 sigma / background density, threshold 0.5")
    at_half = detection[np.isclose(detection.threshold, 0.5)]
    print(
        at_half.pivot_table(
            index="ensemble_size",
            columns="richness",
            values=["flagged_in_band", "background_density"],
        ).to_string(float_format=lambda v: f"{v:.3g}")
    )
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIGURATION)
