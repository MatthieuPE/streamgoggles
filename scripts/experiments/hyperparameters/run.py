"""Train and score the hyperparameter experiment's configurations.

Documented in docs/source/experiments/hyperparameters.md.

Every run rebuilds `notebooks/train_model.ipynb`'s pipeline by executing its
cells (addressed by cell id), with the notebook's selected loss (batch Dice,
batch 8, background fraction 0.05) and magnitude cut, overriding only what a
configuration sets: training windows per epoch, training surface
brightnesses, network depth and width, learning rate, batch size.

Each model is scored with `evaluate_footprint_realizations` on the same skies
for every model: surface brightness EVAL_SB, N_REALIZATIONS each, on an
independently seeded background, with the no-stream control, per-threshold
counts and track-band statistics, so `stream_detection` (per-stream SNR along
the track) and `detection_metrics` (completeness / contamination / contrast)
can be computed at any threshold afterwards.

Outputs (git-ignored), resumable: a (configuration, seed) already in
results.pkl is skipped; a saved model is re-scored without retraining.
  data/experiments/hyperparameters/results.pkl
  data/experiments/hyperparameters/models/<configuration>_seed<seed>.{pt,json}

Run from the repository root with the `streamml` environment:
  python scripts/experiments/hyperparameters/run.py phase1
"""

import copy
import gc
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "hyperparameters"
MODELS = OUT / "models"
RESULTS = OUT / "results.pkl"
NOTEBOOK = REPO / "notebooks" / "train_model.ipynb"

TRAINING_SB = {
    "sb31-34": [31.0, 32.0, 33.0, 34.0],  # the notebook's current range
    "sb32-34.5": [32.0, 33.0, 33.5, 34.0, 34.5],  # weighted towards the faint end
}

# Every configuration starts from these; a phase overrides some of them.
DEFAULT = {
    "steps_per_epoch": 30,  # training windows per epoch (40 epochs)
    "training_sb": "sb31-34",
    "depth": 2,
    "base_width": 12,
    "lr": 2e-3,
    "batch_size": 8,
}


def configuration(**overrides):
    config = {**DEFAULT, **overrides}
    config["name"] = (
        f"w{40 * config['steps_per_epoch']}_{config['training_sb']}_d{config['depth']}"
        f"_b{config['base_width']}_lr{config['lr']:g}_bs{config['batch_size']}"
    )
    return config


PHASES = {
    # Phase 1: training length x training surface brightness range.
    "phase1": [
        configuration(steps_per_epoch=steps, training_sb=sb)
        for sb in TRAINING_SB
        for steps in (30, 120, 240)
    ],
}
# Phase 2: network depth x base width, from phase 1's cleanest setting that
# still gains on the faint training range (4800 windows, SB 32-34.5): phase 1
# found no configuration with both a clean background and SB 34 detections,
# and a deeper network sees more of a faint track at once. The depth-2,
# width-12 configuration is phase 1's own and is reused, not retrained.
PHASES["phase2"] = [
    configuration(
        steps_per_epoch=120, training_sb="sb32-34.5", depth=depth, base_width=width
    )
    for depth in (2, 3, 4)
    for width in (12, 24)
]
SEEDS = {"phase1": [42, 43], "phase2": [42, 43]}

EVAL_SB = [32.0, 33.0, 33.5, 34.0, 34.5, 35.0]
N_REALIZATIONS = 20
N_NULL_BANDS = 200
EVAL_SEED = 2026
INDEPENDENT_BACKGROUND_SEED = 777

SETUP_CELLS = ["95eb95f8", "ac5b36bc"]  # imports, configuration
BACKGROUND_CELL, INJECTOR_CELL = "06d604ea", "d4da0c1e"
TRAIN_CELLS = [
    "a680f20b",
    "b2785c24",
    "d0449892",
    "c1aae991",
]  # datasets, normalizer, bias, model
FIT_CELL = "3e5a150a"  # DataLoaders + PlainTrainer.train


def main(phase):
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import numpy as np
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.evaluation.footprint import (
        THRESHOLD_GRID,
        evaluate_footprint_realizations,
    )

    cells = {c.get("id"): c for c in json.loads(NOTEBOOK.read_text())["cells"]}

    def run_cell(cell_id, namespace):
        source = "".join(
            line
            for line in cells[cell_id]["source"]
            if not line.lstrip().startswith("%")
        )
        exec(source, namespace)  # noqa: S102 -- trusted notebook in this repository

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    g = {"__name__": "notebook"}
    for cell_id in SETUP_CELLS:
        run_cell(cell_id, g)
    base_train_cfg = copy.deepcopy(g["train_cfg"])
    base_stream_cfg = copy.deepcopy(g["stream_param_cfg"])
    base_model_cfg = copy.deepcopy(g["model_cfg"])
    run_cell(BACKGROUND_CELL, g)
    run_cell(INJECTOR_CELL, g)

    independent = dict(g)
    independent["background_cfg"] = {
        **g["background_cfg"],
        "source_kwargs": {"seed": INDEPENDENT_BACKGROUND_SEED},
    }
    run_cell(BACKGROUND_CELL, independent)
    run_cell(INJECTOR_CELL, independent)
    evaluation_injector = independent["injector"]

    stream = base_stream_cfg
    base = {
        k: stream[k]
        for k in ("morphology", "width", "length", "distance_modulus", "age", "z")
    }
    param_sets = [dict(base, richness=sb) for sb in EVAL_SB]

    frames = [pd.read_pickle(RESULTS)] if RESULTS.exists() else []
    done = set()
    if frames:
        done = set(
            map(
                tuple, frames[0][["configuration", "seed"]].drop_duplicates().to_numpy()
            )
        )

    # Seed-major: an interrupted phase leaves every configuration at the same
    # number of seeds.
    for seed in SEEDS[phase]:
        for config in PHASES[phase]:
            if (config["name"], seed) in done:
                continue
            g["SEED"] = seed
            g["train_cfg"] = {
                **base_train_cfg,
                "steps_per_epoch": config["steps_per_epoch"],
                "lr": config["lr"],
                "batch_size": config["batch_size"],
            }
            g["stream_param_cfg"] = {
                **base_stream_cfg,
                "richness": TRAINING_SB[config["training_sb"]],
            }
            g["model_cfg"] = {
                **base_model_cfg,
                "depth": config["depth"],
                "base_width": config["base_width"],
            }
            stem = f"{config['name']}_seed{seed}"
            weights, meta = MODELS / f"{stem}.pt", MODELS / f"{stem}.json"
            start = time.time()
            for cell_id in TRAIN_CELLS:
                run_cell(cell_id, g)
            if weights.exists():
                saved = json.loads(meta.read_text())
                g["normalizer"].mean = np.array(saved["normalizer_mean"])
                g["normalizer"].std = np.array(saved["normalizer_std"])
                g["model"].load_state_dict(torch.load(weights))
                configure_torch_threads(num_workers=0)
                train_s, final_val = saved["train_s"], saved["final_val_loss"]
            else:
                run_cell(FIT_CELL, g)
                train_s = time.time() - start
                final_val = g["result"]["val_losses"][-1]
                torch.save(g["model"].state_dict(), weights)
                meta.write_text(
                    json.dumps(
                        {
                            "configuration": config,
                            "seed": seed,
                            "normalizer_mean": [float(v) for v in g["normalizer"].mean],
                            "normalizer_std": [float(v) for v in g["normalizer"].std],
                            "train_losses": g["result"]["train_losses"],
                            "val_losses": g["result"]["val_losses"],
                            "final_val_loss": final_val,
                            "train_s": train_s,
                            "n_parameters": sum(
                                p.numel() for p in g["model"].parameters()
                            ),
                        }
                    )
                )
            model = g["model"]
            model.eval()
            start = time.time()
            scored = evaluate_footprint_realizations(
                model,
                evaluation_injector,
                g["eval_transform"],
                param_sets,
                n_realizations=N_REALIZATIONS,
                channel=evaluation_injector.filter_names.index("good"),
                seed=EVAL_SEED,
                thresholds=THRESHOLD_GRID,
                n_null_bands=N_NULL_BANDS,
            )
            frames.append(
                scored.assign(
                    configuration=config["name"],
                    phase=phase,
                    seed=seed,
                    train_s=train_s,
                    final_val_loss=final_val,
                    **{f"cfg_{k}": v for k, v in config.items() if k != "name"},
                )
            )
            pd.concat(frames, ignore_index=True).to_pickle(RESULTS)
            print(
                f"{stem}: trained in {train_s:.0f}s, scored in {time.time() - start:.0f}s",
                flush=True,
            )
            for key in ("train_dl", "val_dl", "trainer", "model"):
                g.pop(key, None)
            gc.collect()
    print(f"{phase} done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1")
