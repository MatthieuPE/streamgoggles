"""Train and score the hyperparameter experiment's configurations.

Documented in docs/source/experiments/hyperparameters.md.

Every run rebuilds `notebooks/train_model.ipynb`'s pipeline by executing its
cells (addressed by cell id), with the notebook's selected loss (batch Dice,
batch 8, background fraction 0.05) and magnitude cut, overriding only what a
configuration sets: training windows per epoch, training surface
brightnesses, network depth and width, learning rate, batch size.

Each model is scored with `evaluate_footprint_realizations` on the same streams
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
    "decoy": "shifted",
}

# The survey and study region every model of this experiment was trained and
# scored on. The notebooks have since moved to DES year 6 at Dec -50 (PLAN.md
# 6.38); pinning these keeps re-scoring and resuming consistent with the saved
# models. A new experiment on DES should be a new script, not an edit of this.
EXPERIMENT_SURVEY = {"survey": "lsst", "release": "yr1"}
EXPERIMENT_REGION = {"center_ra": 0.0, "center_dec": -30.0}


def pin_experiment_sky(namespace):
    """Point the notebook's configuration back at this experiment's survey and region."""
    namespace["background_cfg"] = {**namespace["background_cfg"], **EXPERIMENT_SURVEY}
    namespace["namespace"] = (
        f"{EXPERIMENT_SURVEY['survey']}_{EXPERIMENT_SURVEY['release']}"
    )
    namespace["study_region_cfg"] = {
        **namespace["study_region_cfg"],
        **EXPERIMENT_REGION,
    }


# The decoy input channel, written out here rather than read from the notebook
# so that editing the notebook can never silently change what an experiment
# trained on. "shifted" is what every model before the fixed_decoy phase used
# (PLAN.md 6.32); "box" is the fixed colour-magnitude box the notebooks use now.
DECOYS = {
    "shifted": {
        "type": "shifted_box",
        "reference": "good",
        "color_shift": 0.5,
        "color_width": 0.3,
    },
    "box": {"type": "box", "color_range": (1.2, 1.5), "mag_range": (18.0, 24.5)},
}


def configuration(**overrides):
    config = {**DEFAULT, **overrides}
    config["name"] = (
        f"w{40 * config['steps_per_epoch']}_{config['training_sb']}_d{config['depth']}"
        f"_b{config['base_width']}_lr{config['lr']:g}_bs{config['batch_size']}"
    )
    # The shifted decoy keeps the original names, so every model and result
    # trained before the decoy switch stays addressable as it was.
    if config["decoy"] != "shifted":
        config["name"] += f"_{config['decoy']}decoy"
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
# The candidates phase 2 leaves in contention, confirmed with 4 seeds as the
# protocol requires (seeds 42/43 are already trained by phase 2 and skipped).
PHASES["phase2_confirm"] = [
    configuration(
        steps_per_epoch=120, training_sb="sb32-34.5", depth=depth, base_width=width
    )
    for depth, width in ((3, 12), (4, 12), (4, 24))
]
# Depth at width 12, with enough seeds to separate the configurations: with 4
# seeds the candidates were indistinguishable (85/38/17 vs 86/36/10 vs 80/33/9
# at SB 33/33.5/34), because training-to-training variation is larger than the
# differences (one depth-3 seed reached 75% at SB 33.5, another 15%). Six seeds
# give ~120 streams per surface brightness, about +-4 points.
PHASES["depth"] = [
    configuration(
        steps_per_epoch=120, training_sb="sb32-34.5", depth=depth, base_width=12
    )
    for depth in (2, 3, 4)
]
# Phase 1's comparisons, re-measured with 6 seeds: with 2 seeds the same
# configuration measured 88/17/0 at SB 33/33.5/34 and with 6 seeds 92/48/20,
# so the phase-1 differences were mostly the seed lottery.
PHASES["training_data"] = [
    configuration(steps_per_epoch=steps, training_sb=sb)
    for sb in TRAINING_SB
    for steps in (30, 120)
]
# How far does more training data go? The ladder 1200 -> 19200 windows on the
# faint range, with several seeds each, measures both the average detection and
# the spread between trainings. 19200 windows is also four times 4800, so a
# single long training can be compared with an ensemble of four short ones at
# equal total training cost (scripts/experiments/hyperparameters/ensemble.py).
PHASES["training_length"] = [
    configuration(steps_per_epoch=steps, training_sb="sb32-34.5")
    for steps in (240, 480)
]
# Does the selected configuration survive the switch to the fixed decoy box?
# Every conclusion above was reached with the shifted box; the first model
# trained with the fixed one (train_model.ipynb) came out about 20 points low at
# SB 33.5 (PLAN.md 6.35). 4800 windows sits next to 19200 so the same run also
# says what the four-times-longer training buys with the new inputs -- training
# speed matters for everything that comes after.
PHASES["fixed_decoy"] = [
    configuration(steps_per_epoch=steps, training_sb="sb32-34.5", decoy="box")
    for steps in (120, 480)
]
SEEDS = {
    "fixed_decoy": [42, 43, 44, 45, 46, 47],
    "phase1": [42, 43],
    "training_data": [42, 43, 44, 45, 46, 47],
    "training_length": [42, 43, 44, 45, 46, 47],
    "phase2": [42, 43],
    "phase2_confirm": [42, 43, 44, 45],
    "depth": [42, 43, 44, 45, 46, 47],
}

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
    pin_experiment_sky(g)
    # The decoy channel is part of the configuration (see DECOYS). The
    # background and injectors are built once per run, so a phase trains one
    # decoy only.
    decoys = {config["decoy"] for config in PHASES[phase]}
    if len(decoys) != 1:
        raise ValueError(f"phase {phase!r} mixes decoys {sorted(decoys)}")
    g["filters_cfg"] = {**g["filters_cfg"], "decoy": DECOYS[decoys.pop()]}
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
