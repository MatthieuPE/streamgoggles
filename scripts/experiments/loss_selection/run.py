"""Train and score every loss configuration of the loss-selection experiment.

Documented in docs/source/experiments/loss_selection.md.

Each run rebuilds `notebooks/train_model.ipynb`'s pipeline (background,
matched filters, injector, datasets, normalizer, U-Net, trainer) by executing
its cells, overriding only the loss, background_fraction, batch size and
seed. Every trained model is scored with
`evaluate_footprint_realizations` on the same skies (SB 31-35, 10
realizations each, EVAL_SEED) on two backgrounds: the training catalog and
an independently seeded one. Per-threshold counts (THRESHOLD_GRID) are
recorded, so any threshold can be analysed afterwards without retraining.

Outputs (git-ignored), resumable: a configuration/seed already in
results.pkl is skipped, a saved model is re-scored without retraining.
  data/experiments/loss_selection/results.pkl
  data/experiments/loss_selection/models/<config>_seed<seed>.{pt,json}

Run from the repository root with the `streamml` environment:
  python scripts/experiments/loss_selection/run.py
About 2 minutes per model on a laptop CPU (36 models).
"""

import gc
import json
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "loss_selection"
MODELS = OUT / "models"
RESULTS = OUT / "results.pkl"
NOTEBOOK = REPO / "notebooks" / "train_model.ipynb"

# (loss, background_fraction, batch_size). Same number of training samples
# (epochs * steps_per_epoch) for every configuration: batch 8 therefore means
# 4x fewer optimizer steps than batch 2 at the same learning rate.
CONFIGS = [
    ("dice", 0.05, 2),
    ("dice", 0.3, 2),
    ("dice_bce", 0.05, 2),
    ("dice_bce", 0.3, 2),
    ("batch_dice", 0.05, 2),
    ("batch_dice", 0.3, 2),
    ("batch_dice", 0.3, 8),
    ("dice", 0.3, 8),
    ("batch_dice", 0.05, 8),
]
SEEDS = [42, 43, 44, 45]
SB_GRID = [31.0, 32.0, 33.0, 34.0, 35.0]
N_REALIZATIONS = 10
EVAL_SEED = 2026
INDEPENDENT_BACKGROUND_SEED = 777

# Notebook cells used, by id (robust to cells being inserted or moved).
SETUP_CELLS = [
    "95eb95f8",
    "ac5b36bc",
    "06d604ea",
    "d4da0c1e",
]  # imports, config, background, injector
TRAIN_CELLS = [
    "a680f20b",
    "b2785c24",
    "d0449892",
    "c1aae991",
]  # datasets, normalizer, bias init, model
FIT_CELL = "3e5a150a"  # DataLoaders + PlainTrainer.train
BACKGROUND_CELL, INJECTOR_CELL = "06d604ea", "d4da0c1e"


def main():
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.evaluation.footprint import (
        THRESHOLD_GRID,
        evaluate_footprint_realizations,
    )
    from streamgoggles.models.losses import (
        DiceLoss,
        MaskedLoss,
        _broadcast_mask,
        _masked_mean,
    )
    from streamgoggles.models.losses import get_loss as library_get_loss

    class DiceBCELoss(MaskedLoss):
        """Dice + BCE on probabilities, weight 1:1 (tested here, not in the library)."""

        def forward(self, pred, target, valid_mask=None):
            mask = _broadcast_mask(pred, valid_mask)
            p = pred.clamp(1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(p, target, reduction="none")
            return DiceLoss()(pred, target, valid_mask) + _masked_mean(bce, mask)

    def get_loss(name, **kwargs):
        return DiceBCELoss() if name == "dice_bce" else library_get_loss(name, **kwargs)

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
    g["get_loss"] = get_loss

    independent = dict(g)
    independent["background_cfg"] = {
        **g["background_cfg"],
        "source_kwargs": {"seed": INDEPENDENT_BACKGROUND_SEED},
    }
    run_cell(BACKGROUND_CELL, independent)
    run_cell(INJECTOR_CELL, independent)
    injectors = {
        "training bg": g["injector"],
        "independent bg": independent["injector"],
    }

    stream = g["stream_param_cfg"]
    base = {
        k: stream[k]
        for k in ("morphology", "width", "length", "distance_modulus", "age", "z")
    }
    param_sets = [dict(base, richness=sb) for sb in SB_GRID]

    frames = [pd.read_pickle(RESULTS)] if RESULTS.exists() else []
    done = set()
    if frames:
        columns = ["loss", "background_fraction", "batch_size", "seed"]
        done = {
            tuple(r)
            for r in frames[0][columns].drop_duplicates().itertuples(index=False)
        }

    # Seed-major order: an interrupted run leaves every configuration at the
    # same number of seeds.
    for seed in SEEDS:
        for loss, background_fraction, batch_size in CONFIGS:
            if (loss, background_fraction, batch_size, seed) in done:
                continue
            name = f"{loss}_bf{background_fraction}_bs{batch_size}_seed{seed}"
            g["SEED"] = seed
            g["train_cfg"] = {
                **g["train_cfg"],
                "loss_name": loss,
                "background_fraction": background_fraction,
                "batch_size": batch_size,
            }
            start = time.time()
            weights, meta = MODELS / f"{name}.pt", MODELS / f"{name}.json"
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
                            "normalizer_mean": [float(v) for v in g["normalizer"].mean],
                            "normalizer_std": [float(v) for v in g["normalizer"].std],
                            "train_losses": g["result"]["train_losses"],
                            "val_losses": g["result"]["val_losses"],
                            "final_val_loss": final_val,
                            "train_s": train_s,
                            "train_cfg": g["train_cfg"],
                            "seed": seed,
                        }
                    )
                )
            model = g["model"]
            model.eval()
            good = g["injector"].filter_names.index("good")
            for background, injector in injectors.items():
                scored = evaluate_footprint_realizations(
                    model,
                    injector,
                    g["eval_transform"],
                    param_sets,
                    n_realizations=N_REALIZATIONS,
                    channel=good,
                    seed=EVAL_SEED,
                    thresholds=THRESHOLD_GRID,
                )
                frames.append(
                    scored.assign(
                        loss=loss,
                        background_fraction=background_fraction,
                        batch_size=batch_size,
                        seed=seed,
                        background=background,
                        train_s=train_s,
                        final_val_loss=final_val,
                    )
                )
                pd.concat(frames, ignore_index=True).to_pickle(RESULTS)
                print(f"{name} {background}: trained in {train_s:.0f}s", flush=True)
            for key in ("train_dl", "val_dl", "trainer", "model"):
                g.pop(key, None)
            gc.collect()
    print("all runs done", flush=True)


if __name__ == "__main__":
    main()
