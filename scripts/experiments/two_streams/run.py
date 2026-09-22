"""Does a second stream nearby confuse a model trained on one stream per window?

Training injects exactly one stream per window. On real data a window can hold
two (neighbouring or crossing streams), and whether that needs multi-stream
training is the question here -- answered by evaluation alone, no retraining.

Paired design: for each realization, stream A (at the detection edge, SB 33)
is identical in every condition -- same realization, placement and survey
noise -- and only its neighbour B changes: absent, parallel at 1, 2 or 4
degrees, or crossing it. Any change in A's detection is caused by B. B is
either brighter than A (SB 32) or as faint (SB 33).

Models: fixed-decoy 19200-window trainings of the hyperparameter experiment,
on the survey and region they were trained on (LSST year 1, Dec -30).

Run from the repository root:
  python scripts/experiments/two_streams/run.py
"""

import json
import time
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "two_streams"
RESULTS = OUT / "results.pkl"
HYPER = REPO / "scripts" / "experiments" / "hyperparameters"
MODELS = REPO / "data" / "experiments" / "hyperparameters" / "models"
NOTEBOOK = REPO / "notebooks" / "train_model.ipynb"

CONFIGURATION = "w19200_sb32-34.5_d2_b12_lr0.002_bs8_boxdecoy"
SEEDS = [42, 43]
N_REALIZATIONS = 100
N_NULL_BANDS = 200
EVAL_SEED = 2026
INDEPENDENT_BACKGROUND_SEED = 777
# Stream A sits within this many degrees of the study region's centre, so its
# neighbours (up to 4 degrees away, 8 degrees long) stay inside the region.
CENTRE_RADIUS_DEG = 3.0
STREAM = {
    "morphology": "uniform",
    "width": 0.2,
    "length": 8.0,
    "distance_modulus": 16.0,
    "age": 12.5,
    "z": 0.0002,
}
SB_A = 33.0
# name -> (B's surface brightness or None, offset across A's track in degrees,
# B's orientation relative to A in degrees)
CONDITIONS = {"alone": (None, 0.0, 0.0)}
for sb_b in (32.0, 33.0):
    for offset in (1.0, 2.0, 4.0):
        CONDITIONS[f"parallel {offset:g} deg, B at SB {sb_b:g}"] = (sb_b, offset, 0.0)
    CONDITIONS[f"crossing 60 deg, B at SB {sb_b:g}"] = (sb_b, 0.0, 60.0)


def main():
    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", "invalid value encountered in log10")
    import astropy.units as u
    import healpy as hp
    import numpy as np
    import pandas as pd
    import torch
    from astropy.coordinates import SkyCoord

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.evaluation.footprint import THRESHOLD_GRID, score_streams_on_sky

    spec = spec_from_file_location("run", HYPER / "run.py")
    run = module_from_spec(spec)
    spec.loader.exec_module(run)

    cells = {c.get("id"): c for c in json.loads(NOTEBOOK.read_text())["cells"]}

    def run_cell(cell_id, namespace):
        source = "".join(
            line
            for line in cells[cell_id]["source"]
            if not line.lstrip().startswith("%")
        )
        exec(source, namespace)  # noqa: S102 -- trusted notebook in this repository

    settings = json.loads(
        (MODELS / f"{CONFIGURATION}_seed{SEEDS[0]}.json").read_text()
    )["configuration"]
    g = {"__name__": "notebook"}
    for cell_id in run.SETUP_CELLS:
        run_cell(cell_id, g)
    run.pin_experiment_sky(g)
    g["filters_cfg"] = {**g["filters_cfg"], "decoy": run.DECOYS[settings["decoy"]]}
    g["train_cfg"] = {
        **g["train_cfg"],
        "steps_per_epoch": settings["steps_per_epoch"],
        "lr": settings["lr"],
        "batch_size": settings["batch_size"],
    }
    g["stream_param_cfg"] = {
        **g["stream_param_cfg"],
        "richness": run.TRAINING_SB[settings["training_sb"]],
    }
    g["background_cfg"] = {
        **g["background_cfg"],
        "source_kwargs": {"seed": INDEPENDENT_BACKGROUND_SEED},
    }
    run_cell(run.BACKGROUND_CELL, g)
    run_cell(run.INJECTOR_CELL, g)
    injector = g["injector"]
    channel = injector.filter_names.index("good")

    # Where stream A may sit: footprint pixels near the region's centre.
    footprint = np.flatnonzero(injector.background.footprint)
    ra, dec = hp.pix2ang(injector.pix.nside, footprint, lonlat=True)
    region = g["study_region_cfg"]
    centre = SkyCoord(ra=region["center_ra"] * u.deg, dec=region["center_dec"] * u.deg)
    near = centre.separation(SkyCoord(ra=ra * u.deg, dec=dec * u.deg)).deg
    candidates = footprint[near < CENTRE_RADIUS_DEG]

    OUT.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_pickle(RESULTS)] if RESULTS.exists() else []
    done = (
        set(map(tuple, frames[0][["seed", "condition"]].drop_duplicates().to_numpy()))
        if frames
        else set()
    )

    for seed in SEEDS:
        g["SEED"] = seed
        for cell_id in run.TRAIN_CELLS:
            run_cell(cell_id, g)
        saved = json.loads((MODELS / f"{CONFIGURATION}_seed{seed}.json").read_text())
        g["model"].load_state_dict(
            torch.load(MODELS / f"{CONFIGURATION}_seed{seed}.pt")
        )
        g["normalizer"].mean = np.array(saved["normalizer_mean"], dtype=np.float32)
        g["normalizer"].std = np.array(saved["normalizer_std"], dtype=np.float32)
        model = g["model"].eval()
        configure_torch_threads(num_workers=0)

        for condition, (sb_b, offset, relative_angle) in CONDITIONS.items():
            if (seed, condition) in done:
                continue
            start = time.time()
            rows = []
            for realization in range(N_REALIZATIONS):
                # A's position and orientation come from their own generator,
                # the same in every condition, so A is identical throughout.
                where = np.random.default_rng([EVAL_SEED, realization, 99])
                pixel = int(where.choice(candidates))
                a_ra, a_dec = hp.pix2ang(injector.pix.nside, pixel, lonlat=True)
                rotation = float(where.uniform(0.0, 360.0))
                streams = [{**STREAM, "richness": SB_A, "orientation": rotation}]
                centers = [(float(a_ra), float(a_dec))]
                if sb_b is not None:
                    b_centre = SkyCoord(ra=a_ra * u.deg, dec=a_dec * u.deg)
                    if offset:
                        b_centre = b_centre.directional_offset_by(
                            (rotation + 90.0) * u.deg, offset * u.deg
                        )
                    streams.append(
                        {
                            **STREAM,
                            "richness": sb_b,
                            "orientation": (rotation + relative_angle) % 360.0,
                        }
                    )
                    centers.append((float(b_centre.ra.deg), float(b_centre.dec.deg)))
                full_sky = injector.inject_streams_full_sky(
                    streams, np.random.default_rng([EVAL_SEED, realization]), centers
                )
                with torch.no_grad():
                    scored = score_streams_on_sky(
                        model,
                        injector,
                        g["eval_transform"],
                        full_sky,
                        channel,
                        THRESHOLD_GRID,
                        np.random.default_rng([EVAL_SEED, realization, 1]),
                        n_null_bands=N_NULL_BANDS,
                    )
                for row in scored:
                    row.update(
                        seed=seed,
                        condition=condition,
                        realization=realization,
                        which="A" if row["stream"] == 0 else "B",
                        richness=streams[row["stream"]]["richness"],
                    )
                    rows.append(row)
            frames.append(pd.DataFrame(rows))
            pd.concat(frames, ignore_index=True).to_pickle(RESULTS)
            print(f"seed {seed}, {condition}: {time.time() - start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
