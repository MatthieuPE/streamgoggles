"""Figures and numbers for docs/source/experiments/stream_parameters.md.

  detection.png   fraction of streams detected against distance modulus, one
                  panel per surface brightness, one colour per width, quick
                  (4800-window) and long (19200-window) models
  training_spread.png  one point per trained model, at surface brightness 34:
                  how much the same configuration varies from training to
                  training
  contrast        with --contrast: the stream's selected-star density against
                  the background's in the matched filter, per distance, at
                  fixed surface brightness and angular size (builds the sky,
                  about a minute)

Run from the repository root:
  python scripts/experiments/stream_parameters/figures.py [--contrast]
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streamgoggles.evaluation.footprint import (
    THRESHOLD_GRID,
    _wilson_interval,
    detection_at_false_alarm_rate,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "experiments" / "stream_parameters"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "stream_parameters"
TARGET = 1e-3
WIDTH_COLOURS = {0.2: "#08519c", 0.6: "#e6550d", 1.2: "#31a354"}
STYLES = {4800: ("-", "o", "4800 w"), 19200: ("--", "s", "19200 w")}


def pooled_detection(results):
    """Detected / injected per (windows, distance, width, SB), each trained
    model thresholded at the first threshold where it flags at most TARGET of
    the stream-free sky, then pooled over its seeds."""
    matched = detection_at_false_alarm_rate(
        results,
        THRESHOLD_GRID,
        targets=(TARGET,),
        group_by=["windows", "seed", "distance_modulus", "width", "richness"],
    )
    pooled = matched.groupby(["windows", "distance_modulus", "width", "richness"])[
        ["n_detected", "n_realizations"]
    ].sum()
    pooled["fraction"] = pooled["n_detected"] / pooled["n_realizations"]
    return pooled.reset_index(), matched


def figure_detection(pooled):
    surface_brightnesses = [33.0, 34.0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, sb in zip(axes, surface_brightnesses, strict=True):
        for (windows, width), group in pooled[pooled.richness == sb].groupby(
            ["windows", "width"]
        ):
            ls, marker, text = STYLES[int(windows)]
            group = group.sort_values("distance_modulus")
            k = group["n_detected"].to_numpy()
            n = group["n_realizations"].to_numpy()
            low, high = np.array(
                [_wilson_interval(int(a), int(b)) for a, b in zip(k, n, strict=True)]
            ).T
            y = k / n
            shift = 0.04 if windows == 19200 else -0.04
            ax.errorbar(
                group["distance_modulus"] + shift,
                y,
                yerr=[np.clip(y - low, 0, None), np.clip(high - y, 0, None)],
                color=WIDTH_COLOURS[width],
                ls=ls,
                marker=marker,
                capsize=2.5,
                lw=1.6,
                label=f"width {width:g} deg, {text}",
            )
        ax.set_title(f"SB {sb:g}", fontsize=11)
        ax.set_xlabel("distance modulus")
        ax.set_xticks([15, 16, 17, 18, 19])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of streams detected")
    axes[1].legend(fontsize=8, handlelength=3.2, loc="lower right")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "detection.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_training_spread(matched):
    """One point per trained model, SB 34: the spread between trainings."""
    quick = matched[(matched.windows == 4800) & (matched.richness == 34.0)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for width, group in quick.groupby("width"):
        offset = {0.2: -0.12, 0.6: 0.0, 1.2: 0.12}[width]
        for dm, cell in group.groupby("distance_modulus"):
            x = dm + offset
            ax.plot(
                np.full(len(cell), x),
                cell["detection_fraction"],
                "o",
                color=WIDTH_COLOURS[width],
                alpha=0.55,
                ms=6,
            )
            ax.plot(
                [x - 0.05, x + 0.05],
                [cell["detection_fraction"].mean()] * 2,
                color=WIDTH_COLOURS[width],
                lw=2.5,
            )
        ax.plot([], [], "o", color=WIDTH_COLOURS[width], label=f"width {width:g} deg")
    ax.set_xlabel("distance modulus")
    ax.set_ylabel("fraction of streams detected")
    ax.set_xticks([15, 16, 17, 18, 19])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("SB 34: one point per trained model (6 x 4800 windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_spread.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def contrast_table():
    """Selected-star density of a stream against the background's, per distance.

    Fixed surface brightness (33), width (0.2 deg) and length (15 deg); five
    realizations per distance. The stream density is the selected stars within
    1 sigma of the track (68.3% of them) over that band's area.
    """
    import healpy as hp

    spec = importlib.util.spec_from_file_location(
        "sp_run", Path(__file__).parent / "run.py"
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    background, injector = run.build_sky(run.BACKGROUND_SEED)
    pixel_area = hp.nside2pixarea(injector.pix.nside, degrees=True)
    rows = []
    for dm in (15.0, 16.0, 17.0, 18.0, 19.0):
        params = {
            "morphology": "uniform",
            "richness": 33.0,
            "width": 0.2,
            "length": 15.0,
            "distance_modulus": dm,
            "age": 12.5,
            "z": 0.0002,
        }
        detected, selected, nstars = [], [], None
        for k in range(5):
            stars, resolved, _ = injector._realize_and_inject(
                params, np.random.default_rng(k)
            )
            nstars = resolved["nstars"]
            detected.append(len(stars))
            selected.append(
                injector.matched_filters["good"].select(stars, ["g", "r"], dm).sum()
            )
        band_area = 15.0 * 2 * 0.2
        stream_density = 0.683 * np.mean(selected) / band_area
        background_map = background.raw_map_full_dict["good"][dm]
        background_density = (
            background_map[background.valid_mask_full].mean() / pixel_area
        )
        rows.append(
            {
                "distance_modulus": dm,
                "nstars": nstars,
                "detected": np.mean(detected),
                "selected": np.mean(selected),
                "stream_per_deg2": stream_density,
                "background_per_deg2": background_density,
                "contrast": stream_density / background_density,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "contrast.csv", index=False)
    return table


def main(contrast):
    results = pd.read_pickle(DATA / "results.pkl")
    pooled, matched = pooled_detection(results)
    figure_detection(pooled)
    figure_training_spread(matched)
    pd.set_option("display.width", 200)
    table = pooled[pooled.richness >= 33].pivot_table(
        index=["richness", "distance_modulus"],
        columns=["width", "windows"],
        values="fraction",
    )
    print("== fraction detected, at the first threshold above the background floor")
    print((100 * table).round(0).to_string())
    spread = (
        matched[matched.windows == 4800]
        .groupby(["richness", "distance_modulus", "width"])["detection_fraction"]
        .agg(["min", "max", "std"])
    )
    print("\n== spread between the 6 quick trainings (cells that are not 0 or 1)")
    moving = spread[(spread["max"] > 0) & (spread["min"] < 1)]
    print(
        (100 * moving[["min", "max"]])
        .round(0)
        .join(moving[["std"]].round(2))
        .to_string()
    )
    print("\n== stream-free density actually reached (target 1e-3)")
    print(
        matched.groupby("windows")["background_density"]
        .describe()[["min", "50%", "max"]]
        .map(lambda v: f"{v:.1e}")
        .to_string()
    )
    if contrast:
        print("\n== contrast in the matched filter (SB 33, width 0.2, length 15)")
        print(contrast_table().round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--contrast", action="store_true")
    main(parser.parse_args().contrast)
