"""Final figures and numbers for docs/source/experiments/hyperparameters.md.

One figure per statement the page makes:

  1_training_range.png   what the model is trained on (the lever)
  2_training_length.png  how long it is trained for (trades detection against
                         contamination; does not remove the seed lottery)
  3_architecture.png     depth and width (no effect)
  4_variability.png      the same configuration retrained: the spread that
                         motivates averaging
  5_ensemble.png         averaging several trainings, including the equal-cost
                         comparison with one long training
  6_matched_background.png  every model compared at the same background level,
                         since a fixed 0.5 threshold puts them at very
                         different operating points

Reads data/experiments/hyperparameters/{results,ensemble_results}.pkl and
writes to docs/source/experiments/figures/hyperparameters/.

Run from the repository root:
  python scripts/experiments/hyperparameters/summary.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streamgoggles.evaluation.footprint import THRESHOLD_GRID, stream_detection

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "experiments" / "hyperparameters"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "hyperparameters"
FIGURES.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.5
TARGET = (33.0, 34.0)
SB_LABEL = "surface brightness (mag arcsec$^{-2}$)"
FAINT, NARROW = "sb32-34.5", "sb31-34"
DPI = 110


def label(windows, training_sb, depth=2, width=12):
    return f"{windows} windows, SB {training_sb[2:]}, depth {depth}, width {width}"


def load():
    results = pd.read_pickle(DATA / "results.pkl")
    results["windows"] = 40 * results["cfg_steps_per_epoch"]
    return results


def detection(results, group_by=("configuration", "richness"), thresholds=(THRESHOLD,)):
    return stream_detection(
        results, THRESHOLD_GRID, at=thresholds, group_by=list(group_by)
    )


def curve(ax, table, name, color, marker, linestyle="-", label_text=None, offset=0.0):
    data = table[table.configuration == name].sort_values("richness")
    if data.empty:
        return
    y = data["detection_fraction"].to_numpy()
    ax.errorbar(
        data["richness"] + offset,
        y,
        yerr=[
            np.clip(y - data["detection_low"], 0, None),
            np.clip(data["detection_high"] - y, 0, None),
        ],
        marker=marker,
        color=color,
        ls=linestyle,
        capsize=2.5,
        lw=1.7,
        ms=6,
        alpha=0.9,
        label=label_text,
    )


def decorate(ax, ylabel="fraction of streams detected", log=False):
    ax.axvspan(*TARGET, color="#fde0dd", zorder=0)
    ax.set_xlabel(SB_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both" if log else "major")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([32, 33, 34, 35])


def seeds_of(results, name):
    return results[results.configuration == name]["seed"].nunique()


def figure_training_range(results, table):
    """1: what the model is trained on."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {
        (1200, NARROW): ("#1f77b4", "o", "-"),
        (4800, NARROW): ("#08306b", "s", "--"),
        (1200, FAINT): ("#e6550d", "^", "-"),
        (4800, FAINT): ("#a63603", "v", "--"),
    }
    for i, ((windows, sb), (color, marker, ls)) in enumerate(styles.items()):
        name = f"w{windows}_{sb}_d2_b12_lr0.002_bs8"
        curve(
            ax,
            table,
            name,
            color,
            marker,
            ls,
            f"{label(windows, sb)} ({seeds_of(results, name)} seeds)",
            offset=(i - 1.5) * 0.02,
        )
    decorate(ax)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Training on fainter streams is what reaches SB 34", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "1_training_range.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_training_length(results, table):
    """2: how long it is trained for."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = plt.get_cmap("viridis")
    lengths = sorted(results[results.cfg_training_sb == FAINT]["windows"].unique())
    for i, windows in enumerate(lengths):
        name = f"w{windows}_{FAINT}_d2_b12_lr0.002_bs8"
        color = colours(i / max(len(lengths) - 1, 1))
        curve(
            axes[0],
            table,
            name,
            color,
            "o",
            "-",
            f"{windows} windows ({seeds_of(results, name)} seeds)",
            offset=(i - len(lengths) / 2) * 0.02,
        )
        data = table[table.configuration == name].sort_values("richness")
        axes[1].plot(
            data["richness"],
            data["background_density"],
            marker="o",
            color=color,
            lw=1.7,
            label=f"{windows} windows",
        )
    decorate(axes[0])
    decorate(axes[1], "background density in stream-free bands", log=True)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Detection", fontsize=11)
    axes[1].set_title("Background flagged", fontsize=11)
    fig.suptitle(
        "Training longer trades detection for a cleaner background "
        f"(training SB {FAINT[2:]}, threshold {THRESHOLD})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "2_training_length.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_architecture(results, table):
    """3: depth and width."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {
        (2, 12): ("#31a354", "o", "-"),
        (3, 12): ("#006d2c", "s", "--"),
        (4, 12): ("#c51b8a", "D", ":"),
        (4, 24): ("#7a0177", "^", "-."),
    }
    for i, ((depth, width), (color, marker, ls)) in enumerate(styles.items()):
        name = f"w4800_{FAINT}_d{depth}_b{width}_lr0.002_bs8"
        curve(
            ax,
            table,
            name,
            color,
            marker,
            ls,
            f"depth {depth}, width {width} ({seeds_of(results, name)} seeds)",
            offset=(i - 1.5) * 0.02,
        )
    decorate(ax)
    ax.legend(fontsize=8)
    ax.set_title(
        "Depth and width change nothing (4800 windows, SB 32-34.5)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "3_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_variability(results):
    """4: the same configuration, retrained."""
    per_seed = detection(results, ("configuration", "seed", "richness"))
    names = [
        f"w4800_{FAINT}_d2_b12_lr0.002_bs8",
        f"w1200_{FAINT}_d2_b12_lr0.002_bs8",
        f"w4800_{NARROW}_d2_b12_lr0.002_bs8",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colours = ["#e6550d", "#fdae6b", "#1f77b4"]
    for i, (name, color) in enumerate(zip(names, colours, strict=True)):
        data = per_seed[per_seed.configuration == name]
        if data.empty:
            continue
        for sb, group in data.groupby("richness"):
            x = sb + (i - 1) * 0.07
            ax.plot(
                np.full(len(group), x),
                group["detection_fraction"],
                "o",
                color=color,
                alpha=0.55,
                ms=7,
            )
            ax.plot(
                [x - 0.03, x + 0.03],
                [group["detection_fraction"].mean()] * 2,
                color=color,
                lw=2.5,
            )
        windows = int(name.split("_")[0][1:])
        sb_range = name.split("_")[1]
        ax.plot(
            [],
            [],
            "o",
            color=color,
            label=f"{label(windows, sb_range)} ({seeds_of(results, name)} seeds)",
        )
    decorate(ax)
    ax.set_title(
        "Each point is one training: the spread motivates averaging", fontsize=11
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "4_variability.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_ensemble(results, table):
    """5: averaging trainings, and the equal-cost comparison."""
    path = DATA / "ensemble_results.pkl"
    if not path.exists():
        return
    ensembles = pd.read_pickle(path)
    detection_ensemble = stream_detection(
        ensembles,
        THRESHOLD_GRID,
        at=(THRESHOLD,),
        group_by=["ensemble_size", "richness"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = plt.get_cmap("plasma")
    sizes = sorted(detection_ensemble["ensemble_size"].unique())
    for i, size in enumerate(sizes):
        data = detection_ensemble[detection_ensemble.ensemble_size == size].sort_values(
            "richness"
        )
        y = data["detection_fraction"].to_numpy()
        axes[0].errorbar(
            data["richness"] + (i - len(sizes) / 2) * 0.02,
            y,
            yerr=[
                np.clip(y - data["detection_low"], 0, None),
                np.clip(data["detection_high"] - y, 0, None),
            ],
            marker="o",
            color=colours(i / max(len(sizes) - 1, 1)),
            capsize=2.5,
            lw=1.7,
            label=f"{size} model{'s' if size > 1 else ''} averaged",
        )
    decorate(axes[0])
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"Averaging N trainings of 4800 windows (threshold {THRESHOLD})", fontsize=11
    )

    # Equal training cost: 4 x 4800 windows averaged vs one 19200-window model.
    long_name = f"w19200_{FAINT}_d2_b12_lr0.002_bs8"
    per_seed = detection(results, ("configuration", "seed", "richness"))
    long_seeds = per_seed[per_seed.configuration == long_name]
    if not long_seeds.empty:
        summary = long_seeds.groupby("richness")["detection_fraction"].agg(
            ["mean", "min", "max"]
        )
        axes[1].errorbar(
            summary.index,
            summary["mean"],
            yerr=[summary["mean"] - summary["min"], summary["max"] - summary["mean"]],
            marker="s",
            color="#08306b",
            capsize=3,
            lw=1.8,
            label=f"one model, 19200 windows ({len(long_seeds.seed.unique())} seeds, min-max)",
        )
    four = detection_ensemble[detection_ensemble.ensemble_size == 4].sort_values(
        "richness"
    )
    if not four.empty:
        axes[1].plot(
            four["richness"],
            four["detection_fraction"],
            marker="o",
            color="#e6550d",
            lw=2.2,
            label="4 models of 4800 windows, averaged",
        )
    decorate(axes[1])
    axes[1].legend(fontsize=8)
    axes[1].set_title("Same training cost, two ways to spend it", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "5_ensemble.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_matched_background(results):
    """6: every model at the same background level."""
    targets = [1e-4, 1e-3]
    rows = []
    for (name, seed, sb), group in results[results["n_above_band"].notna()].groupby(
        ["configuration", "seed", "richness"]
    ):
        control = np.sum(np.stack(list(group["n_above_no_stream"])), axis=0)
        control_pixels = float((group["fp_no_stream"] + group["tn_no_stream"]).sum())
        density = control / control_pixels
        flagged = np.stack(list(group["n_above_band"]))
        band_pixels = group["band_pixels"].to_numpy()[:, None]
        for target in targets:
            reachable = np.flatnonzero(density <= target)
            if not reachable.size:
                continue
            k = reachable[0]
            detected = (flagged[:, k] >= 20) & (flagged[:, k] / band_pixels[:, 0] > 0)
            rows.append(
                {
                    "configuration": name,
                    "seed": seed,
                    "richness": sb,
                    "target": target,
                    "detected": detected.mean(),
                }
            )
    matched = pd.DataFrame(rows)
    if matched.empty:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(13, 5), sharey=True)
    styles = {
        f"w1200_{NARROW}_d2_b12_lr0.002_bs8": ("#1f77b4", "o", "1200 w, SB 31-34"),
        f"w4800_{NARROW}_d2_b12_lr0.002_bs8": ("#08306b", "s", "4800 w, SB 31-34"),
        f"w1200_{FAINT}_d2_b12_lr0.002_bs8": ("#e6550d", "^", "1200 w, SB 32-34.5"),
        f"w4800_{FAINT}_d2_b12_lr0.002_bs8": ("#a63603", "v", "4800 w, SB 32-34.5"),
        f"w19200_{FAINT}_d2_b12_lr0.002_bs8": ("#54278f", "D", "19200 w, SB 32-34.5"),
    }
    for ax, target in zip(axes, targets, strict=True):
        for name, (color, marker, text) in styles.items():
            data = matched[(matched.configuration == name) & (matched.target == target)]
            if data.empty:
                continue
            summary = data.groupby("richness")["detected"].agg(["mean", "std"])
            ax.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                marker=marker,
                color=color,
                capsize=2.5,
                lw=1.7,
                label=text,
            )
        decorate(ax)
        ax.set_title(f"background density {target:.0e}", fontsize=11)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Compared at the same background level, not at the same threshold", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "6_matched_background.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load()
    table = detection(results)
    figure_training_range(results, table)
    figure_training_length(results, table)
    figure_architecture(results, table)
    figure_variability(results)
    figure_ensemble(results, table)
    figure_matched_background(results)
    pd.set_option("display.width", 220)
    print("== streams detected at threshold 0.5 (all seeds of each configuration)")
    print(
        table.pivot_table(
            index="configuration", columns="richness", values="detection_fraction"
        )
        .round(2)
        .to_string()
    )
    print("\n== seeds per configuration")
    print(results.groupby("configuration")["seed"].nunique().to_string())
    print("\n== spread between trainings (std over seeds), threshold 0.5")
    per_seed = detection(results, ("configuration", "seed", "richness"))
    print(
        per_seed.pivot_table(
            index="configuration",
            columns="richness",
            values="detection_fraction",
            aggfunc="std",
        )
        .round(2)
        .to_string()
    )
    print("\nfigures in", FIGURES)


if __name__ == "__main__":
    main()
