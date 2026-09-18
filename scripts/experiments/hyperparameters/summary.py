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

Two different uncertainties appear here, drawn differently:

  error bars   sampling. Each model is scored on 20 injected streams per
               surface brightness (120 when 6 seeds are pooled). "k detected
               out of n injected" is binomial (n is fixed, k <= n), not
               Poisson, so the bars are a Wilson 68% interval: it stays inside
               [0, 1] and keeps a sensible width at k = 0 or k = n, where
               sqrt(k) would give zero or reach past 1.
  thin lines   one trained model each, where the spread between trainings is
               the point (figures 4 and 5). The comparison figures show only
               the mean over a configuration's trainings, to stay readable;
               the spread itself is figure 4.

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


def label(windows, training_sb):
    return f"{windows} w, SB {training_sb[2:]}"


def load():
    results = pd.read_pickle(DATA / "results.pkl")
    results["windows"] = 40 * results["cfg_steps_per_epoch"]
    return results


def detection(results, group_by=("configuration", "richness"), thresholds=(THRESHOLD,)):
    return stream_detection(
        results, THRESHOLD_GRID, at=thresholds, group_by=list(group_by)
    )


def bars(data, y):
    """Asymmetric sampling interval, clipped at 0 for matplotlib."""
    return [
        np.clip(y - data["detection_low"].to_numpy(), 0, None),
        np.clip(data["detection_high"].to_numpy() - y, 0, None),
    ]


def curve(
    ax,
    table,
    name,
    color,
    marker,
    linestyle="-",
    label_text=None,
    offset=0.0,
):
    """One configuration: the mean over its trainings, with sampling bars."""
    data = table[table.configuration == name].sort_values("richness")
    if data.empty:
        return
    y = data["detection_fraction"].to_numpy()
    ax.errorbar(
        data["richness"] + offset,
        y,
        yerr=bars(data, y),
        marker=marker,
        color=color,
        ls=linestyle,
        capsize=2.5,
        lw=1.7,
        ms=6,
        alpha=0.9,
        label=label_text,
    )


def legend(ax, **kwargs):
    """Series names only: what the bars and lines mean belongs in the page text.

    Handles are long enough to tell solid from dashed and dotted.
    """
    ax.legend(fontsize=8.5, handlelength=3.4, **kwargs)


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


def figure_training_range(results, table):
    """1: what the model is trained on."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
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
            label(windows, sb),
            offset=(i - 1.5) * 0.02,
        )
    decorate(ax)
    legend(ax, loc="upper right")
    ax.set_title("Training range", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "1_training_range.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_training_length(results, table):
    """2: how long it is trained for."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
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
            f"{windows} w",
            offset=(i - len(lengths) / 2) * 0.02,
        )
        data = table[table.configuration == name].sort_values("richness")
        axes[1].plot(
            data["richness"],
            data["background_density"],
            marker="o",
            color=color,
            lw=1.7,
            label=f"{windows} w",
        )
    decorate(axes[0])
    decorate(axes[1], "background density in stream-free bands", log=True)
    legend(axes[0])
    legend(axes[1])
    axes[0].set_title("Detection", fontsize=11)
    axes[1].set_title("Background flagged", fontsize=11)
    fig.suptitle(f"Training length (training SB {FAINT[2:]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "2_training_length.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_architecture(results, table):
    """3: depth and width."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
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
            f"depth {depth}, width {width}",
            offset=(i - 1.5) * 0.02,
        )
    decorate(ax)
    legend(ax)
    ax.set_title("Network size", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_variability(results, per_seed):
    """4: the same configuration, retrained."""
    names = [
        f"w4800_{FAINT}_d2_b12_lr0.002_bs8",
        f"w1200_{FAINT}_d2_b12_lr0.002_bs8",
        f"w4800_{NARROW}_d2_b12_lr0.002_bs8",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
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
                ms=6,
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
            label=label(windows, sb_range),
        )
    decorate(ax)
    legend(ax)
    ax.set_title("One point per trained model", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "4_variability.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_ensemble(results, per_seed):
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
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
            yerr=bars(data, y),
            marker="o",
            color=colours(i / max(len(sizes) - 1, 1)),
            capsize=2.5,
            lw=1.7,
            label=f"{size} model{'s' if size > 1 else ''} averaged",
        )
    decorate(axes[0])
    legend(axes[0])
    axes[0].set_title("Averaging models (4800 w each)", fontsize=11)

    # Equal training cost: 4 x 4800 = 19200 windows either way. One training at
    # 19200 windows is the alternative to averaging four at 4800; the several
    # 19200-window trainings shown are repeats of that same single-model
    # outcome, not a larger spend.
    long_name = f"w19200_{FAINT}_d2_b12_lr0.002_bs8"
    long_seeds = per_seed[per_seed.configuration == long_name]
    if not long_seeds.empty:
        for j, (_, one) in enumerate(long_seeds.groupby("seed")):
            one = one.sort_values("richness")
            axes[1].plot(
                one["richness"],
                one["detection_fraction"],
                color="#08306b",
                lw=0.9,
                alpha=0.45,
                label="single 19200 w models" if j == 0 else None,
            )
        mean = long_seeds.groupby("richness")["detection_fraction"].mean()
        axes[1].plot(
            mean.index,
            mean.to_numpy(),
            marker="s",
            color="#08306b",
            lw=2.2,
            label="their curve average",
        )
    four = detection_ensemble[detection_ensemble.ensemble_size == 4].sort_values(
        "richness"
    )
    if not four.empty:
        y = four["detection_fraction"].to_numpy()
        axes[1].errorbar(
            four["richness"],
            y,
            yerr=bars(four, y),
            marker="o",
            color="#e6550d",
            capsize=3,
            lw=2.2,
            label="4 x 4800 w averaged",
        )
    decorate(axes[1])
    legend(axes[1])
    axes[1].set_title("Same training cost: 19200 windows", fontsize=11)
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
        for target in targets:
            reachable = np.flatnonzero(density <= target)
            if not reachable.size:
                continue
            k = reachable[0]
            rows.append(
                {
                    "configuration": name,
                    "seed": seed,
                    "richness": sb,
                    "target": target,
                    "detected": (flagged[:, k] >= 20).mean(),
                }
            )
    matched = pd.DataFrame(rows)
    if matched.empty:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(13.5, 5.2), sharey=True)
    styles = {
        f"w1200_{NARROW}_d2_b12_lr0.002_bs8": ("#1f77b4", "o", "-", "1200 w, SB 31-34"),
        f"w4800_{NARROW}_d2_b12_lr0.002_bs8": (
            "#08306b",
            "s",
            "--",
            "4800 w, SB 31-34",
        ),
        f"w1200_{FAINT}_d2_b12_lr0.002_bs8": (
            "#e6550d",
            "^",
            "-",
            "1200 w, SB 32-34.5",
        ),
        f"w4800_{FAINT}_d2_b12_lr0.002_bs8": (
            "#a63603",
            "v",
            "--",
            "4800 w, SB 32-34.5",
        ),
        f"w19200_{FAINT}_d2_b12_lr0.002_bs8": (
            "#54278f",
            "D",
            ":",
            "19200 w, SB 32-34.5",
        ),
    }
    for ax, target in zip(axes, targets, strict=True):
        for name, (color, marker, ls, text) in styles.items():
            data = matched[(matched.configuration == name) & (matched.target == target)]
            if data.empty:
                continue
            summary = data.groupby("richness")["detected"].mean()
            ax.plot(
                summary.index,
                summary.to_numpy(),
                marker=marker,
                color=color,
                ls=ls,
                lw=1.7,
                label=text,
            )
        decorate(ax)
        ax.set_title(f"background density {target:.0e}", fontsize=10.5)
    legend(axes[0])
    fig.suptitle("Matched background level", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "6_matched_background.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load()
    table = detection(results)
    per_seed = detection(results, ("configuration", "seed", "richness"))
    figure_training_range(results, table)
    figure_training_length(results, table)
    figure_architecture(results, table)
    figure_variability(results, per_seed)
    figure_ensemble(results, per_seed)
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
