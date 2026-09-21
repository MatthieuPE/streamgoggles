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
  7_sampling.png         the same models scored on 20, 100 and 500 streams per
                         surface brightness -- evaluation precision, not a
                         difference between models
  8_decoy.png            the fixed decoy box against the shifted one

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

from streamgoggles.evaluation.footprint import (
    THRESHOLD_GRID,
    detection_at_false_alarm_rate,
    stream_detection,
)

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


MATCHED_TARGETS = (1e-4, 1e-3)


def matched_background(frame, keys):
    """Detection with each group's threshold set by a false-alarm budget.

    A thin adapter over the library's `detection_at_false_alarm_rate`, keeping
    the ``detected`` column name the figures below use.
    """
    table = detection_at_false_alarm_rate(
        frame, THRESHOLD_GRID, targets=MATCHED_TARGETS, group_by=list(keys)
    )
    return table.rename(columns={"detection_fraction": "detected"})[
        [*keys, "target", "detected"]
    ]


def _wilson(k, n, z=1.0):
    """Wilson interval, the same one `stream_detection` reports."""
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def figure_sampling(results):
    """7: how many evaluation streams per surface brightness the ensembles needed.

    The scorings are nested -- the realization seed is [seed, surface-brightness
    index, realization], so the first 20 of the 100 and the first 100 of the 500
    are the same skies -- which makes this a pure sample-size comparison of the
    same trained models.
    """
    paths = {
        20: DATA / "ensemble_results_20streams.pkl",
        100: DATA / "ensemble_results_100streams.pkl",
        500: DATA / "ensemble_results.pkl",
    }
    runs = {n: pd.read_pickle(path) for n, path in paths.items() if path.exists()}
    if len(runs) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    styles = {
        20: ("#c6dbef", "o", ":"),
        100: ("#6baed6", "s", "--"),
        500: ("#08306b", "D", "-"),
    }
    shifts = {20: -0.03, 100: 0.0, 500: 0.03}

    for n, frame in runs.items():
        table = stream_detection(
            frame,
            THRESHOLD_GRID,
            at=(THRESHOLD,),
            group_by=["ensemble_size", "richness"],
        )
        data = table[table.ensemble_size == 6].sort_values("richness")
        colour, marker, ls = styles[n]
        y = data["detection_fraction"].to_numpy()
        axes[0].errorbar(
            data["richness"] + shifts[n],
            y,
            yerr=bars(data, y),
            marker=marker,
            ls=ls,
            color=colour,
            capsize=3,
            lw=1.8,
            label=f"{n} evaluation streams per point",
        )
    decorate(axes[0])
    legend(axes[0])
    axes[0].set_title("6 models averaged", fontsize=11)

    # The comparison that 20 streams could not settle: does averaging still
    # help once both models are put at the same false-alarm rate?
    positions = {1: 0.0, 6: 1.0}
    spread = {20: -0.14, 100: 0.0, 500: 0.14}
    for n, frame in runs.items():
        matched = matched_background(frame, ("ensemble_size", "richness"))
        colour, marker, _ = styles[n]
        for size in (1, 6):
            row = matched[
                (matched.ensemble_size == size)
                & (matched.richness == 33.5)
                & (matched.target == 1e-3)
            ]
            if row.empty:
                continue
            k = round(row["detected"].iloc[0] * n)
            low, high = _wilson(k, n)
            x = positions[size] + spread[n]
            axes[1].errorbar(
                [x],
                [k / n],
                yerr=[[k / n - low], [high - k / n]],
                marker=marker,
                color=colour,
                capsize=4,
                ms=8,
                lw=2,
                label=f"{n} evaluation streams per point" if size == 1 else None,
            )
            axes[1].annotate(
                f"{k}/{n}",
                (x, k / n),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8.5,
                color=colour,
            )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["1 model", "6 models averaged"])
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylim(0, 0.9)
    axes[1].set_ylabel("fraction of streams detected")
    axes[1].grid(alpha=0.3, axis="y")
    legend(axes[1], loc="upper left")
    axes[1].set_title("SB 33.5, matched background 1e-3", fontsize=11)
    fig.suptitle(
        "Same trained models, scored on 20, 100 and 500 streams per surface brightness",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "7_sampling.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_matched_background(results):
    """6: every model at the same background level, ensembles included."""
    matched = matched_background(results, ("configuration", "seed", "richness"))
    if matched.empty:
        return
    path = DATA / "ensemble_results.pkl"
    ensembles = (
        matched_background(pd.read_pickle(path), ("ensemble_size", "richness"))
        if path.exists()
        else pd.DataFrame()
    )
    fig, axes = plt.subplots(1, len(MATCHED_TARGETS), figsize=(13.5, 5.2), sharey=True)
    styles = {
        f"w4800_{NARROW}_d2_b12_lr0.002_bs8": ("#1f77b4", "o", "-", "4800 w, SB 31-34"),
        f"w1200_{FAINT}_d2_b12_lr0.002_bs8": (
            "#fdae6b",
            "^",
            "-",
            "1200 w, SB 32-34.5",
        ),
        f"w4800_{FAINT}_d2_b12_lr0.002_bs8": (
            "#e6550d",
            "v",
            "--",
            "4800 w, SB 32-34.5",
        ),
        f"w19200_{FAINT}_d2_b12_lr0.002_bs8": (
            "#a63603",
            "D",
            ":",
            "19200 w, SB 32-34.5",
        ),
    }
    for ax, target in zip(axes, MATCHED_TARGETS, strict=True):
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
        if not ensembles.empty:
            for size, colour, ls in ((1, "#7fcdbb", "-"), (6, "#00441b", "-")):
                data = ensembles[
                    (ensembles.ensemble_size == size) & (ensembles.target == target)
                ].sort_values("richness")
                if data.empty:
                    continue
                ax.plot(
                    data["richness"],
                    data["detected"],
                    marker="*",
                    ms=10,
                    color=colour,
                    ls=ls,
                    lw=2.0,
                    label=f"{size} model{'s' if size > 1 else ''} averaged",
                )
        decorate(ax)
        ax.set_title(f"background density {target:.0e}", fontsize=10.5)
    legend(axes[0])
    fig.suptitle("Matched background level", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "6_matched_background.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_decoy(results):
    """8: the fixed decoy box against the shifted one, at a matched 1e-3."""
    names = {
        "w4800_sb32-34.5_d2_b12_lr0.002_bs8": (
            "4800 w, shifted box",
            "#fdae6b",
            "o",
            "--",
        ),
        "w4800_sb32-34.5_d2_b12_lr0.002_bs8_boxdecoy": (
            "4800 w, fixed box",
            "#e6550d",
            "o",
            "-",
        ),
        "w19200_sb32-34.5_d2_b12_lr0.002_bs8": (
            "19200 w, shifted box",
            "#9ecae1",
            "s",
            "--",
        ),
        "w19200_sb32-34.5_d2_b12_lr0.002_bs8_boxdecoy": (
            "19200 w, fixed box",
            "#08519c",
            "s",
            "-",
        ),
    }
    subset = results[results.configuration.isin(names)]
    if subset.empty or subset.configuration.nunique() < len(names):
        return
    # Each trained model gets its own threshold, then the streams detected by
    # the models of one configuration are pooled, so the Wilson bar reflects
    # every stream behind the point.
    matched = detection_at_false_alarm_rate(
        subset,
        THRESHOLD_GRID,
        targets=(1e-3,),
        group_by=["configuration", "seed", "richness"],
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, (name, (text, color, marker, ls)) in enumerate(names.items()):
        pooled = (
            matched[matched.configuration == name]
            .groupby("richness")[["n_detected", "n_realizations"]]
            .sum()
        )
        k = pooled["n_detected"].to_numpy()
        n = pooled["n_realizations"].to_numpy()
        y = k / n
        low, high = np.array([_wilson(a, b) for a, b in zip(k, n, strict=True)]).T
        ax.errorbar(
            pooled.index + (i - 1.5) * 0.02,
            y,
            yerr=[np.clip(y - low, 0, None), np.clip(high - y, 0, None)],
            marker=marker,
            color=color,
            ls=ls,
            capsize=2.5,
            lw=1.7,
            label=text,
        )
    decorate(ax)
    legend(ax)
    ax.set_title("Decoy channel, at 1e-3 of stream-free sky flagged", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "8_decoy.png", dpi=DPI, bbox_inches="tight")
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
    figure_sampling(results)
    figure_decoy(results)
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
    matched = matched_background(results, ("configuration", "seed", "richness"))
    for target in MATCHED_TARGETS:
        print(f"\n== streams detected at a matched background density of {target:.0e}")
        print(
            matched[matched.target == target]
            .pivot_table(index="configuration", columns="richness", values="detected")
            .round(2)
            .to_string()
        )
    print("\nfigures in", FIGURES)


if __name__ == "__main__":
    main()
