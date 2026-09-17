"""Figures and numbers for docs/source/experiments/hyperparameters.md.

Reads data/experiments/hyperparameters/results.pkl (written by run.py) and
writes figures to docs/source/experiments/figures/hyperparameters/.

For every configuration, pooling the streams of all its seeds:
- fraction of streams detected (`stream_detection`: >= 20 pixels flagged
  within 1 sigma of the track, SNR >= 2 against stream-shaped background
  bands), at thresholds 0.5 and 0.1;
- completeness C = S_s/S_t, contamination F = B_s/B_t and contrast C/F at
  threshold 0.5 (`detection_metrics`).
Per-seed detection fractions are printed too, to show the seed spread.

Run from the repository root:
  python scripts/experiments/hyperparameters/figures.py phase1
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streamgoggles.evaluation.footprint import (
    THRESHOLD_GRID,
    detection_metrics,
    stream_detection,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "data" / "experiments" / "hyperparameters" / "results.pkl"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "hyperparameters"
FIGURES.mkdir(parents=True, exist_ok=True)
TARGET = (33.0, 34.0)
SB_LABEL = "surface brightness (mag arcsec$^{-2}$)"
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]


def label(config_row):
    """Readable configuration label from its cfg_ columns."""
    return (
        f"{40 * config_row.cfg_steps_per_epoch} windows, SB {config_row.cfg_training_sb[2:]}, "
        f"depth {config_row.cfg_depth}, width {config_row.cfg_base_width}, "
        f"lr {config_row.cfg_lr:g}, batch {config_row.cfg_batch_size}"
    )


def main(phase):
    results = pd.read_pickle(RESULTS)
    results = results[results["phase"] == phase]
    cfg_columns = [c for c in results.columns if c.startswith("cfg_")]
    configs = results[["configuration", *cfg_columns]].drop_duplicates("configuration")
    configs = configs.sort_values(cfg_columns).reset_index(drop=True)
    colors = plt.get_cmap("tab10")

    detection = stream_detection(
        results, THRESHOLD_GRID, at=(0.1, 0.5), group_by=["configuration", "richness"]
    )
    per_seed = stream_detection(
        results,
        THRESHOLD_GRID,
        at=(0.1, 0.5),
        group_by=["configuration", "seed", "richness"],
    )
    metrics = detection_metrics(
        results, THRESHOLD_GRID, group_by=["configuration", "richness"], at=(0.5,)
    )
    n_seeds = results.groupby("configuration")["seed"].nunique()

    # Figure 1: fraction of streams detected, thresholds 0.5 and 0.1.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, threshold in zip(axes, (0.5, 0.1), strict=True):
        ax.axvspan(*TARGET, color="#fde0dd", zorder=0, label="target range")
        for i, config in enumerate(configs.itertuples(index=False)):
            d = detection[
                (detection.configuration == config.configuration)
                & np.isclose(detection.threshold, threshold, atol=2e-3)
            ].sort_values("richness")
            y = d["detection_fraction"].to_numpy()
            ax.errorbar(
                d["richness"] + (i - len(configs) / 2) * 0.02,
                y,
                yerr=[
                    np.clip(y - d["detection_low"].to_numpy(), 0, None),
                    np.clip(d["detection_high"].to_numpy() - y, 0, None),
                ],
                marker=MARKERS[i % len(MARKERS)],
                color=colors(i % 10),
                capsize=2,
                lw=1.6,
                alpha=0.9,
                label=f"{label(config)} ({n_seeds[config.configuration]} seeds)",
            )
        ax.set_title(f"threshold {threshold}")
        ax.set_xlabel(SB_LABEL)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of streams detected")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"{phase}_detection.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: completeness, contamination, contrast at 0.5.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = (
        ("completeness", r"completeness $C = S_s/S_t$", False),
        ("contamination", r"contamination $F = B_s/B_t$", True),
        ("contrast", r"contrast $C/F$", True),
    )
    for i, config in enumerate(configs.itertuples(index=False)):
        m = metrics[metrics.configuration == config.configuration].sort_values(
            "richness"
        )
        for ax, (column, _, _) in zip(axes, panels, strict=True):
            ax.plot(
                m["richness"],
                m[column],
                marker=MARKERS[i % len(MARKERS)],
                color=colors(i % 10),
                lw=1.6,
                alpha=0.9,
                label=label(config),
            )
    for ax, (_, ylabel, log) in zip(axes, panels, strict=True):
        ax.axvspan(*TARGET, color="#fde0dd", zorder=0)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(SB_LABEL)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both" if log else "major")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("threshold 0.5, all seeds pooled", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{phase}_pixel_metrics.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Numbers.
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    table = detection.pivot_table(
        index=["configuration", "threshold"],
        columns="richness",
        values="detection_fraction",
    )
    print("== fraction of streams detected (all seeds pooled)")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n== per seed, threshold 0.5")
    print(
        per_seed[np.isclose(per_seed.threshold, 0.5)]
        .pivot_table(
            index=["configuration", "seed"],
            columns="richness",
            values="detection_fraction",
        )
        .to_string(float_format=lambda v: f"{v:.2f}")
    )
    print("\n== threshold 0.5: median SNR, flagged within 1 sigma, background density")
    at_half = detection[np.isclose(detection.threshold, 0.5)]
    print(
        at_half.pivot_table(
            index="configuration",
            columns="richness",
            values=["median_snr", "flagged_in_band", "background_density"],
        ).to_string(float_format=lambda v: f"{v:.3g}")
    )
    print("\n== threshold 0.5: C, F, C/F")
    print(
        metrics.pivot_table(
            index="configuration",
            columns="richness",
            values=["completeness", "contamination", "contrast"],
        ).to_string(float_format=lambda v: f"{v:.3g}")
    )
    print("\n== training time (s), mean over seeds")
    print(results.groupby("configuration")["train_s"].mean().round(0).to_string())
    print("figures in", FIGURES)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1")
