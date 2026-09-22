"""Figure and numbers for docs/source/experiments/two_streams.md.

Reads data/experiments/two_streams/results.pkl (run.py) and writes
docs/source/experiments/figures/two_streams/two_streams.png.

Run from the repository root:
  python scripts/experiments/two_streams/figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from streamgoggles.evaluation.footprint import THRESHOLD_GRID, band_snr

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "data" / "experiments" / "two_streams" / "results.pkl"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "two_streams"
TARGET = 1e-3
SEPARATIONS = (1.0, 2.0, 4.0)
COLOURS = {32.0: "#e6550d", 33.0: "#3182bd"}


def detected(frame, k):
    """The stream_detection criterion per realization, at threshold index k."""
    flagged = np.array([row[k] for row in frame["n_above_band"]], dtype=float)
    snr = band_snr(
        flagged,
        frame["band_pixels"].to_numpy(dtype=float),
        np.array([row[k] for row in frame["null_density_mean"]], dtype=float),
        np.array([row[k] for row in frame["null_density_std"]], dtype=float),
    )
    return (flagged >= 20) & (snr >= 2)


def matched_index(frame):
    """Threshold index at which the stream-free sky is flagged at most TARGET."""
    control = np.sum(np.stack(list(frame["n_above_no_stream"])), axis=0)
    pixels = float((frame["fp_no_stream"] + frame["tn_no_stream"]).sum())
    return int(np.flatnonzero(control / pixels <= TARGET)[0])


def stream_a(results, seed, condition):
    """Stream A's detections, in realization order, at a matched TARGET."""
    sky = results[(results.seed == seed) & (results.condition == condition)]
    a = sky[sky.which == "A"].sort_values("realization")
    return detected(a, matched_index(sky)), a


def summarise(results):
    """Per condition: A detected, paired losses and gains against A alone."""
    rows = []
    for condition in results["condition"].unique():
        lost = gained = 0
        detections, flagged = [], []
        for seed in sorted(results["seed"].unique()):
            alone, _ = stream_a(results, seed, "alone")
            with_b, a = stream_a(results, seed, condition)
            lost += int((alone & ~with_b).sum())
            gained += int((~alone & with_b).sum())
            detections.append(with_b)
            k = int(np.flatnonzero(THRESHOLD_GRID == 0.5)[0])
            flagged.extend(row[k] for row in a["n_above_band"])
        rows.append(
            {
                "condition": condition,
                "detected": float(np.concatenate(detections).mean()),
                "lost": lost,
                "gained": gained,
                "p": binomtest(lost, lost + gained).pvalue if lost + gained else 1.0,
                "flagged_median": float(np.median(flagged)),
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def main():
    results = pd.read_pickle(RESULTS)
    table = summarise(results)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for column, ax, ylabel in (
        ("detected", axes[0], "fraction of stream A detected"),
        ("flagged_median", axes[1], "flagged pixels within 1 sigma of A (median)"),
    ):
        ax.axhline(
            table.loc["alone", column], color="#636363", ls=":", lw=1.5, label="A alone"
        )
        for sb_b, colour in COLOURS.items():
            parallel = [
                table.loc[f"parallel {s:g} deg, B at SB {sb_b:g}", column]
                for s in SEPARATIONS
            ]
            ax.plot(
                SEPARATIONS,
                parallel,
                marker="o",
                color=colour,
                lw=1.8,
                label=f"B parallel, SB {sb_b:g}",
            )
            ax.plot(
                [5.0],
                [table.loc[f"crossing 60 deg, B at SB {sb_b:g}", column]],
                marker="X",
                ms=10,
                ls="none",
                color=colour,
                label=f"B crossing at 60 deg, SB {sb_b:g}",
            )
        ax.set_xticks([1, 2, 4, 5])
        ax.set_xticklabels(["1", "2", "4", "crossing"])
        ax.set_xlabel("separation of B from A across the track (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8.5, handlelength=3)
    axes[0].set_title("A at SB 33, at 1e-3 of stream-free sky flagged", fontsize=10.5)
    axes[1].set_title("A's response at threshold 0.5", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "two_streams.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
