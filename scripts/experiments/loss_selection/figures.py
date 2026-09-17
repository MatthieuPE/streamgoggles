"""Figures and numbers for docs/source/experiments/loss_selection.md.

Reads data/experiments/loss_selection/results.pkl (written by run.py) and
writes the page's figures to docs/source/experiments/figures/loss_selection/.

Metrics, per trained model (one seed) and surface brightness, counts summed
over the 10 realizations, at a probability threshold t:
  S_t true stream pixels,     S_s of them with p > t
  B_t true background pixels, B_s of them with p > t
  completeness  C = S_s / S_t
  contamination F = B_s / B_t      (B_s counted as at least 1)
  contrast      C / F
Curves show C and F averaged over seeds, and the contrast as the ratio of
those averages (a per-seed ratio explodes for a seed with F near 0); bars are
the per-seed minimum and maximum. Points where the median seed flags fewer
than MIN_STREAM_PIXELS stream pixels are drawn hollow: too few counts to
measure a contrast.

Run from the repository root:  python scripts/experiments/loss_selection/figures.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streamgoggles.evaluation.footprint import THRESHOLD_GRID

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "data" / "experiments" / "loss_selection" / "results.pkl"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "loss_selection"
FIGURES.mkdir(parents=True, exist_ok=True)

SELECTED = ("batch_dice", 0.05, 8)
REFERENCE_THRESHOLDS = (0.5, 0.9)
SENSITIVITY_THRESHOLDS = (0.1, 0.5, 0.9, 0.99, 0.999)
MIN_STREAM_PIXELS = 20
TRAINING_RANGE = (31, 34)
SB_LABEL = "surface brightness (mag arcsec$^{-2}$)"
DPI = 110

# One distinct color + marker per configuration; losses share a hue family.
CONFIG_STYLE = {
    ("dice", 0.05, 2): {"color": "#1f77b4", "marker": "o", "ls": "-"},
    ("dice", 0.3, 2): {"color": "#6baed6", "marker": "s", "ls": "--"},
    ("dice", 0.3, 8): {"color": "#08306b", "marker": "D", "ls": ":"},
    ("dice_bce", 0.05, 2): {"color": "#e6550d", "marker": "^", "ls": "-"},
    ("dice_bce", 0.3, 2): {"color": "#fdae6b", "marker": "v", "ls": "--"},
    ("batch_dice", 0.05, 2): {"color": "#31a354", "marker": "P", "ls": "-"},
    ("batch_dice", 0.3, 2): {"color": "#a1d99b", "marker": "X", "ls": "--"},
    ("batch_dice", 0.3, 8): {"color": "#006d2c", "marker": "*", "ls": ":"},
    ("batch_dice", 0.05, 8): {"color": "#c51b8a", "marker": "h", "ls": "-"},
}
LOSS_NAME = {"dice": "Dice", "dice_bce": "Dice+BCE", "batch_dice": "batch Dice"}


def config_label(config):
    loss, bf, bs = config
    return f"{LOSS_NAME[loss]}, bg_frac {bf}, batch {bs}"


def threshold_index(t):
    return int(np.argmin(np.abs(THRESHOLD_GRID - t)))


def load_counts():
    """Per (config, seed, background, SB): summed counts at every threshold."""
    df = pd.read_pickle(RESULTS)
    df = df[df["n_above_stream"].notna()]
    rows = []
    keys = [
        "loss",
        "background_fraction",
        "batch_size",
        "seed",
        "background",
        "richness",
    ]
    for (loss, bf, bs, seed, background, sb), g in df.groupby(keys):
        rows.append(
            {
                "config": (loss, bf, bs),
                "seed": seed,
                "background": background,
                "sb": sb,
                "S_s": np.sum(np.stack(list(g["n_above_stream"])), axis=0).astype(
                    float
                ),
                "B_s": np.sum(np.stack(list(g["n_above_background"])), axis=0).astype(
                    float
                ),
                "S_t": float(g["n_true_pixels"].sum()),
                "B_t": float((g["fp"] + g["tn"]).sum()),
            }
        )
    return pd.DataFrame(rows)


COUNTS = load_counts()


def metrics(config, background, t):
    """Per SB: mean C, mean F, contrast of means, per-seed ranges, low-count flag."""
    k = threshold_index(t)
    sub = COUNTS[(COUNTS.config == config) & (COUNTS.background == background)]
    out = []
    for sb, g in sub.groupby("sb"):
        C = np.array([r.S_s[k] / r.S_t for r in g.itertuples()])
        F = np.array([max(r.B_s[k], 1.0) / r.B_t for r in g.itertuples()])
        contrast = C / F
        # No stream pixel flagged at all: the contrast is undefined, not zero.
        positive = contrast[contrast > 0]
        out.append(
            {
                "sb": sb,
                "C": C.mean(),
                "C_lo": C.min(),
                "C_hi": C.max(),
                "F": F.mean(),
                "F_lo": F.min(),
                "F_hi": F.max(),
                "contrast": C.mean() / F.mean() if C.mean() > 0 else np.nan,
                "contrast_lo": positive.min() if positive.size else np.nan,
                "contrast_hi": positive.max() if positive.size else np.nan,
                "low_counts": np.median([r.S_s[k] for r in g.itertuples()])
                < MIN_STREAM_PIXELS,
            }
        )
    return pd.DataFrame(out)


PANELS = [
    ("C", "completeness $C = S_s/S_t$", False),
    ("F", "contamination $F = B_s/B_t$", True),
    ("contrast", "contrast $C/F$", True),
]


def draw(ax, m, metric, offset=0.0, label=None, alpha=0.9, lw=1.6, ms=6, **style):
    """Mean curve with faint per-seed min-max bars; hollow where counts are too low."""
    x = m["sb"].to_numpy() + offset
    y = m[metric].to_numpy()
    lower = np.clip(y - m[f"{metric}_lo"].to_numpy(), 0, None)
    upper = np.clip(m[f"{metric}_hi"].to_numpy() - y, 0, None)
    container = ax.errorbar(
        x,
        y,
        yerr=[lower, upper],
        capsize=2,
        elinewidth=0.9,
        lw=lw,
        ms=ms,
        alpha=alpha,
        label=label,
        mfc=style["color"],
        **style,
    )
    # The seed range matters, but must not drown the curves: faint bars.
    for artist in (*container[1], *container[2]):
        artist.set_alpha(0.3)
    # Contamination does not depend on how many stream pixels were found.
    hollow = m["low_counts"].to_numpy() & (metric != "F")
    if hollow.any():
        ax.plot(
            x[hollow],
            y[hollow],
            ls="none",
            marker=style["marker"],
            ms=ms + 1,
            mfc="white",
            mec=style["color"],
            alpha=1.0,
            zorder=5,
        )


LOG_LIMITS = {"F": (2e-6, 0.1), "contrast": (1.0, 3e4)}


def decorate(ax, log, ylabel, metric=None):
    ax.axvspan(*TRAINING_RANGE, color="0.94", zorder=0)
    ax.grid(alpha=0.3, which="both" if log else "major")
    if log:
        ax.set_yscale("log")
        if metric in LOG_LIMITS:
            ax.set_ylim(*LOG_LIMITS[metric])
    else:
        ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(ylabel)
    ax.set_xticks([31, 32, 33, 34, 35])


def figure_fixed_threshold():
    configs = list(CONFIG_STYLE)
    offsets = np.linspace(-0.2, 0.2, len(configs))
    fig, axes = plt.subplots(
        len(REFERENCE_THRESHOLDS), 3, figsize=(16, 9.5), sharex=True
    )
    for row, t in enumerate(REFERENCE_THRESHOLDS):
        for col, (metric, ylabel, log) in enumerate(PANELS):
            ax = axes[row, col]
            for config, offset in zip(configs, offsets):
                draw(
                    ax,
                    metrics(config, "independent bg", t),
                    metric,
                    offset=offset,
                    label=config_label(config),
                    **CONFIG_STYLE[config],
                )
            decorate(ax, log, ylabel, metric)
            ax.set_title(f"threshold {t}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(SB_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "fixed_threshold_comparison.png", dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)


def figure_threshold_sensitivity():
    configs = list(CONFIG_STYLE)
    cmap = plt.get_cmap("viridis")
    for metric, ylabel, log in (PANELS[0], PANELS[2]):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, config in zip(axes.ravel(), configs):
            for j, t in enumerate(SENSITIVITY_THRESHOLDS):
                style = {
                    "color": cmap(j / (len(SENSITIVITY_THRESHOLDS) - 1)),
                    "marker": "o",
                    "ls": "-",
                }
                draw(
                    ax,
                    metrics(config, "independent bg", t),
                    metric,
                    label=f"threshold {t}",
                    ms=4,
                    lw=1.4,
                    **style,
                )
            decorate(ax, log, ylabel, metric)
            ax.set_title(
                config_label(config), fontsize=10, color=CONFIG_STYLE[config]["color"]
            )
        for ax in axes[-1]:
            ax.set_xlabel(SB_LABEL)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout()
        name = "completeness" if metric == "C" else "contrast"
        fig.savefig(
            FIGURES / f"threshold_sensitivity_{name}.png", dpi=DPI, bbox_inches="tight"
        )
        plt.close(fig)


def figure_background_impact(t=0.5):
    configs = [SELECTED, ("dice", 0.05, 2)]
    fig, axes = plt.subplots(len(configs), 3, figsize=(15, 8), sharex=True)
    for row, config in enumerate(configs):
        for col, (metric, ylabel, log) in enumerate(PANELS):
            ax = axes[row, col]
            for background, style in (
                ("training bg", {"ls": "-", "marker": "o"}),
                ("independent bg", {"ls": "--", "marker": "s"}),
            ):
                draw(
                    ax,
                    metrics(config, background, t),
                    metric,
                    label=background,
                    offset=-0.06 if background == "training bg" else 0.06,
                    color=CONFIG_STYLE[config]["color"]
                    if background == "training bg"
                    else "0.35",
                    **style,
                )
            decorate(ax, log, ylabel, metric)
            ax.set_title(f"{config_label(config)}, threshold {t}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(SB_LABEL)
    for row in range(len(configs)):
        axes[row, 0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "background_impact.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_selected(t=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    k = threshold_index(t)
    sub = COUNTS[(COUNTS.config == SELECTED) & (COUNTS.background == "independent bg")]
    for ax, (metric, ylabel, log) in zip(axes, PANELS):
        for i, (seed, g) in enumerate(sub.groupby("seed")):
            g = g.sort_values("sb")
            C = np.array([r.S_s[k] / r.S_t for r in g.itertuples()])
            F = np.array([max(r.B_s[k], 1.0) / r.B_t for r in g.itertuples()])
            contrast = np.where(C > 0, C / F, np.nan)  # undefined, not zero
            y = {"C": C, "F": F, "contrast": contrast}[metric]
            ax.plot(
                g.sb,
                y,
                color=CONFIG_STYLE[SELECTED]["color"],
                lw=0.9,
                alpha=0.35,
                label="individual seeds" if i == 0 else None,
            )
        draw(
            ax,
            metrics(SELECTED, "independent bg", t),
            metric,
            label="mean over 4 seeds",
            lw=2.4,
            ms=7,
            **CONFIG_STYLE[SELECTED],
        )
        decorate(ax, log, ylabel, metric)
        ax.set_xlabel(SB_LABEL)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(
        f"Selected model: {config_label(SELECTED)}, threshold {t} (independent background)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_model.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def table(t):
    rows = []
    for config in CONFIG_STYLE:
        m = metrics(config, "independent bg", t).set_index("sb")
        collapsed = int(
            sum(
                r.S_s[threshold_index(t)] == 0 and r.B_s[threshold_index(t)] == 0
                for r in COUNTS[
                    (COUNTS.config == config)
                    & (COUNTS.background == "independent bg")
                    & (COUNTS.sb == 31.0)
                ].itertuples()
            )
        )
        rows.append(
            dict(
                configuration=config_label(config),
                **{f"C@{sb:.0f}": m.loc[sb, "C"] for sb in (32.0, 33.0)},
                **{f"F@{sb:.0f}": m.loc[sb, "F"] for sb in (32.0, 33.0)},
                **{f"C/F@{sb:.0f}": m.loc[sb, "contrast"] for sb in (32.0, 33.0)},
                seeds_flagging_nothing=collapsed,
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    figure_fixed_threshold()
    figure_threshold_sensitivity()
    figure_background_impact()
    figure_selected()
    pd.set_option("display.width", 250)
    for t in REFERENCE_THRESHOLDS:
        print(f"== threshold {t}, independent background")
        print(table(t).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("figures in", FIGURES)
