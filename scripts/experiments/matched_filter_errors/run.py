"""Matched filter on DES: which photometric errors, and how many of them.

The matched filter is an isochrone widened on each side by `error_multiplier`
times the photometric error at each magnitude. Called with no error model,
streamobs uses one fitted on LSST DC2. This compares four filters around the
same 13 Gyr, Z = 0.0002 isochrone:

  current      streamobs's default (LSST DC2) errors, 2 sigma
  DES 1 sigma  DES_YR6_ERROR_MODEL, error_multiplier [1, 1]
  DES 1.5 sigma                     [1.5, 1.5]
  DES 2 sigma                       [2, 2]

and measures, from m-M 15 to 19, the fraction of a simulated DES stream's own
stars each keeps and a counting figure of merit, stream stars kept over the
square root of background stars kept, on the real masked DES Y6 background.
The same is repeated for stream populations drawn as training draws them and
for the sky's densest and emptiest halves in galactic latitude.

Writes docs/source/experiments/figures/matched_filter_errors/*.png and
data/experiments/matched_filter_errors/results.csv.

Run from the repository root:
  python scripts/experiments/matched_filter_errors/run.py
      [--background ~/Documents/data/DES_yr6/des_yr6_background.parquet]
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "matched_filter_errors"
FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "matched_filter_errors"

ISOCHRONE = {"age": 13.0, "z": 0.0002}
DISTANCES = [15.0 + 0.5 * i for i in range(9)]
SHOWN_DISTANCES = [15.0, 17.0, 19.0]
MULTIPLIERS = [1.0, 1.5, 2.0]
CLIP = (16.0, 24.5)
# A bright stream, so each distance has thousands of stars to count: the
# fraction kept does not depend on the surface brightness, only its noise does.
STREAM = {"morphology": "uniform", "richness": 32.0, "width": 0.5, "length": 10.0}
N_REALIZATIONS = 3
# Populations drawn as the adopted training set draws them (stream-parameters
# experiment, conclusion 10), to check the ranking does not depend on the
# stream sharing the filter's isochrone.
N_POPULATIONS = 6
POPULATION_SEED = 7

# Colour by job: the current filter is one identity (blue); the DES filters
# are one quantity at three levels, so one hue from light to dark.
CURRENT_COLOUR = "#2a78d6"
DES_RAMP = plt.get_cmap("Oranges")
TEXT = "#3a3a38"


def filter_configs():
    """The four filters, by name, as a filters configuration."""
    from streamgoggles.matched_filter import DES_YR6_ERROR_MODEL

    base = {"type": "isochrone", "reference_isochrone": dict(ISOCHRONE)}
    configs = {"current (LSST errors, 2σ)": base}
    for multiplier in MULTIPLIERS:
        configs[f"DES errors, {multiplier:g}σ"] = {
            **base,
            "error_model": DES_YR6_ERROR_MODEL,
            "error_multiplier": [multiplier, multiplier],
        }
    return configs


def colour_of(name):
    if name.startswith("current"):
        return CURRENT_COLOUR
    multiplier = float(name.split(", ")[1].rstrip("σ"))
    return DES_RAMP(0.45 + 0.25 * MULTIPLIERS.index(multiplier))


def injector():
    """The stream-parameters experiment's injector: the training pipeline."""
    spec = importlib.util.spec_from_file_location(
        "sp_run", REPO / "scripts" / "experiments" / "stream_parameters" / "run.py"
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    _, stream_injector = run.build_sky(run.BACKGROUND_SEED)
    return stream_injector


def stream_stars(stream_injector, dm, age, z, seeds):
    """Observed stars of simulated streams, inside the analysis clip."""
    params = {**STREAM, "distance_modulus": dm, "age": age, "z": z}
    stars = pd.concat(
        [
            stream_injector._realize_and_inject(params, np.random.default_rng(seed))[0]
            for seed in seeds
        ],
        ignore_index=True,
    )
    g, r = stars["des_yr6_g_obs"], stars["des_yr6_r_obs"]
    return stars[g.between(*CLIP) & r.between(*CLIP)]


def galactic_latitude(ra, dec):
    """|b| in degrees."""
    rotator = hp.Rotator(coord=["C", "G"])
    _, b = rotator(ra, dec, lonlat=True)
    return np.abs(b)


def figure_error_fit(depth, fitted, wide):
    """The DES Y6 errors, the fit to them, and the models it replaces."""
    import streamobs as so
    from streamobs.match_filter import default_errors, default_errors_des2018

    from streamgoggles.matched_filter import error_model

    survey = so.surveys.Survey.load("des", release="yr6")
    mags = np.linspace(16.0, depth + 2.0, 100)
    sample = survey.get_photo_error("g", magnitude=mags, maglim=depth, kind="sample")
    catalog = survey.get_photo_error("g", magnitude=mags, maglim=depth, kind="catalog")

    fig, (ax, residual) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(
        mags,
        sample,
        "o",
        ms=3.5,
        color="#1a1a19",
        label="DES Y6 scatter (streamobs, fitted)",
    )
    ax.plot(
        mags,
        catalog,
        ls="--",
        lw=1.4,
        color="0.55",
        label="DES Y6 reported errors (not fitted)",
    )
    ax.plot(
        mags,
        error_model(mags, **fitted),
        lw=2,
        color=DES_RAMP(0.95),
        label="fit to g 24.5: DES_YR6_ERROR_MODEL",
    )
    ax.plot(
        mags,
        error_model(mags, **wide),
        lw=1.6,
        ls="--",
        color=DES_RAMP(0.6),
        label=f"fit to g {depth + 2:.1f} (not used)",
    )
    ax.plot(
        mags,
        error_model(mags, **default_errors),
        lw=2,
        color=CURRENT_COLOUR,
        label="streamobs default (LSST DC2)",
    )
    ax.plot(
        mags,
        error_model(mags, **default_errors_des2018),
        lw=1.4,
        ls=":",
        color="0.35",
        label="DES 2018 constants",
    )
    for x, text in ((CLIP[1], "clip 24.5"), (depth, f"depth {depth:.2f}")):
        ax.axvline(x, color="0.6", lw=1, ls=":")
        ax.text(
            x - 0.08,
            0.0015,
            text,
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=TEXT,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.0)
    ax.set_ylabel("g magnitude error")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    residual.plot(
        mags, error_model(mags, **fitted) / sample - 1, lw=2, color=DES_RAMP(0.95)
    )
    residual.plot(
        mags,
        error_model(mags, **wide) / sample - 1,
        lw=1.6,
        ls="--",
        color=DES_RAMP(0.6),
    )
    residual.axhline(0, color="0.6", lw=1)
    # Past the clip the filter never selects a star, so the fit is only asked
    # to follow the scatter up to it; beyond, it is shown greyed.
    for panel in (ax, residual):
        panel.axvspan(CLIP[1], mags.max() + 0.2, color="0.93", zorder=0)
    residual.text(
        CLIP[1] + 0.1,
        0.3,
        "beyond the clip:\nnot used",
        fontsize=8,
        color=TEXT,
        va="top",
    )
    residual.set_ylim(-0.4, 0.4)
    residual.set_xlim(mags.min() - 0.2, mags.max() + 0.2)
    residual.set_ylabel("fit / scatter − 1")
    residual.set_xlabel("g")
    residual.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_fit.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return float(
        np.max(np.abs(error_model(mags, **fitted) / sample - 1)[mags <= CLIP[1]])
    )


def figure_filters(filters, stars_by_dm):
    """The four filters over a simulated stream's own stars."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6), sharey=True)
    for ax, dm in zip(axes, SHOWN_DISTANCES, strict=True):
        stars = stars_by_dm[dm]
        ax.scatter(
            stars["des_yr6_g_obs"] - stars["des_yr6_r_obs"],
            stars["des_yr6_g_obs"],
            s=1,
            color="0.7",
            alpha=0.35,
            rasterized=True,
        )
        for name, matched_filter in filters.items():
            polygon = matched_filter._polygon(["g", "r"], dm)
            closed = np.vstack([polygon, polygon[:1]])
            kept = matched_filter.select(stars, ["g", "r"], dm).mean()
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=colour_of(name),
                lw=1.8,
                label=f"{name}: {100 * kept:.0f}% kept",
            )
        ax.set_title(f"m−M = {dm:g}", fontsize=11, color=TEXT)
        ax.set_xlabel("g − r")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(24.7, 16.5)
        ax.grid(alpha=0.25)
        # Each panel's empty corner, clear of the faint end being compared;
        # at m-M 19 the giant branch fills the upper right.
        ax.legend(
            fontsize=7.8,
            loc="upper left" if dm >= 19 else "upper right",
            framealpha=0.9,
        )
    axes[0].set_ylabel("g")
    fig.tight_layout()
    fig.savefig(FIGURES / "filters.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_curves(results):
    """Fraction of the stream kept, and relative S/sqrt(B), against distance."""
    own = results[results.population == "filter's own"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, group in own.groupby("filter", sort=False):
        group = group.sort_values("distance_modulus")
        style = {
            "color": colour_of(name),
            "lw": 2,
            "marker": "o",
            "ms": 5,
            "label": name,
        }
        left.plot(group.distance_modulus, 100 * group.kept, **style)
        right.plot(group.distance_modulus, group.relative_merit, **style)
    left.set_ylabel("stream stars kept (%)")
    left.set_ylim(40, 101)
    right.axhline(1.0, color="0.6", lw=1, ls=":")
    right.set_ylabel("S/√B, relative to the current filter")
    for ax, title in (
        (left, "how much of the stream the filter keeps"),
        (right, "stream stars kept / √(background stars kept)"),
    ):
        ax.set_xlabel("distance modulus")
        ax.set_xticks([15, 16, 17, 18, 19])
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=11, color=TEXT)
    left.legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "completeness_snr.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(background_path):
    from streamgoggles.matched_filter import (
        build_matched_filters,
        fit_survey_error_model,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    fitted, depth = fit_survey_error_model("des", "yr6")
    wide, _ = fit_survey_error_model("des", "yr6", max_magnitude=depth + 2.0)
    worst = figure_error_fit(depth, fitted, wide)
    print(
        f"fit: {fitted}  (depth {depth:.2f}; worst residual to g 24.5: {100 * worst:.1f}%)"
    )

    filters = build_matched_filters(filter_configs(), namespace="des_yr6")
    stream_injector = injector()

    background = pd.read_parquet(Path(background_path).expanduser())
    latitude = galactic_latitude(background.ra.to_numpy(), background.dec.to_numpy())
    dense = latitude < np.median(latitude)
    print(
        f"background: {len(background):,} stars; |b| median {np.median(latitude):.1f} deg"
    )

    rng = np.random.default_rng(POPULATION_SEED)
    populations = [
        (float(rng.uniform(9.0, 13.5)), float(10 ** rng.uniform(-4, -3)))
        for _ in range(N_POPULATIONS)
    ]

    rows, shown = [], {}
    for dm in DISTANCES:
        own = stream_stars(
            stream_injector, dm, ISOCHRONE["age"], ISOCHRONE["z"], range(N_REALIZATIONS)
        )
        drawn = pd.concat(
            [
                stream_stars(stream_injector, dm, age, z, [100 + i])
                for i, (age, z) in enumerate(populations)
            ],
            ignore_index=True,
        )
        if dm in SHOWN_DISTANCES:
            shown[dm] = own
        background_kept = {
            name: f.select(background, ["g", "r"], dm) for name, f in filters.items()
        }
        for label, stars in (("filter's own", own), ("drawn as in training", drawn)):
            for name, f in filters.items():
                kept = float(f.select(stars, ["g", "r"], dm).mean())
                selected = background_kept[name]
                rows.append(
                    {
                        "distance_modulus": dm,
                        "population": label,
                        "filter": name,
                        "kept": kept,
                        "stream_stars": len(stars),
                        "background_kept": int(selected.sum()),
                        "background_kept_dense": int(selected[dense].sum()),
                        "background_kept_sparse": int(selected[~dense].sum()),
                    }
                )
        print(f"m-M {dm:g}: done", flush=True)

    results = pd.DataFrame(rows)
    for column, suffix in (
        ("background_kept", ""),
        ("background_kept_dense", "_dense"),
        ("background_kept_sparse", "_sparse"),
    ):
        merit = results.kept / np.sqrt(results[column])
        reference = results[results["filter"].str.startswith("current")].set_index(
            ["distance_modulus", "population"]
        )
        reference_merit = reference.kept / np.sqrt(reference[column])
        results[f"relative_merit{suffix}"] = (
            merit.to_numpy()
            / reference_merit.loc[
                list(zip(results.distance_modulus, results.population, strict=True))
            ].to_numpy()
        )
    results.to_csv(OUT / "results.csv", index=False)

    figure_filters(filters, shown)
    figure_curves(results)

    pd.set_option("display.width", 200)
    for label in ("filter's own", "drawn as in training"):
        part = results[results.population == label]
        print(f"\n== stream stars kept (%), {label} population")
        print(
            (
                100
                * part.pivot(index="distance_modulus", columns="filter", values="kept")
            )
            .round(1)
            .to_string()
        )
        print(f"\n== relative S/sqrt(B), {label} population")
        print(
            part.pivot(
                index="distance_modulus", columns="filter", values="relative_merit"
            )
            .round(3)
            .to_string()
        )
    for suffix, text in (
        ("_dense", "dense half (low |b|)"),
        ("_sparse", "sparse half (high |b|)"),
    ):
        part = results[results.population == "filter's own"]
        print(f"\n== relative S/sqrt(B), {text}")
        print(
            part.pivot(
                index="distance_modulus",
                columns="filter",
                values=f"relative_merit{suffix}",
            )
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--background", default="~/Documents/data/DES_yr6/des_yr6_background.parquet"
    )
    main(parser.parse_args().background)
