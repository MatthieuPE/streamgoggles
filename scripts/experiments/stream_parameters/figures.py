"""Figures and numbers for docs/source/experiments/stream_parameters.md.

  detection.png   fraction of streams detected against distance modulus, one
                  panel per surface brightness, one colour per width, quick
                  (4800-window) and long (19200-window) models
  training_spread.png  one point per trained model, at surface brightness 34:
                  how much the same configuration varies from training to
                  training
  des_streams.png fraction of injections recovered for each DES 2018 stream,
                  with the range over the trainings
  isochrone.png   detection when the injected stream's age or metallicity is
                  not the one the matched filter assumes
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
# The current series uses the 13 Gyr filter of Shipp et al. (2018) and writes
# under this prefix; the 12 Gyr series owns the untagged names. Figures built
# from a results file that only exists untagged say so in their title.
TAG = "a13_"
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
    ax.set_title(
        "SB 34: one point per trained model (6 x 4800 windows, 12 Gyr filter)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "training_spread.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_des_streams():
    """Recovery of each DES 2018 stream, with the range over the trainings."""
    path = DATA / f"{TAG}des_results.pkl"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "sp_run", Path(__file__).parent / "run.py"
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    results = pd.read_pickle(path)
    matched = detection_at_false_alarm_rate(
        results, THRESHOLD_GRID, targets=(TARGET,), group_by=["eval_set", "seed"]
    )
    pooled = matched.groupby("eval_set")[["n_detected", "n_realizations"]].sum()
    spread = matched.groupby("eval_set")["detection_fraction"].agg(["min", "max"])
    table = pd.DataFrame(
        [
            {
                "stream": name,
                "width": width,
                "length": length,
                "distance_modulus": dm,
                "surface_brightness": sb,
                "fraction": pooled.loc[name, "n_detected"]
                / pooled.loc[name, "n_realizations"],
                "min": spread.loc[name, "min"],
                "max": spread.loc[name, "max"],
            }
            for name, (width, length, dm, sb) in run.DES_STREAMS.items()
        ]
    ).sort_values("fraction")

    ensemble_path = DATA / f"{TAG}des_results_ensemble.pkl"
    if ensemble_path.exists():
        ens = detection_at_false_alarm_rate(
            pd.read_pickle(ensemble_path),
            THRESHOLD_GRID,
            targets=(TARGET,),
            group_by=["eval_set", "seed"],
        ).set_index("eval_set")
        table["ensemble"] = [ens.loc[n, "detection_fraction"] for n in table["stream"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(table))
    ax.barh(y, table["fraction"], color="#6baed6", height=0.6)
    ax.hlines(y, table["min"], table["max"], color="#08306b", lw=2)
    ax.plot(table["min"], y, "|", color="#08306b", ms=9)
    ax.plot(table["max"], y, "|", color="#08306b", ms=9)
    if "ensemble" in table:
        ax.plot(
            table["ensemble"],
            y,
            "D",
            color="#cb181d",
            ms=6,
            label="the six averaged into one map (60 injections)",
        )
        ax.legend(fontsize=8.5, loc="lower left")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{row.stream}  (m-M {row.distance_modulus:g}, "
            f"{row.width:g} deg, SB {row.surface_brightness:g})"
            for row in table.itertuples()
        ],
        fontsize=8.5,
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("fraction of injections recovered at the stream's known position")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(
        "DES 2018 streams, simulated with their own parameters "
        "(13 Gyr filter)\n"
        "bars: mean of 6 trainings (20 injections each) — "
        "lines: lowest to highest training, not a confidence interval",
        fontsize=10.5,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "des_streams.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return table


def figure_isochrone():
    """Detection when the stream's population is not the filter's isochrone.

    The matched filter, the model's training and the label all assume the
    experiment's own population. Here the injected stream has another age or
    metallicity at the same surface brightness, so what changes is which of
    its stars the filter selects, not how bright the stream is.
    """
    path = DATA / f"{TAG}isochrone_results.pkl"
    if not path.exists():
        return None
    results = pd.read_pickle(path)
    matched = detection_at_false_alarm_rate(
        results, THRESHOLD_GRID, targets=(TARGET,), group_by=["eval_set", "seed"]
    )
    pooled = matched.groupby("eval_set")[["n_detected", "n_realizations"]].sum()
    rows = []
    for name, row in pooled.iterrows():
        age, z, model, width, sb = name.split("_")
        rows.append(
            {
                "age": float(age[3:]),
                "z": float(z[1:]),
                "model": model,
                "width": float(width[1:]),
                "richness": float(sb[2:]),
                "n_detected": row.n_detected,
                "n_realizations": row.n_realizations,
                "fraction": row.n_detected / row.n_realizations,
            }
        )
    table = pd.DataFrame(rows)
    # The scan runs in the filter's own family; the other family appears once,
    # at the filter's own values, to measure that systematic on its own.
    scanned = table[table.model == "Marigo2017"]
    own = scanned[(scanned.age == 13.0) & (scanned.z == 0.0002)]
    reference = dict(zip(own["width"], own["fraction"], strict=True))
    other = table[table.model != "Marigo2017"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    held = {"age": 0.0002, "z": 13.0}
    labels = {"age": "age (Gyr)", "z": "metallicity Z"}
    for ax, varied in zip(axes, ["age", "z"], strict=True):
        fixed = "z" if varied == "age" else "age"
        scan = scanned[scanned[fixed] == held[varied]]
        for width, group in scan.groupby("width"):
            group = group.sort_values(varied)
            k = group["n_detected"].to_numpy()
            n = group["n_realizations"].to_numpy()
            low, high = np.array(
                [_wilson_interval(int(a), int(b)) for a, b in zip(k, n, strict=True)]
            ).T
            y = k / n
            colour = WIDTH_COLOURS[width]
            ax.errorbar(
                group[varied],
                y,
                yerr=[np.clip(y - low, 0, None), np.clip(high - y, 0, None)],
                color=colour,
                marker="o",
                capsize=2.5,
                lw=1.6,
                label=f"width {width:g} deg",
            )
            if width in reference:
                ax.axhline(reference[width], color=colour, ls=":", lw=1.2)
        ax.axvline(13.0 if varied == "age" else 0.0002, color="0.5", ls="--", lw=1.0)
        for _, row in other.iterrows():
            ax.plot(
                row["age"] if varied == "age" else row["z"],
                row["fraction"],
                "x",
                color=WIDTH_COLOURS[row["width"]],
                ms=8,
                mew=2,
                label="Bressan2012, same values"
                if (varied == "age" and row["width"] == 0.2)
                else None,
            )
        if varied == "z":
            ax.set_xscale("log")
        ax.set_xlabel(labels[varied])
        ax.set_title(
            f"varying {varied}, at the filter's "
            + ("Z = 0.0002" if varied == "age" else "13 Gyr"),
            fontsize=10.5,
        )
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of streams detected")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Stream population against the filter's isochrone "
        "(13 Gyr, Z = 0.0002), SB 34, m-M 17\n"
        "dashed: the filter's own values — dotted: the rate there, per width",
        fontsize=10.5,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "isochrone.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return table


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
    des = figure_des_streams()
    if des is not None:
        print("\n== DES 2018 streams recovered (pooled over 6 trainings, range)")
        des = des.assign(
            recovered=[
                f"{100 * row.fraction:.0f}% ({100 * row.min:.0f}-{100 * row.max:.0f})"
                for row in des.itertuples()
            ]
        )
        print(
            des[
                [
                    "stream",
                    "width",
                    "length",
                    "distance_modulus",
                    "surface_brightness",
                    "recovered",
                ]
            ]
            .sort_values("recovered")
            .to_string(index=False)
        )
    isochrone = figure_isochrone()
    if isochrone is not None:
        print("\n== detected when the stream's population is not the filter's")
        print(
            isochrone.assign(percent=(100 * isochrone.fraction).round(0))
            .sort_values(["width", "model", "age", "z"])[
                ["width", "model", "age", "z", "percent", "n_realizations"]
            ]
            .to_string(index=False)
        )
    for name, label in (
        (f"{TAG}des_results_ensemble.pkl", "DES streams"),
        ("des_results_ensemble.pkl", "DES streams, 12 Gyr filter"),
    ):
        if (DATA / name).exists():
            ens = detection_at_false_alarm_rate(
                pd.read_pickle(DATA / name),
                THRESHOLD_GRID,
                targets=(TARGET,),
                group_by=["eval_set", "seed"],
            )
            print(f"\n== ensemble of the six trainings, {label}")
            print(
                ens.assign(percent=(100 * ens.detection_fraction).round(0))[
                    ["eval_set", "percent", "n_realizations"]
                ]
                .sort_values("percent")
                .to_string(index=False)
            )
    if contrast:
        print("\n== contrast in the matched filter (SB 33, width 0.2, length 15)")
        print(contrast_table().round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--contrast", action="store_true")
    main(parser.parse_args().contrast)
