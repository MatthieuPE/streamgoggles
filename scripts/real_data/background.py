"""The real DES Y6 background: its cuts, its mask, and the figures documenting it.

Reads the declination strips fetched by scripts/fetch_des_dr2.py, applies the
analysis cuts, masks known streams and catalogued objects with
`streamgoggles.objects_overlap.build_background_mask`, and writes

  data/others/mask_des_yr6_background_nside512.fits.gz
                    the usable sky, which training and evaluation both read
                    (`get_footprint("des_yr6_background")`)
  docs/source/narrative/figures/real_background/
                    every figure of docs/source/narrative/real_des_background.md,
                    plus the two tables it includes
  <data>/des_yr6_background.parquet and .json
                    the background itself, with --write

Run from the repository root:
  python scripts/real_data/background.py [--data ~/Documents/data/DES_yr6] [--write]
"""

import argparse
import json
import warnings
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skyproj
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from streamgoggles.objects_overlap import (
    DATA as CATALOGUES,
)
from streamgoggles.objects_overlap import (
    DES2018_STREAM_WIDTHS,
    SAGITTARIUS_WIDTH,
    build_background_mask,
    mask_radius,
    stream_tracks,
)

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "docs" / "source" / "narrative" / "figures" / "real_background"
MASK_FILE = CATALOGUES / "mask_des_yr6_background_nside512.fits.gz"

# The analysis cuts, on top of what the download already applied. SNR 5 is
# where streamobs anchors its DES Y6 depth (the magnitude at which the
# photometric scatter reaches 0.2171 mag), so the real catalogue and the
# simulated one are cut at the same point. The clip is what the trained
# models assume; 16 is also streamobs's saturation limit.
SNR_MIN = 5.0
CLIP = (16.0, 24.5)
NSIDE = 512  # the pipeline's resolution, for the mask
PLOT_NSIDE = 256  # coarser, for maps a reader can see

# Mask parameters -- the defaults of build_background_mask, spelled out so the
# page can quote them and a change here is a change there.
MASK = {
    "width_factor": 3.0,
    "wide_stream_deg": 2.0,
    "wide_stream_factor": 1.0,
    "exclude_tracks": (),
    "object_radius_factor": 5.0,
    "object_min_radius_deg": 0.05,
}

# Colour, by job. Density is magnitude: one hue, light to dark, the same scale
# on every map so the before and after panels compare directly. Identity uses
# the first three slots of the reference categorical palette, the ones that
# stay distinguishable in every pairing, and never on text.
DENSITY_CMAP = "Greys"
STREAM_COLOUR = "#2a78d6"
DWARF_COLOUR = "#eb6834"
CLUSTER_COLOUR = "#2a78d6"
KEPT_COLOUR = "#d9d9d6"
MASKED_STREAM_COLOUR = "#eb6834"
MASKED_OBJECT_COLOUR = "#2a78d6"
TEXT = "#3a3a38"
MUTED = "#6f6e69"


def load(data):
    """The downloaded catalogue, with the columns this needs."""
    columns = ["ra", "dec", "g", "r", "g_err", "r_err"]
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in sorted(Path(data).glob("des_y6gold_dec*.parquet"))
    ]
    catalogue = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return catalogue.astype({"ra": "float64", "dec": "float64"})


def analysis_cut(catalogue):
    """Signal-to-noise and the magnitude clip, and what each keeps alone."""
    max_error = 1.0857 / SNR_MIN
    snr = (catalogue.g_err < max_error) & (catalogue.r_err < max_error)
    clip = catalogue.g.between(*CLIP) & catalogue.r.between(*CLIP)
    return snr & clip, {"snr": float(snr.mean()), "clip": float(clip.mean())}


def counts_map(ra, dec, nside=PLOT_NSIDE):
    """Stars per pixel, with empty pixels marked unseen so they draw blank."""
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True)
    counts = np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(float)
    counts[counts == 0] = hp.UNSEEN
    return counts


def des_map(counts, scale, title, ax=None):
    """One density map of the DES footprint, on the shared scale."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxmap(
        counts, zoom=False, cmap=DENSITY_CMAP, vmin=scale[0], vmax=scale[1], norm="log"
    )
    sp.draw_colorbar(label=f"stars per pixel (nside {PLOT_NSIDE})", fontsize=9)
    # skyproj draws on its own axes; a title set on the one passed in is lost.
    sp.ax.set_title(title, fontsize=11, color=TEXT, pad=24)
    return sp


HALO = [patheffects.withStroke(linewidth=3, foreground="white")]


def angular_distance(ra1, dec1, ra2, dec2):
    """Great-circle distance in degrees."""
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cosine = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def place_labels(ax, items, min_separation):
    """Label each item at the first of its candidate spots not already taken.

    items: (text, [(ra, dec), ...]) in priority order. A label with no free
    spot is left out rather than drawn over another -- the tables beside the
    figures carry every name anyway.
    """
    taken = []
    for text, spots in items:
        for ra, dec in spots:
            if all(angular_distance(ra, dec, r, d) >= min_separation for r, d in taken):
                ax.annotate(
                    text,
                    (float(ra), float(dec)),
                    fontsize=7.5,
                    color=TEXT,
                    xytext=(4, 4),
                    textcoords="offset points",
                    path_effects=HALO,
                    zorder=6,
                )
                taken.append((ra, dec))
                break


def split_on_wrap(ra, dec):
    """Break a track wherever it crosses RA 0/360, so no line spans the map."""
    jumps = np.flatnonzero(np.abs(np.diff(ra)) > 180) + 1
    return zip(np.split(ra, jumps), np.split(dec, jumps), strict=True)


def figure_data(catalogue, scale):
    """The downloaded catalogue: where it is, and what it looks like."""
    counts = counts_map(catalogue.ra.to_numpy(), catalogue.dec.to_numpy())
    des_map(counts, scale, f"DES Y6 Gold stars as downloaded: {len(catalogue):,}")
    plt.savefig(FIGURES / "density.png", dpi=110, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colour = catalogue.g - catalogue.r
    image = axes[0].hexbin(
        colour,
        catalogue.g,
        gridsize=260,
        extent=(-0.5, 2.0, 15.5, 25.0),
        bins="log",
        cmap="Greys",
        mincnt=1,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("g − r")
    axes[0].set_ylabel("g")
    axes[0].set_title("colour-magnitude diagram", fontsize=11, color=TEXT)
    fig.colorbar(image, ax=axes[0], label="stars per bin (log)")
    for band, series in (("g", "#2a78d6"), ("r", "#eb6834")):
        axes[1].hist(
            catalogue[band],
            bins=190,
            range=(15.5, 25.0),
            histtype="step",
            lw=1.6,
            color=series,
            label=band,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("magnitude (dereddened PSF)")
    axes[1].set_ylabel("stars per 0.05 mag")
    axes[1].set_title("magnitude distribution", fontsize=11, color=TEXT)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "cmd.png", dpi=110, bbox_inches="tight")
    plt.close()


def figure_signal_to_noise(catalogue):
    """Where the signal-to-noise cut and the clip each bite."""
    max_error = 1.0857 / SNR_MIN
    _, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, band in zip(axes, ["g", "r"], strict=True):
        ax.hexbin(
            catalogue[band],
            catalogue[f"{band}_err"],
            gridsize=220,
            extent=(15.5, 25.0, 0, 0.55),
            bins="log",
            cmap="Greys",
            mincnt=1,
        )
        ax.axhline(max_error, color=MUTED, lw=1.2, ls="--")
        ax.text(15.7, max_error + 0.012, f"SNR {SNR_MIN:g}", color=TEXT, fontsize=9)
        for edge in CLIP:
            ax.axvline(edge, color=MUTED, lw=1.2, ls=":")
        ax.text(CLIP[1] - 0.1, 0.5, "clip", color=TEXT, fontsize=9, ha="right")
        ax.set_xlabel(band)
        ax.set_title(f"{band}-band magnitude error", fontsize=11, color=TEXT)
    axes[0].set_ylabel("magnitude error")
    plt.tight_layout()
    plt.savefig(FIGURES / "signal_to_noise.png", dpi=110, bbox_inches="tight")
    plt.close()


def figure_streams(cut, scale, masks):
    """The tracks over the data, and the data once they are masked."""
    counts = counts_map(cut.ra.to_numpy(), cut.dec.to_numpy())
    sp = des_map(counts, scale, "known streams crossing the footprint")
    footprint = masks["footprint"]
    widths = {**DES2018_STREAM_WIDTHS, "Sagittarius": SAGITTARIUS_WIDTH}
    labels = []
    for name in widths:
        spots = []
        for ra, dec in stream_tracks(name, MASK["exclude_tracks"]).values():
            # skyproj's axes take lon/lat as data coordinates directly.
            for piece_ra, piece_dec in split_on_wrap(ra, dec):
                sp.ax.plot(piece_ra, piece_dec, color=STREAM_COLOUR, lw=1.3)
            # Candidate label spots along the part inside the survey -- not
            # the track's middle, which for Sagittarius is far outside it.
            inside = np.flatnonzero(footprint[hp.ang2pix(NSIDE, ra, dec, lonlat=True)])
            for fraction in (0.5, 0.3, 0.7, 0.15, 0.85, 0.05, 0.95):
                if inside.size:
                    at = inside[int(fraction * (inside.size - 1))]
                    spots.append((ra[at], dec[at]))
        labels.append((name, spots))
    # Five of these streams overlap on the sky near RA 325, Dec -55, so the
    # labels need placing, not just drawing.
    place_labels(sp.ax, labels, min_separation=4.0)
    plt.savefig(FIGURES / "streams.png", dpi=110, bbox_inches="tight")
    plt.close()

    pixels = hp.ang2pix(NSIDE, cut.ra.to_numpy(), cut.dec.to_numpy(), lonlat=True)
    kept = ~masks["streams"][pixels]
    counts = counts_map(cut.ra.to_numpy()[kept], cut.dec.to_numpy()[kept])
    des_map(counts, scale, f"after the stream mask: {kept.sum():,} stars")
    plt.savefig(FIGURES / "streams_masked.png", dpi=110, bbox_inches="tight")
    plt.close()


def object_table(masks):
    """Every masked object with its position and the radius masked."""
    rows = []
    for kind, (catalogue, inside, radii) in (
        ("dwarf galaxy", masks["dwarfs"]),
        ("globular cluster", masks["clusters"]),
    ):
        for row, radius in zip(catalogue[inside], radii, strict=True):
            rhalf = float(row["rhalf"])
            rows.append(
                {
                    "name": str(row["name"]),
                    "kind": kind,
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "rhalf_arcmin": rhalf if np.isfinite(rhalf) else np.nan,
                    "radius_deg": float(radius),
                }
            )
    return pd.DataFrame(rows).sort_values("radius_deg", ascending=False)


def figure_objects(cut, scale, objects):
    """Dwarfs and clusters over the data, big enough to find."""
    counts = counts_map(cut.ra.to_numpy(), cut.dec.to_numpy())
    sp = des_map(counts, scale, "catalogued objects in the footprint")
    for kind, plural, colour, marker in (
        ("dwarf galaxy", "dwarf galaxies", DWARF_COLOUR, "o"),
        ("globular cluster", "globular clusters", CLUSTER_COLOUR, "s"),
    ):
        rows = objects[objects.kind == kind]
        # plot rather than scatter: scatter does not render on skyproj axes.
        sp.ax.plot(
            rows.ra.to_numpy(),
            rows.dec.to_numpy(),
            ls="none",
            marker=marker,
            ms=10,
            mfc="none",
            mec=colour,
            mew=2.0,
            label=f"{plural} ({len(rows)})",
            zorder=5,
        )
    place_labels(
        sp.ax,
        [(row.name, [(row.ra, row.dec)]) for row in objects.head(10).itertuples()],
        min_separation=2.5,
    )
    sp.ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)
    plt.savefig(FIGURES / "objects.png", dpi=110, bbox_inches="tight")
    plt.close()


def figure_final(cut, scale, masks, kept):
    """The final mask, and the background it leaves."""
    categories = np.full(hp.nside2npix(NSIDE), hp.UNSEEN)
    footprint = masks["footprint"]
    # One category per pixel, streams last so they win where the two overlap:
    # the picture then partitions the footprint exactly as the legend does.
    categories[footprint] = 0
    categories[footprint & masks["objects"]] = 2
    categories[footprint & masks["streams"]] = 1
    categories = hp.ud_grade(categories, nside_out=PLOT_NSIDE, power=None)
    _, ax = plt.subplots(figsize=(9, 6.2))
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxmap(
        categories,
        zoom=False,
        cmap=ListedColormap([KEPT_COLOUR, MASKED_STREAM_COLOUR, MASKED_OBJECT_COLOUR]),
        vmin=-0.5,
        vmax=2.5,
    )
    area = hp.nside2pixarea(NSIDE, degrees=True)
    total = footprint.sum()
    usable = masks["usable"].sum()
    streams = (footprint & masks["streams"]).sum()
    objects = (footprint & masks["objects"] & ~masks["streams"]).sum()
    handles = [
        Patch(color=KEPT_COLOUR, label=f"background: {usable * area:,.0f} deg²"),
        Patch(
            color=MASKED_STREAM_COLOUR,
            label=f"known streams: {streams * area:,.0f} deg²",
        ),
        Patch(
            color=MASKED_OBJECT_COLOUR,
            label=f"clusters and dwarfs, outside the streams: {objects * area:,.1f} deg²",
        ),
    ]
    # Upper right, below the equatorial stripe, is the one empty corner.
    sp.ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.83),
        fontsize=9,
        framealpha=0.9,
    )
    sp.ax.set_title(
        f"the final mask: {100 * usable / total:.1f}% of {total * area:,.0f} deg² kept",
        fontsize=11,
        color=TEXT,
        pad=24,
    )
    plt.savefig(FIGURES / "final_mask.png", dpi=110, bbox_inches="tight")
    plt.close()

    counts = counts_map(cut.ra.to_numpy()[kept], cut.dec.to_numpy()[kept])
    des_map(counts, scale, f"the background used for injection: {kept.sum():,} stars")
    plt.savefig(FIGURES / "background.png", dpi=110, bbox_inches="tight")
    plt.close()


def stream_table(masks):
    """Each masked stream, its width, how far out it is masked, what it costs."""
    footprint = masks["footprint"]
    widths = {**DES2018_STREAM_WIDTHS, "Sagittarius": SAGITTARIUS_WIDTH}
    rows = []
    for name, width in widths.items():
        pixels = masks["stream_pixels"].get(name, np.array([], dtype=np.int64))
        cost = footprint[pixels].sum() / footprint.sum()
        references = sorted(stream_tracks(name, MASK["exclude_tracks"]))
        rows.append(
            {
                "stream": name,
                "width_deg": width,
                "radius_deg": mask_radius(
                    width,
                    MASK["width_factor"],
                    MASK["wide_stream_deg"],
                    MASK["wide_stream_factor"],
                ),
                "footprint_percent": 100 * cost,
                "tracks": ", ".join(references),
            }
        )
    return pd.DataFrame(rows).sort_values("footprint_percent", ascending=False)


def write_tables(streams, objects):
    """The two long tables, as Markdown the page includes, so they cannot drift."""
    lines = [
        "| stream | width (deg) | masked to (deg) | share of footprint | galstreams tracks |",
        "|---|---|---|---|---|",
    ]
    for row in streams.itertuples():
        lines.append(
            f"| {row.stream} | {row.width_deg:.2f} | {row.radius_deg:.2f} | "
            f"{row.footprint_percent:.1f}% | {row.tracks} |"
        )
    (FIGURES / "streams_table.md").write_text("\n".join(lines) + "\n")

    lines = [
        "| object | kind | RA | Dec | half-light radius (arcmin) | masked to (deg) |",
        "|---|---|---|---|---|---|",
    ]
    for row in objects.itertuples():
        rhalf = "—" if np.isnan(row.rhalf_arcmin) else f"{row.rhalf_arcmin:.1f}"
        lines.append(
            f"| {row.name} | {row.kind} | {row.ra:.2f} | {row.dec:.2f} | "
            f"{rhalf} | {row.radius_deg:.2f} |"
        )
    (FIGURES / "objects_table.md").write_text("\n".join(lines) + "\n")


def main(data, write):
    FIGURES.mkdir(parents=True, exist_ok=True)
    catalogue = load(data)
    keep, alone = analysis_cut(catalogue)
    cut = catalogue[keep].reset_index(drop=True)
    print(f"downloaded: {len(catalogue):,} stars")
    print(
        f"  SNR > {SNR_MIN:g} keeps {100 * alone['snr']:.2f}% alone, "
        f"the {CLIP[0]:g}-{CLIP[1]:g} clip {100 * alone['clip']:.2f}%"
    )
    print(f"  both: {len(cut):,} ({100 * keep.mean():.1f}%)")

    # One density scale for every map, from the downloaded catalogue, so a
    # masked region reads as missing rather than as a change of contrast.
    reference = counts_map(catalogue.ra.to_numpy(), catalogue.dec.to_numpy())
    seen = reference[reference != hp.UNSEEN]
    scale = (float(np.percentile(seen, 2)), float(np.percentile(seen, 98)))

    masks = build_background_mask(nside=NSIDE, **MASK)
    pixels = hp.ang2pix(NSIDE, cut.ra.to_numpy(), cut.dec.to_numpy(), lonlat=True)
    kept = masks["usable"][pixels]

    figure_data(catalogue, scale)
    figure_signal_to_noise(catalogue)
    figure_streams(cut, scale, masks)
    objects = object_table(masks)
    figure_objects(cut, scale, objects)
    figure_final(cut, scale, masks, kept)
    streams = stream_table(masks)
    write_tables(streams, objects)

    hp.write_map(
        MASK_FILE,
        masks["usable"].astype(np.uint8),
        dtype=np.uint8,
        overwrite=True,
        coord="C",
        column_names=["USABLE"],
    )

    area = hp.nside2pixarea(NSIDE, degrees=True)
    footprint = masks["footprint"]
    summary = {
        "footprint_deg2": float(footprint.sum() * area),
        "streams_deg2": float((footprint & masks["streams"]).sum() * area),
        "objects_deg2": float((footprint & masks["objects"]).sum() * area),
        "objects_outside_streams_deg2": float(
            (footprint & masks["objects"] & ~masks["streams"]).sum() * area
        ),
        "usable_deg2": float(masks["usable"].sum() * area),
        "usable_percent": float(100 * masks["usable"].sum() / footprint.sum()),
        "stars_after_cuts": len(cut),
        "stars_removed_by_streams": int(masks["streams"][pixels].sum()),
        "stars_removed_by_objects": int(masks["objects"][pixels].sum()),
        "stars_in_background": int(kept.sum()),
        "globular_clusters": int(masks["clusters"][1].sum()),
        "dwarf_galaxies": int(masks["dwarfs"][1].sum()),
    }
    print("\n" + json.dumps(summary, indent=2))
    print("\nstreams:\n" + streams.to_string(index=False))
    print(
        f"\nobjects: {len(objects)} (largest 8)\n"
        + objects.head(8).to_string(index=False)
    )
    print(f"\nmask -> {MASK_FILE}")

    if write:
        out = Path(data) / "des_yr6_background.parquet"
        pd.DataFrame(
            {
                "ra": cut.ra.to_numpy()[kept],
                "dec": cut.dec.to_numpy()[kept],
                "des_yr6_g_obs": cut.g.to_numpy()[kept],
                "des_yr6_r_obs": cut.r.to_numpy()[kept],
            }
        ).to_parquet(out, index=False)
        manifest = json.loads((Path(data) / "manifest.json").read_text())
        sidecar = {
            "description": (
                "Stream-free DES Y6 Gold background for the streamgoggles "
                "injection pipeline: known streams, globular clusters and "
                "dwarf galaxies masked, so an injected stream is the only "
                "stream present."
            ),
            "built_by": "scripts/real_data/background.py",
            "built_from": manifest["source"],
            "inherited_selection": manifest["selection"],
            "applied_here": {
                "signal_to_noise": f"> {SNR_MIN:g} in both bands",
                "magnitude_clip": list(CLIP),
                "mask": {**MASK, "exclude_tracks": list(MASK["exclude_tracks"])},
                "mask_file": str(MASK_FILE.relative_to(REPO)),
            },
            "streams": {
                row.stream: {
                    "width_deg": round(row.width_deg, 3),
                    "masked_to_deg": round(row.radius_deg, 3),
                    "footprint_percent": round(row.footprint_percent, 2),
                }
                for row in streams.itertuples()
            },
            "summary": summary,
            "columns": ["ra", "dec", "des_yr6_g_obs", "des_yr6_r_obs"],
            "rows": int(kept.sum()),
        }
        (Path(data) / "des_yr6_background.json").write_text(
            json.dumps(sidecar, indent=2) + "\n"
        )
        print(f"background -> {out} ({kept.sum():,} stars)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="~/Documents/data/DES_yr6")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    main(Path(arguments.data).expanduser(), arguments.write)
