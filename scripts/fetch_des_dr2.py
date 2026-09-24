"""Download DES Y6 Gold stars from NOIRLab Astro Data Lab, one declination
strip at a time.

`des_dr2.y6_gold` is the Y6 Gold value-added catalogue: 343 columns, with the
quality flags, the fitted photometry and the several star/galaxy classifiers
that `des_dr2.main` lacks. It does not appear in the schema listing but it is
queryable. Gold is used here rather than `main` because of `flags_foreground`
(bright stars, nearby galaxies -- exactly the things that fake a localized
overdensity) and because `main`'s coadd classifier degrades at the faint end.

Stars are `0 <= EXT_XGB <= 1`, because that is the selection streamobs's DES
Y6 model reproduces (`config/surveys/des_yr6.yaml`: its stellar completeness
and galaxy-misclassification curves are both derived for that cut). Using any
other classifier would make the real catalogue and the injected streams two
different samples. It matters: the union of ext_coadd and ext_mash, which
looks more generous, misses 14.6% of all EXT_XGB stars.

Photometry is `psf_mag_aper_8_*_corrected`: PSF magnitudes, right for point
sources, with the extinction already applied (the raw magnitude minus
`a_fiducial_*`, whose ratio to `ebv_sfd98` is the DES coefficient 3.186 in g).
Magnitudes use a -9999000000 sentinel rather than null, which the magnitude
range in the selection removes.

A single full-footprint query is refused ("Query exceeded the maximum
execution time"), so this splits the sky into declination strips and writes
one parquet file per strip. Strips that time out are bisected and retried, so
a dense strip costs more queries rather than failing. Finished strips are
skipped, which makes the whole thing restartable.

What comes back, per row: `ra`, `dec`, dereddened `g` and `r`, and their
magnitude errors. The errors are kept deliberately -- the signal-to-noise cut
belongs downstream, in notebooks/real_des_data.ipynb, where its effect on the
depth can be seen before it is chosen.

Run from the repository root:
  python scripts/fetch_des_dr2.py --out ~/Documents/data/DES_yr6 [--count]
      [--manifest]
      [--dec-min -70] [--dec-max 5] [--step 1.0] [--snr-floor 2]
"""

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd

TABLE = "des_dr2.y6_gold"
# Cuts applied server side, so the download holds only what could ever be
# used. They are deliberately looser than the analysis cuts: the magnitude
# clip (16 to 24.5) and the real signal-to-noise threshold are applied in the
# notebook, where their effect is visible.
COLUMNS = [
    "ra",
    "dec",
    "psf_mag_aper_8_g_corrected AS g",
    "psf_mag_aper_8_r_corrected AS r",
    "psf_mag_err_aper_8_g AS g_err",
    "psf_mag_err_aper_8_r AS r_err",
    # The other two classifiers come along for diagnostics only; the sample
    # is defined by ext_xgb, and they disagree strongly (at g 24-25.5,
    # ext_coadd calls 12.9% of objects stars where ext_mash calls 3.5%).
    "ext_coadd",
    "ext_mash",
    "ext_xgb",
]
BRIGHT, FAINT = 15.5, 25.0
MIN_STRIP = 0.05  # degrees; below this, give up rather than bisect further


# Which objects count as stars. "xgb" is the streamobs-consistent default;
# "union" is broader but is NOT a superset -- it misses 14.6% of EXT_XGB
# stars -- so it exists for comparison, not as a safer choice.
CLASSIFIERS = {
    "xgb": "ext_xgb BETWEEN 0 AND 1",
    "union": (
        "(ext_xgb BETWEEN 0 AND 1 OR ext_coadd BETWEEN 0 AND 1"
        " OR ext_mash BETWEEN 0 AND 1)"
    ),
}


def where_clause(dec_low, dec_high, max_magerr, classifier="xgb"):
    """The selection, minus the columns: stars, clean pixels, usable photometry."""
    return f"""
        flags_footprint = 1
        AND flags_foreground = 0
        AND flags_gold = 0
        AND {CLASSIFIERS[classifier]}
        AND psf_mag_err_aper_8_g < {max_magerr}
        AND psf_mag_err_aper_8_r < {max_magerr}
        AND psf_mag_aper_8_g_corrected BETWEEN {BRIGHT} AND {FAINT}
        AND psf_mag_aper_8_r_corrected BETWEEN {BRIGHT} AND {FAINT}
        AND dec >= {dec_low} AND dec < {dec_high}
    """


def run(query_client, sql):
    """One query, as a DataFrame. Raises whatever Data Lab raises."""
    return pd.read_csv(StringIO(query_client.query(sql=sql, fmt="csv")))


def compact(frame):
    """Magnitudes as float32, positions left alone.

    float32 holds about seven significant digits, which is a thousandth of a
    magnitude but only a tenth of an arcsecond on a right ascension near 360 --
    fine for the magnitudes, not worth risking on the coordinates that decide
    which HEALPix pixel a star lands in.
    """
    classifiers = [c for c in frame.columns if c.startswith("ext_")]
    magnitudes = [c for c in frame.columns if c not in ("ra", "dec", *classifiers)]
    types = {c: "float32" for c in magnitudes}
    types.update({c: "int16" for c in classifiers})
    return frame.astype(types)


COLUMN_NOTES = {
    "ra": "right ascension, degrees (J2000)",
    "dec": "declination, degrees (J2000)",
    "g": "psf_mag_aper_8_g_corrected: PSF magnitude, extinction applied",
    "r": "psf_mag_aper_8_r_corrected: PSF magnitude, extinction applied",
    "g_err": "psf_mag_err_aper_8_g; SNR = 1.0857 / err, equal to "
    "psf_flux_s2n_aper_8_g to three decimals",
    "r_err": "psf_mag_err_aper_8_r; same convention",
    "ext_coadd": "coadd classifier: 0 high-confidence star, 1 candidate star, "
    "2-3 galaxies, -9 no data. Degrades faintward",
    "ext_mash": "MASH classifier, 0-1 stars, 2 ambiguous, 3-4 galaxies. The "
    "more reliable one at faint magnitudes",
    "ext_xgb": "boosted-tree classifier, same 0-4 convention",
}


def write_manifest(out_dir, dec_min, dec_max, step, snr_floor, classifier="xgb"):
    """Record what was downloaded, beside the files themselves.

    A directory of parquet files says nothing about which table it came from,
    which cuts are already applied, or when -- and those decide whether an
    analysis is valid. Anything reading this data should read this first.
    """
    files = sorted(out_dir.glob("des_y6gold_dec*.parquet"))
    per_file = []
    for path in files:
        frame = pd.read_parquet(path, columns=["dec"])
        per_file.append(
            {
                "file": path.name,
                "rows": len(frame),
                "dec_min": round(float(frame["dec"].min()), 4),
                "dec_max": round(float(frame["dec"].max()), 4),
                "bytes": path.stat().st_size,
            }
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None

    manifest = {
        "description": (
            "DES DR2 point sources for the streamgoggles stream search. One "
            "parquet file per one-degree declination strip; concatenate them "
            "for the full footprint."
        ),
        "source": {
            "service": "NOIRLab Astro Data Lab (datalab.noirlab.edu)",
            "table": TABLE,
            "table_rows_total": 691_483_608,
            "release": (
                "DES Y6 Gold, the value-added catalogue: quality flags, fitted "
                "photometry and several star/galaxy classifiers. Not listed in "
                "the Data Lab schema browser, but queryable."
            ),
            "downloaded_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "fetched_by": "scripts/fetch_des_dr2.py",
            "git_commit": commit,
        },
        "selection": {
            "sql_where": " ".join(
                where_clause(dec_min, dec_max, 1.0857 / snr_floor, classifier).split()
            ),
            "star_galaxy": (
                f"{CLASSIFIERS[classifier]} -- 0 <= EXT_XGB <= 1 is the "
                "selection streamobs's DES Y6 model reproduces, so the real "
                "catalogue and the injected streams are the same sample. "
                "ext_coadd and ext_mash are stored for diagnostics but do not "
                "define the sample; the union of those two misses 14.6% of "
                "EXT_XGB stars."
            ),
            "quality": (
                "flags_footprint = 1, flags_foreground = 0, flags_gold = 0. "
                "flags_foreground is the reason to prefer Gold: it removes "
                "bright-star haloes and nearby galaxies, which are what fake "
                "a localized overdensity."
            ),
            "signal_to_noise": (
                f"> {snr_floor:g} in both g and r (a floor, not the analysis"
                " cut: choose that downstream, the errors are kept)"
            ),
            "magnitudes": (
                f"{BRIGHT} to {FAINT} dereddened in both bands, wider than the"
                " pipeline's 16 to 24.5 clip on purpose"
            ),
            "extinction": (
                "already applied: the _corrected PSF magnitudes carry it"
                " (a_fiducial / ebv_sfd98 = 3.186 in g, the DES coefficient)"
            ),
            "not_applied": [
                "the 16 to 24.5 magnitude clip the models assume",
                "the final signal-to-noise cut",
            ],
        },
        "columns": COLUMN_NOTES,
        "dtypes": {
            "ra": "float64",
            "dec": "float64",
            "magnitudes and errors": "float32",
            "classifiers": "int16",
        },
        "coverage": {
            "dec_min": dec_min,
            "dec_max": dec_max,
            "strip_degrees": step,
            "ra": "unrestricted (the full footprint at each declination)",
        },
        "totals": {
            "files": len(per_file),
            "rows": sum(f["rows"] for f in per_file),
            "bytes": sum(f["bytes"] for f in per_file),
        },
        "files": per_file,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"manifest: {manifest['totals']['rows']:,} stars in "
        f"{manifest['totals']['files']} files "
        f"({manifest['totals']['bytes'] / 1e9:.2f} GB) -> {path}"
    )
    return manifest


def fetch_strip(query_client, dec_low, dec_high, max_magerr, classifier, depth=0):
    """One declination strip, bisected as many times as the server needs.

    A strip that exceeds the execution limit is split in two and each half
    retried, so the number of queries follows the density on the sky rather
    than being guessed in advance.
    """
    sql = (
        f"SELECT {', '.join(COLUMNS)} FROM {TABLE} "
        f"WHERE {where_clause(dec_low, dec_high, max_magerr, classifier)}"
    )
    try:
        return run(query_client, sql)
    except Exception as error:
        width = dec_high - dec_low
        if width <= MIN_STRIP:
            raise RuntimeError(
                f"dec {dec_low:+.3f} to {dec_high:+.3f} fails even at "
                f"{width:g} degrees: {error}"
            ) from error
        middle = (dec_low + dec_high) / 2
        print(
            f"    dec {dec_low:+.2f} to {dec_high:+.2f} refused "
            f"({str(error)[:60].strip()}); splitting",
            flush=True,
        )
        halves = [
            fetch_strip(
                query_client, dec_low, middle, max_magerr, classifier, depth + 1
            ),
            fetch_strip(
                query_client, middle, dec_high, max_magerr, classifier, depth + 1
            ),
        ]
        return pd.concat(halves, ignore_index=True)


def count_rows(query_client, dec_min, dec_max, max_magerr, classifier="xgb"):
    """How many rows the whole selection holds, before downloading any."""
    sql = (
        f"SELECT count(*) AS n FROM {TABLE} "
        f"WHERE {where_clause(dec_min, dec_max, max_magerr, classifier)}"
    )
    return int(run(query_client, sql)["n"].iloc[0])


def main(
    out_dir,
    dec_min,
    dec_max,
    step,
    snr_floor,
    count_only,
    manifest_only=False,
    classifier="xgb",
):
    from dl import authClient as ac
    from dl import queryClient as qc

    ac.login("anonymous")
    max_magerr = 1.0857 / snr_floor

    if manifest_only:
        write_manifest(
            Path(out_dir).expanduser(), dec_min, dec_max, step, snr_floor, classifier
        )
        return

    if count_only:
        start = time.time()
        total = count_rows(qc, dec_min, dec_max, max_magerr, classifier)
        print(
            f"{total:,} rows over dec {dec_min:+g} to {dec_max:+g} "
            f"at SNR > {snr_floor:g} ({time.time() - start:.0f}s)"
        )
        return

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # round, not int: (dec_max - dec_min) / step lands just under the integer
    # often enough that truncating drops the last strip, or every strip.
    edges = [dec_min + step * i for i in range(round((dec_max - dec_min) / step))]
    kept = 0
    for low in edges:
        high = low + step
        path = out_dir / f"des_y6gold_dec{low:+06.1f}.parquet"
        if path.exists():
            kept += len(pd.read_parquet(path, columns=["ra"]))
            print(f"dec {low:+.1f}: already downloaded", flush=True)
            continue
        start = time.time()
        frame = compact(fetch_strip(qc, low, high, max_magerr, classifier))
        frame.to_parquet(path, index=False)
        kept += len(frame)
        print(
            f"dec {low:+.1f} to {high:+.1f}: {len(frame):>9,} stars "
            f"in {time.time() - start:5.0f}s  ->  {path.name}",
            flush=True,
        )
    print(f"\n{kept:,} stars in {out_dir}")
    write_manifest(out_dir, dec_min, dec_max, step, snr_floor, classifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="~/Documents/data/DES_yr6")
    parser.add_argument("--dec-min", type=float, default=-70.0)
    parser.add_argument("--dec-max", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument(
        "--snr-floor",
        type=float,
        default=2.0,
        help="loosest signal-to-noise kept; the analysis cut is applied later",
    )
    parser.add_argument(
        "--count", action="store_true", help="count rows, download none"
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="rewrite manifest.json from the files already downloaded",
    )
    parser.add_argument(
        "--classifier",
        choices=list(CLASSIFIERS),
        default="xgb",
        help="which objects count as stars (default: the streamobs selection)",
    )
    arguments = parser.parse_args()
    main(
        arguments.out,
        arguments.dec_min,
        arguments.dec_max,
        arguments.step,
        arguments.snr_floor,
        arguments.count,
        arguments.manifest,
        arguments.classifier,
    )
