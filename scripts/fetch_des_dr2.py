"""Download DES DR2 stars from NOIRLab Astro Data Lab, one declination strip
at a time.

DR2 is the public six-year release (691,483,608 objects in `des_dr2.main`).
The internal Y6 Gold value-added catalogue is not hosted at Data Lab, so
`des_dr2.main` is what there is; its `mag_auto_*_dered` columns already carry
the SFD98 extinction correction, so nothing needs dereddening afterwards.

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
      [--dec-min -70] [--dec-max 5] [--step 1.0] [--snr-floor 2]
"""

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd

TABLE = "des_dr2.main"
# Cuts applied server side, so the download holds only what could ever be
# used. They are deliberately looser than the analysis cuts: the magnitude
# clip (16 to 24.5) and the real signal-to-noise threshold are applied in the
# notebook, where their effect is visible.
COLUMNS = [
    "ra",
    "dec",
    "mag_auto_g_dered",
    "mag_auto_r_dered",
    "magerr_auto_g",
    "magerr_auto_r",
]
BRIGHT, FAINT = 15.5, 25.0
MIN_STRIP = 0.05  # degrees; below this, give up rather than bisect further


def where_clause(dec_low, dec_high, max_magerr):
    """The selection, minus the columns: stars, clean pixels, usable photometry."""
    return f"""
        extended_class_coadd BETWEEN 0 AND 1
        AND flags_g < 4 AND flags_r < 4
        AND imaflags_iso_g = 0 AND imaflags_iso_r = 0
        AND magerr_auto_g < {max_magerr} AND magerr_auto_r < {max_magerr}
        AND mag_auto_g_dered BETWEEN {BRIGHT} AND {FAINT}
        AND mag_auto_r_dered BETWEEN {BRIGHT} AND {FAINT}
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
    magnitudes = [c for c in frame.columns if c not in ("ra", "dec")]
    return frame.astype({c: "float32" for c in magnitudes})


def fetch_strip(query_client, dec_low, dec_high, max_magerr, depth=0):
    """One declination strip, bisected as many times as the server needs.

    A strip that exceeds the execution limit is split in two and each half
    retried, so the number of queries follows the density on the sky rather
    than being guessed in advance.
    """
    sql = (
        f"SELECT {', '.join(COLUMNS)} FROM {TABLE} "
        f"WHERE {where_clause(dec_low, dec_high, max_magerr)}"
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
            fetch_strip(query_client, dec_low, middle, max_magerr, depth + 1),
            fetch_strip(query_client, middle, dec_high, max_magerr, depth + 1),
        ]
        return pd.concat(halves, ignore_index=True)


def count_rows(query_client, dec_min, dec_max, max_magerr):
    """How many rows the whole selection holds, before downloading any."""
    sql = (
        f"SELECT count(*) AS n FROM {TABLE} "
        f"WHERE {where_clause(dec_min, dec_max, max_magerr)}"
    )
    return int(run(query_client, sql)["n"].iloc[0])


def main(out_dir, dec_min, dec_max, step, snr_floor, count_only):
    from dl import authClient as ac
    from dl import queryClient as qc

    ac.login("anonymous")
    max_magerr = 1.0857 / snr_floor

    if count_only:
        start = time.time()
        total = count_rows(qc, dec_min, dec_max, max_magerr)
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
        path = out_dir / f"des_dr2_dec{low:+06.1f}.parquet"
        if path.exists():
            kept += len(pd.read_parquet(path, columns=["ra"]))
            print(f"dec {low:+.1f}: already downloaded", flush=True)
            continue
        start = time.time()
        frame = compact(fetch_strip(qc, low, high, max_magerr))
        frame.to_parquet(path, index=False)
        kept += len(frame)
        print(
            f"dec {low:+.1f} to {high:+.1f}: {len(frame):>9,} stars "
            f"in {time.time() - start:5.0f}s  ->  {path.name}",
            flush=True,
        )
    print(f"\n{kept:,} stars in {out_dir}")


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
    arguments = parser.parse_args()
    main(
        arguments.out,
        arguments.dec_min,
        arguments.dec_max,
        arguments.step,
        arguments.snr_floor,
        arguments.count,
    )
