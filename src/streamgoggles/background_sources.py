"""Pluggable background sources (real DP2, synthetic LSST yr1, etc).

Three implementations (decision 23):

1. DataFileBackgroundSource: real DP2 parquet (+ dust correction).
2. StreamObsLightBackgroundSource: streamobs's fast synthetic background
   (streamobs.background.background.Background(method="light")).
3. StreamObsCatalogueBackgroundSource: streamobs's fuller injection-based
   catalog generation (Background(method="injection") -- PLAN.md section 6.1.2
   corrected the original "catalogue" name guess to streamobs's real name).

All three produce the same downstream pipeline input (streamobs column convention),
so cuts/clipping/select/caching are identical regardless of source.

Rationale: Enables starting work against LSST yr1 forecast (no real data) before
switching to real DP2 once released. Same code path throughout.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Protocol

import astropy.units as u
import gala.coordinates as gc
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from streamobs.background.background import Background
from streamobs.columns import err_col, obs_col
from streamobs.surveys import Survey

from streamgoggles.data_preparation import deredden_dataframe

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StudyRegion:
    """Spatial region for synthetic background generation (decision 24).

    Attributes:
        center_ra: Region center RA (degrees). Fixed manually at 0.0 by
            default (2026-09-08 decision) rather than computed from a
            footprint centroid — the LSST yr1 forecast footprint has no
            unambiguous centroid (PLAN.md §6.1.5).
        center_dec: Region center Dec (degrees). Default -30.0, see above.
        width_deg: Region width in degrees (default 70).
        height_deg: Region height in degrees (default 20).

    Rationale: Restricts synthetic background generation to a region much smaller
    than LSST's enormous footprint, saving compute. Only used by light/catalogue
    sources; data_file sources ignore it (real catalog is used in full).
    """

    center_ra: float = 0.0
    center_dec: float = -30.0
    width_deg: float = 70.0
    height_deg: float = 20.0

    def is_inside(self, ra, dec):
        """Return boolean mask: stars inside this region.

        Parameters:
            ra, dec: RA and Dec arrays/series (degrees).

        Returns:
            Boolean numpy array (positional; works directly as a mask for a
            DataFrame or Series regardless of its index: `df[region.is_inside(df['ra'], df['dec'])]`).

        Rationale: A simple RA/Dec bounding box (decision 24 describes this as
        a plain rectangular patch, not a spherical-cap region) — no cos(dec)
        correction on the RA extent, matching the "just a configurable patch"
        framing. RA difference is wrapped into [-180, 180) so a region
        straddling RA=0/360 is handled correctly (unlike `tile_footprint`'s
        documented limitation).
        """
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        d_ra = (ra - self.center_ra + 180.0) % 360.0 - 180.0
        ra_ok = np.abs(d_ra) <= self.width_deg / 2.0
        dec_ok = np.abs(dec - self.center_dec) <= self.height_deg / 2.0
        return ra_ok & dec_ok


def _study_region_to_phi_box(region: StudyRegion):
    """Translate a StudyRegion (ra/dec center + box size) into the
    (gc_frame, phi1_limits, phi2_limits) that streamobs's Background.generate()
    expects.

    Builds a great-circle frame whose (phi1, phi2) = (0, 0) is exactly the
    region's center — via `GreatCircleICRSFrame.from_endpoints`, using the
    center itself as `origin` and a point 1 degree east of it (same
    declination) to define the circle. Verified empirically (not just by
    construction) that the region center round-trips to (0, 0) and that box
    corners map to sensible sky positions.

    This was an open design gap (PLAN.md §6.1.2/§6.1 open questions) — the
    prompt's `select`/`Window` machinery never needed this translation
    before; `Background.generate()` is the first thing that requires an
    actual `gala` great-circle frame rather than a plain ra/dec box.
    """
    center = SkyCoord(ra=region.center_ra * u.deg, dec=region.center_dec * u.deg)
    east_point = SkyCoord(
        ra=(region.center_ra + 1.0) * u.deg, dec=region.center_dec * u.deg
    )
    gc_frame = gc.GreatCircleICRSFrame.from_endpoints(center, east_point, origin=center)

    phi1_limits = (-region.width_deg / 2.0, region.width_deg / 2.0)
    phi2_limits = (-region.height_deg / 2.0, region.height_deg / 2.0)
    return gc_frame, phi1_limits, phi2_limits


class BackgroundSource(Protocol):
    """Protocol for loading background catalogs from various sources.

    All implementations return a catalog in streamobs column convention,
    enabling identical downstream processing.

    Rationale: Pluggable sources decouple "where background comes from"
    from "how to process it". Enables switching between real/synthetic
    data with a config change only.
    """

    def load(
        self, survey: str, release: str, region: StudyRegion | None, cfg: dict
    ) -> pd.DataFrame:
        """Load background star catalog.

        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "dp2", "yr1").
            region: StudyRegion (used only by light/catalogue sources; ignored by data_file).
            cfg: Source-specific config dict (e.g., {path: ...} for data_file).

        Returns:
            DataFrame with columns in streamobs convention
            (e.g., 'lsst_g_obs', 'lsst_r_obs', 'lsst_g_err', 'lsst_r_err', etc.)
            and standard coordinate columns ('ra', 'dec').

        Rationale: Standardized output format enables downstream code
        (cuts, clipping, select) to be identical regardless of source.
        """


class DataFileBackgroundSource(BackgroundSource):
    """Load real DP2 from parquet; apply dust correction; ignore region.

    Rationale: Real data sources don't need region restriction (the catalog
    is already a bounded skim of the full survey). Dust correction is mandatory
    (decision 25) to match observed magnitudes to isochrones.

    Attributes:
        path_key: Key in cfg dict pointing to file path (default "path").
    """

    #: Default location of the real DP2 skim, relative to the project root.
    _DEFAULT_PATH = "data/background/dp2_star_gmax_27_skim.parquet"

    def __init__(self, path_key: str = "path"):
        """Initialize source.

        Parameters:
            path_key: Key in cfg dict for file path (default "path").
        """
        self.path_key = path_key

    def load(
        self, survey: str, release: str, region: StudyRegion | None, cfg: dict
    ) -> pd.DataFrame:
        """Load DP2 parquet, dust-correct, harmonize columns.

        Ignores region (region is not passed to downstream code for data_file).

        Parameters:
            survey, release: Documentation/provenance (the real catalog is what it is)
                and used to build the output column namespace (`<survey>_<release>`).
            region: Ignored.
            cfg: dict with key self.path_key -> file path, e.g., cfg['path'] =
                'data/background/dp2_star_gmax_27_skim.parquet'.

        Returns:
            DataFrame in streamobs convention, dust-corrected.

        Raises:
            FileNotFoundError if cfg[self.path_key] doesn't exist.
        """
        raw_path = Path(cfg.get(self.path_key, self._DEFAULT_PATH))
        if not raw_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            raw_path = project_root / raw_path
        if not raw_path.exists():
            raise FileNotFoundError(f"DataFileBackgroundSource: no file at {raw_path}")

        df = pd.read_parquet(raw_path)

        bands = [b for b in "ugrizy" if f"{b}_psfMag" in df.columns]
        df = deredden_dataframe(df, bands=bands, ebv_col="ebv")

        namespace = f"{survey}_{release}" if release else survey
        return harmonize_columns(df, namespace=namespace, bands=bands, source="dp2")


class PreparedCatalogBackgroundSource(BackgroundSource):
    """A catalogue already in the pipeline's shape, read as it is.

    For catalogues a script has already cut, dereddened and renamed -- such as
    the DES Y6 backgrounds `scripts/real_data/background.py` writes: ``ra``,
    ``dec`` and ``<survey>_<release>_<band>_obs`` columns. Nothing is
    corrected here, unlike `DataFileBackgroundSource`, which dereddens and
    renames the raw LSST DP2 skim.

    Configuration (``cfg``):

    - ``path``: the parquet file.
    - ``fold`` (optional): which fold of a spatial cross-validation to keep
      (`objects_overlap.spatial_fold`), as a dict with ``index``, and
      optionally ``n_folds`` (2), ``stripe_deg`` (20) and ``nside`` (512).
      A star goes with the HEALPix pixel it falls in, and the pixel with its
      centre: splitting the stars by their own positions would leave pixels
      on a stripe edge half in each fold, seen in training by the model later
      asked to predict them. The fold is part of the configuration, so each
      fold's background maps are cached separately.
    """

    def load(
        self, survey: str, release: str, region: StudyRegion | None, cfg: dict
    ) -> pd.DataFrame:
        """The catalogue, restricted to one fold if asked.

        Raises:
            FileNotFoundError if the file is missing.
            ValueError if it lacks ``ra``, ``dec`` or a magnitude column.
        """
        from streamgoggles.objects_overlap import spatial_fold

        path = Path(cfg["path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"PreparedCatalogBackgroundSource: no file at {path}"
            )
        catalogue = pd.read_parquet(path)
        namespace = f"{survey}_{release}" if release else survey
        if not {"ra", "dec"} <= set(catalogue.columns) or not any(
            column.startswith(f"{namespace}_") and column.endswith("_obs")
            for column in catalogue.columns
        ):
            raise ValueError(
                f"{path} needs ra, dec and {namespace}_<band>_obs columns; "
                f"it has {list(catalogue.columns)}"
            )
        fold = cfg.get("fold")
        if fold is not None:
            import healpy as hp

            nside = int(fold.get("nside", 512))
            pixel = hp.ang2pix(
                nside,
                catalogue["ra"].to_numpy(),
                catalogue["dec"].to_numpy(),
                lonlat=True,
            )
            pixel_ra, _ = hp.pix2ang(nside, pixel, lonlat=True)
            keep = spatial_fold(
                pixel_ra,
                stripe_deg=float(fold.get("stripe_deg", 20.0)),
                n_folds=int(fold.get("n_folds", 2)),
            ) == int(fold["index"])
            catalogue = catalogue[keep]
        return catalogue.reset_index(drop=True)


class StreamObsLightBackgroundSource(BackgroundSource):
    """Wrap streamobs's fast/light synthetic background generation.

    Rationale: Fast, good for early testing. Region-restricted at generation time
    to avoid generating unnecessary far-field stars.

    Note (PLAN.md §6.1.2): despite "fast synthetic", this is NOT data-free —
    it reads a precomputed CMD histogram resource
    (streamobs's `BackgroundStorage`, under streamobs's own package data
    directory) that must be built once from a real/simulated reference
    catalog via `streamobs.background.resource_builder.BackgroundResourceBuilder`.
    No such resource ships with streamobs yet, so `.load()` will raise
    `FileNotFoundError` until one is built — that is a separate, not-yet-scoped
    prerequisite, not a bug in this wrapper (verified: the same
    `FileNotFoundError` comes directly from `streamobs`, not from this file).

    Unlike `StreamObsCatalogueBackgroundSource`/`DataFileBackgroundSource`,
    this method produces NO per-star reported-error columns (`<namespace>_<band>_err`)
    at all — it's a CMD-histogram draw with no per-star photometric-error
    model, only `<namespace>_<band>_obs` (verified empirically). Cuts using
    `quantity: "snr"` (background.py's/data_preparation.py's Cut, which reads
    `err_col`) will raise `KeyError` against a catalog from this source —
    exactly what broke `config/background.yaml`'s original default cuts,
    since `light` is this project's default `source` (fixed 2026-09-09 by
    switching those defaults to `quantity: "mag"`, which this source does
    produce).
    """

    def load(
        self, survey: str, release: str, region: StudyRegion | None, cfg: dict
    ) -> pd.DataFrame:
        """Generate synthetic background via streamobs.

        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "yr1").
            region: StudyRegion for generation (if None, uses default study region).
            cfg: Source-specific kwargs. `bands` (tuple of 2 band names,
                default ("g", "r")) selects the color/reference bands
                forwarded to `streamobs.background.background.Background`;
                `seed` (int) or `rng` (np.random.Generator) are forwarded to
                `Background.generate()`, which is where streamobs actually
                reads them -- **pass one of these if reproducibility
                matters**, since generation is otherwise unseeded and
                produces a different catalog on every call, including within
                a single process and regardless of numpy's global seed;
                everything else in `cfg` (e.g. `storage=`) is forwarded to
                the constructor as-is.

        Returns:
            DataFrame in streamobs convention, region-restricted.

        Raises:
            FileNotFoundError if release is not supported by streamobs (no
                matching survey config file), or if the precomputed CMD
                resource for this survey/bands hasn't been built yet (see
                class docstring) — both surface as the same exception type,
                streamobs's own, not raised by this wrapper.
        """
        region = region if region is not None else StudyRegion()
        gc_frame, phi1_limits, phi2_limits = _study_region_to_phi_box(region)

        survey_obj = Survey.load(survey=survey, release=release)
        # `seed`/`rng` belong to generate(), not to Background's constructor:
        # streamobs reads them from generate()'s **kwargs
        # (`rng = np.random.default_rng(kwargs.get("seed"))`). Forwarding
        # them to the constructor instead -- which is what happened before
        # this split existed -- silently left generation unseeded, so every
        # call produced a different background catalog even within one
        # process, and even with numpy's global seed fixed (verified
        # directly: two loads in one process differ in star count). That
        # made every downstream "reproducible, fixed-seed" run irreproducible
        # at the data level; see PLAN.md section 6.15.
        generate_kwargs = {key: cfg[key] for key in ("seed", "rng") if key in cfg}
        extra_kwargs = {
            k: v for k, v in cfg.items() if k not in ("bands", "seed", "rng")
        }
        bands = tuple(cfg.get("bands", ("g", "r")))

        background = Background(
            surveys=survey_obj, method="light", bands=bands, **extra_kwargs
        )
        catalog, _meta = background.generate(
            phi1_limits=phi1_limits,
            phi2_limits=phi2_limits,
            gc_frame=gc_frame,
            **generate_kwargs,
        )
        return catalog


class StreamObsCatalogueBackgroundSource(BackgroundSource):
    """Wrap streamobs's full injection-based catalog generation
    (`Background(method="injection")`).

    Rationale: More realistic than light (runs the actual survey injection
    pipeline — photometric errors, detection efficiency — on a real input
    catalog), but slower, and requires that input catalog to already exist.

    Note (PLAN.md §6.1.2): unlike `light`, streamobs's `method="injection"`
    does NOT restrict generation by `phi1_limits`/`phi2_limits`/`gc_frame` at
    all internally (verified against `streamobs/background/background.py`:
    `_generate_injection` ignores those three arguments — it just injects
    `catalog_stars` wholesale). So region-restriction is applied here as an
    explicit post-filter via `StudyRegion.is_inside()` — a plain ra/dec box
    check, which is what a `StudyRegion` fundamentally means (a ra/dec patch,
    decision 24); the underlying `gc_frame`/`phi1`/`phi2` machinery is only
    there because `Background.generate()`'s signature requires it, not
    because injection's restriction needs to match `light`'s geometry.
    """

    def load(
        self, survey: str, release: str, region: StudyRegion | None, cfg: dict
    ) -> pd.DataFrame:
        """Generate synthetic background via streamobs (injection backend).

        Parameters:
            survey: Survey name (e.g., "lsst").
            release: Release name (e.g., "yr1").
            region: StudyRegion for generation (if None, uses default study region).
                Applied as a post-filter (see class docstring).
            cfg: Must contain `catalog_stars` (a DataFrame, or a path
                `pandas.read_parquet`/`read_csv` can load, of true stellar
                positions/magnitudes to inject through the survey model — see
                `streamobs.observed.StreamInjector`). Optional
                `catalog_galaxies` for galaxy injection. `bands` (default
                ("g", "r")) as in `StreamObsLightBackgroundSource`. Everything
                else is forwarded to `Background`'s constructor.

        Returns:
            DataFrame in streamobs convention, region-restricted.

        Raises:
            ValueError if `cfg` doesn't contain `catalog_stars`.
            FileNotFoundError if release is not supported by streamobs (no
                matching survey config file) — streamobs's own exception,
                not raised by this wrapper.
        """
        if "catalog_stars" not in cfg:
            raise ValueError(
                "StreamObsCatalogueBackgroundSource (method='injection') requires "
                "cfg['catalog_stars'] -- a real or simulated true stellar catalog "
                "to inject through the survey model. No such reference catalog is "
                "available in this project yet (PLAN.md §6.1.2); this must be "
                "supplied before this source can be used."
            )

        region = region if region is not None else StudyRegion()
        gc_frame, phi1_limits, phi2_limits = _study_region_to_phi_box(region)

        survey_obj = Survey.load(survey=survey, release=release)
        bands = tuple(cfg.get("bands", ("g", "r")))
        extra_kwargs = {
            k: v
            for k, v in cfg.items()
            if k not in ("bands", "catalog_stars", "catalog_galaxies")
        }

        background = Background(
            surveys=survey_obj,
            method="injection",
            bands=bands,
            catalog_stars=cfg["catalog_stars"],
            catalog_galaxies=cfg.get("catalog_galaxies"),
            **extra_kwargs,
        )
        catalog = background.generate(
            phi1_limits=phi1_limits, phi2_limits=phi2_limits, gc_frame=gc_frame
        )
        return catalog[region.is_inside(catalog["ra"], catalog["dec"])].reset_index(
            drop=True
        )


def harmonize_columns(
    df: pd.DataFrame,
    namespace: str,
    bands: list[str] | None = None,
    source: str = "dp2",
    dust_corrected: bool = True,
) -> pd.DataFrame:
    """Rename/derive DP2 columns to streamobs convention.

    Parameters:
        df: Input DataFrame with DP2 column names (`coord_ra`, `coord_dec`,
            `<band>_psfMag[Err]`, `refExtendedness`, `ebv`, and — if
            `dust_corrected` — `<band>_psfMag_dered`/`A_<band>` as added by
            `data_preparation.deredden_dataframe`).
        namespace: Output column namespace (`<survey>_<release>`, e.g.
            "lsst_dp2") — matches `streamobs.columns.obs_col`'s convention,
            so the harmonized background and any injected stream catalog use
            identical column names for the same survey/release.
        bands: Bands to harmonize (default: every band with a `<band>_psfMag`
            column present, i.e. up to ugrizy).
        source: Source type ("dp2" only for now).
        dust_corrected: Whether `df` already has `deredden_dataframe`'s
            `<band>_psfMag_dered` columns to use as the "observed" magnitude
            (True — the mandatory case for `data_file`, decision 25) or
            whether to fall back to the raw (non-dereddened) `<band>_psfMag`.

    Returns:
        DataFrame with streamobs-convention columns
        (e.g., 'lsst_dp2_g_obs', 'lsst_dp2_r_obs', etc.), plus 'ra', 'dec',
        'extendedness', 'ebv', and per-band 'A_<band>' (if dust-corrected).

    Rationale: Enables the same downstream select() function to work on
    both background and stream stars (both must be in same convention).

    Raises:
        ValueError if source is not supported.
    """
    if source != "dp2":
        raise ValueError(f"harmonize_columns: unsupported source {source!r}")

    if bands is None:
        bands = [b for b in "ugrizy" if f"{b}_psfMag" in df.columns]

    out = pd.DataFrame(index=df.index)
    out["ra"] = df["coord_ra"]
    out["dec"] = df["coord_dec"]
    if "refExtendedness" in df.columns:
        out["extendedness"] = df["refExtendedness"]
    if "ebv" in df.columns:
        out["ebv"] = df["ebv"]

    mag_col_suffix = "_psfMag_dered" if dust_corrected else "_psfMag"
    for band in bands:
        mag_col = f"{band}{mag_col_suffix}"
        err_col_raw = f"{band}_psfMagErr"
        if mag_col in df.columns:
            out[obs_col(band, namespace)] = df[mag_col]
        if err_col_raw in df.columns:
            out[err_col(band, namespace)] = df[err_col_raw]
        if dust_corrected and f"A_{band}" in df.columns:
            out[f"A_{band}"] = df[f"A_{band}"]

    return out
