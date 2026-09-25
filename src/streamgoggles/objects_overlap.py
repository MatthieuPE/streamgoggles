"""Known objects that are not streams, and the footprints they sit in.

A stream search must not be asked to find a globular cluster or a dwarf
galaxy: both are real stellar overdensities, both are already catalogued, and
neither is what the model is for. They are masked out of training so the model
never learns them as signal, and out of evaluation so finding one is not
counted either way.

Catalogues live in the repository's `data/others/`, outside the package and
outside version control, so nothing here needs the network. Point
`STREAMGOGGLES_DATA` elsewhere to use another copy; an installed wheel has
no data directory of its own, so that variable is how it finds one.

- `dwarf_all.csv` and `gc_harris.csv`, both from the local volume database
  (https://github.com/apace7/local_volume_database), sharing one schema:
  `ra`, `dec` in degrees, `rhalf` the half-light radius in arcminutes, `bb`
  the galactic latitude.
- `footprint_des_yr6_nside512.fits.gz`, the DES Y6 Gold coverage built from
  the downloaded catalogue: 383,237 pixels, 5,026 square degrees.
- `mask_des_yr6_background_nside512.fits.gz`, that coverage with known
  streams and objects removed: 4,017 square degrees, the sky training and
  evaluation both use (`get_footprint("des_yr6_background")`).
"""

import os
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.table import Table

DATA = Path(
    os.environ.get(
        "STREAMGOGGLES_DATA", Path(__file__).resolve().parents[2] / "data" / "others"
    )
)
GC_URL = (
    "https://raw.githubusercontent.com/apace7/local_volume_database/"
    "main/data/gc_harris.csv"
)
# Footprints that ship with the package, by name. Each is a HEALPix mask in
# ring order whose nside is part of the file name.
FOOTPRINTS = {
    # The survey's coverage, from the downloaded Y6 Gold catalogue.
    "des_yr6": DATA / "footprint_des_yr6_nside512.fits.gz",
    # That coverage minus known streams, globular clusters and dwarfs: the sky
    # a stream can be injected into and a detection can count on. Built by
    # scripts/real_data/background.py.
    "des_yr6_background": DATA / "mask_des_yr6_background_nside512.fits.gz",
    # The sky trained models are run on: the same, but with the known streams
    # left in -- only Sagittarius stays masked -- since finding them is the
    # point. Also built by scripts/real_data/background.py.
    "des_yr6_inference": DATA / "mask_des_yr6_inference_nside512.fits.gz",
}

_GC_CACHE = None
_DWARF_CACHE = None


def get_GC(force_reload=False, path=None, **_):
    """The Harris globular cluster catalogue, cached at module level.

    Reads the packaged snapshot by default. `force_reload` re-fetches the
    live catalogue over the network instead, which is the only way this
    touches the network at all.
    """
    global _GC_CACHE
    if _GC_CACHE is None or force_reload:
        source = GC_URL if force_reload else (path or DATA / "gc_harris.csv")
        _GC_CACHE = Table.read(
            source, format="csv" if str(source).endswith("csv") else None
        )
    return _GC_CACHE


def get_dwarf(dwarfs_path=None, force_reload=False, **_):
    """The dwarf galaxy catalogue, cached at module level (see `get_GC`)."""
    global _DWARF_CACHE
    if _DWARF_CACHE is None or force_reload:
        _DWARF_CACHE = Table.read(dwarfs_path or DATA / "dwarf_all.csv", format="csv")
    return _DWARF_CACHE


def get_footprint(footprint=None, nside=512, nest=False, **_):
    """The sky this search covers, as (mask, pixels, npix).

    Parameters:
        footprint: what to cover. `None` is the whole sky. A name from
            `FOOTPRINTS` ("des_yr6") loads the packaged mask. A path loads a
            HEALPix mask from that file. A boolean array of length npix, or an
            array of pixel indices, is used as given.
        nside: resolution to return. A packaged mask of a different nside is
            resampled to it, up or down.
        nest: True for nested ordering; packaged masks are ring, and are
            reordered on the way out.

    Returns:
        (mask, pixels, npix): a boolean map, the indices where it is True, and
        its length. Every caller here wants at least two of the three, so all
        three are returned rather than recomputed.
    """
    npix = hp.nside2npix(nside)

    if footprint is None:
        mask = np.ones(npix, dtype=bool)
        return mask, np.arange(npix, dtype=np.int64), npix

    if isinstance(footprint, (str, Path)):
        path = FOOTPRINTS.get(str(footprint), footprint)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no footprint {footprint!r} at {path}; known names are "
                f"{sorted(FOOTPRINTS)}, or give a path or a mask. Data lives in "
                f"{DATA}, overridable with STREAMGOGGLES_DATA."
            )
        stored = np.asarray(hp.read_map(path), dtype=float)
        stored_nside = hp.npix2nside(stored.size)
        if stored_nside != nside:
            # A resampled mask is a coverage fraction; anything partly covered
            # counts as covered, which keeps a footprint from shrinking at the
            # edges each time it is regridded.
            stored = hp.ud_grade(
                stored, nside_out=nside, order_in="RING", order_out="RING"
            )
        mask = stored > 0
        if nest:
            mask = hp.reorder(mask.astype(np.uint8), r2n=True).astype(bool)
    else:
        given = np.asarray(footprint)
        if given.dtype == bool:
            if given.size != npix:
                raise ValueError(
                    f"footprint mask has {given.size} pixels, nside {nside} "
                    f"needs {npix}"
                )
            mask = given
        else:
            mask = np.zeros(npix, dtype=bool)
            mask[given.astype(np.int64)] = True

    return mask, np.flatnonzero(mask).astype(np.int64), npix


def _within(catalogue, footprint_pix, npix, nest=False):
    """Which rows of a catalogue land on a footprint pixel."""
    nside = hp.npix2nside(npix)
    pix = hp.ang2pix(
        nside,
        np.asarray(catalogue["ra"], dtype=float),
        np.asarray(catalogue["dec"], dtype=float),
        lonlat=True,
        nest=nest,
    )
    return pix, np.isin(pix, footprint_pix)


def get_GC_within_footprint(min_galactic_latitude=10.0, **kwargs):
    """Globular clusters inside the footprint, as (catalogue, selection).

    Clusters near the galactic plane are dropped: the plane is crowded enough
    that the search is not run there anyway.
    """
    gc_harris = get_GC(**kwargs)
    _, footprint_pix, npix = get_footprint(**kwargs)
    _, inside = _within(gc_harris, footprint_pix, npix, nest=kwargs.get("nest", False))
    selection = inside & (
        np.abs(np.asarray(gc_harris["bb"], dtype=float)) > min_galactic_latitude
    )
    return gc_harris, selection


def get_dwarf_within_footprint(**kwargs):
    """Dwarf galaxies inside the footprint, as (catalogue, selection)."""
    dwarf_all = get_dwarf(**kwargs)
    _, footprint_pix, npix = get_footprint(**kwargs)
    _, inside = _within(dwarf_all, footprint_pix, npix, nest=kwargs.get("nest", False))
    return dwarf_all, inside


def mask_objects(
    catalogue,
    selection=None,
    nside=512,
    radius_factor=5.0,
    min_radius_deg=0.05,
    nest=False,
):
    """A HEALPix mask covering each object, out to a multiple of its size.

    Parameters:
        catalogue: table with `ra`, `dec` and `rhalf` (arcminutes).
        selection: boolean over the catalogue; None uses every row.
        radius_factor: how many half-light radii to mask. The default 5 is
            well outside the half-light radius, where the profile of a cluster
            or dwarf still contributes stars to a matched-filter map.
        min_radius_deg: floor for objects with a missing or zero `rhalf`, of
            which the catalogues hold a fair number.

    Returns:
        (mask, radii): the boolean map, and the radius in degrees used for
        each selected object, so a caller can report what it masked.
    """
    rows = catalogue if selection is None else catalogue[selection]
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    radii = []
    for row in rows:
        half_light = float(row["rhalf"]) / 60.0 if np.isfinite(row["rhalf"]) else 0.0
        radius = max(radius_factor * half_light, min_radius_deg)
        vector = hp.ang2vec(float(row["ra"]), float(row["dec"]), lonlat=True)
        mask[hp.query_disc(nside, vector, np.radians(radius), nest=nest)] = True
        radii.append(radius)
    return mask, np.array(radii)


# Widths (degrees) of the DES 2018 streams, Shipp et al. (2018) Table 1. The
# same numbers as DES_STREAMS in scripts/experiments/stream_parameters/run.py,
# which a test keeps in step, so a stream is masked at the width it is
# injected with.
DES2018_STREAM_WIDTHS = {
    "Tucana III": 0.18,
    "ATLAS": 0.24,
    "Molonglo": 0.32,
    "Phoenix": 0.16,
    "Indus": 0.83,
    "Jhelum": 1.16,
    "Ravi": 0.72,
    "Chenab": 0.71,
    "Elqui": 0.54,
    "Aliqa Uma": 0.26,
    "Turbio": 0.25,
    "Willka Yaku": 0.21,
    "Turranburra": 0.60,
    "Wambelong": 0.40,
}
# Sagittarius is masked too, as in the DES 2018 analysis: no cold stream is
# being looked for inside it. Its 6 degrees is galstreams' width_phi2.
SAGITTARIUS_WIDTH = 6.0

# Names the DES 2018 paper uses that galstreams files under something else.
# Chenab and Palca map to the larger structures they belong to, which is what
# should be masked.
GALSTREAMS_NAMES = {
    "Tucana III": "TucanaIII",
    "ATLAS": "AAU-ATLAS",
    "Aliqa Uma": "AAU-AliqaUma",
    "Willka Yaku": "Willka_Yaku",
    "Chenab": "Orphan-Chenab",
}


# Which galstreams tracks define each stream's mask: the DES measurement
# (Shipp et al. 2018, 2019) wherever galstreams carries one. The other
# references extend some streams far past what DES measured -- Ibata et al.
# (2024) trace Jhelum over 95.6 degrees and Indus over 90.3, against 24.6 and
# 18.2 in DES -- and masking those extensions cost 9 points of the footprint
# for Jhelum alone. Streams absent from this table use every track galstreams
# has: ATLAS, Aliqa Uma and Molonglo have no DES track there, and Chenab is
# masked along the whole Orphan-Chenab stream, which is one established
# stream crossing the footprint rather than a disputed extension.
STREAM_TRACKS = {
    "Jhelum": ("Jhelum-a.shipp2019", "Jhelum-b.shipp2019"),
    "Indus": ("Indus.shipp2019",),
    "Tucana III": ("TucanaIII.shipp2019",),
    "Phoenix": ("Phoenix.shipp2019",),
    "Elqui": ("Elqui.shipp2019",),
    "Turranburra": ("Turranburra.shipp2019",),
    "Ravi": ("Ravi.shipp2018",),
    "Turbio": ("Turbio.shipp2018",),
    "Wambelong": ("Wambelong.shipp2018",),
    "Willka Yaku": ("Willka_Yaku.shipp2018",),
}


def stream_tracks(name, exclude=(), tracks=STREAM_TRACKS):
    """The galstreams tracks for one stream, as {reference: (ra, dec)}.

    Parameters:
        name: the stream, as the DES 2018 paper names it.
        exclude: references to leave out, as "<name>.<reference>".
        tracks: {stream: references} choosing the tracks explicitly; a stream
            not in it gets every track whose name matches exactly (so "Jhelum"
            alone finds Ibata 2021 and 2024 but not the Jhelum-a and Jhelum-b
            components). Pass {} for every matching track of every stream.

    The shipped track files are read directly, because
    `galstreams.MWStreams()` raises an IndexError against astropy 8. Both kinds
    are already densified: `track.st.*` are measured tracks, `track.ep.*` are
    endpoint pairs interpolated to 200 points.
    """
    import galstreams
    from astropy.table import Table as _Table

    folder = Path(galstreams.__file__).parent / "tracks"
    chosen = tracks.get(name) if tracks else None
    if chosen is not None:
        paths = []
        for reference in chosen:
            found = sorted(folder.glob(f"track.??.{reference}.ecsv"))
            if not found:
                raise FileNotFoundError(f"galstreams has no track {reference!r}")
            paths.extend(found)
    else:
        galstreams_name = GALSTREAMS_NAMES.get(name, name.replace(" ", "_"))
        paths = sorted(folder.glob(f"track.??.{galstreams_name}.*.ecsv"))

    found_tracks = {}
    for path in paths:
        if path.name.endswith(".summary.ecsv"):
            continue
        reference = path.name[len("track.st.") : -len(".ecsv")]
        if reference in exclude:
            continue
        table = _Table.read(path)
        found_tracks[reference] = (
            np.asarray(table["ra"].value, dtype=float),
            np.asarray(table["dec"].value, dtype=float),
        )
    return found_tracks


def mask_radius(width, width_factor=3.0, wide_stream_deg=2.0, wide_stream_factor=1.0):
    """How far from its track a stream is masked, in degrees.

    A cold stream's width is a Gaussian sigma, so it is masked to a multiple
    of it. A broad structure's quoted width is already its extent -- tripling
    Sagittarius's 6 degrees would mask 27% of the DES footprint by itself --
    so anything at least `wide_stream_deg` wide gets `wide_stream_factor`.
    """
    factor = wide_stream_factor if width >= wide_stream_deg else width_factor
    return factor * width


def mask_streams(
    widths=None,
    nside=512,
    width_factor=3.0,
    wide_stream_deg=2.0,
    wide_stream_factor=1.0,
    exclude=(),
    nest=False,
    tracks=STREAM_TRACKS,
):
    """A HEALPix mask over the tracks of known streams.

    Parameters:
        widths: {stream name: width in degrees}. None is the DES 2018 streams
            plus Sagittarius.
        width_factor, wide_stream_deg, wide_stream_factor: see `mask_radius`.
        exclude, tracks: which galstreams tracks to use (see `stream_tracks`).

    Returns:
        (mask, pixels): the union, and each stream's own pixels, so a caller
        can say what each one costs.
    """
    if widths is None:
        widths = {**DES2018_STREAM_WIDTHS, "Sagittarius": SAGITTARIUS_WIDTH}
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    pixels = {}
    for name, width in widths.items():
        radius = np.radians(
            mask_radius(width, width_factor, wide_stream_deg, wide_stream_factor)
        )
        own = np.zeros_like(mask)
        for ra, dec in stream_tracks(name, exclude, tracks).values():
            vectors = hp.ang2vec(ra, dec, lonlat=True)
            # A few hundred points per track is plenty: consecutive discs of
            # this radius overlap heavily, so sampling costs no coverage.
            for vector in vectors[:: max(1, len(vectors) // 600)]:
                own[hp.query_disc(nside, vector, radius, nest=nest)] = True
        if own.any():
            pixels[name] = np.flatnonzero(own)
            mask |= own
    return mask, pixels


def build_background_mask(
    footprint="des_yr6",
    nside=512,
    stream_widths=None,
    width_factor=3.0,
    wide_stream_deg=2.0,
    wide_stream_factor=1.0,
    exclude_tracks=(),
    tracks=STREAM_TRACKS,
    object_radius_factor=5.0,
    object_min_radius_deg=0.05,
):
    """Where a stream can be injected: the footprint, minus known streams,
    minus globular clusters and dwarf galaxies.

    Training and evaluation both need this same sky -- training so the model
    never learns a real overdensity as background, evaluation so finding one
    is not counted either way -- so it is built in one place.

    Returns:
        dict with the boolean maps `footprint`, `streams`, `objects` and
        `usable` (footprint and neither of the others), plus
        `stream_pixels` (see `mask_streams`) and the object catalogues with
        their in-footprint selections.
    """
    inside, _, _ = get_footprint(footprint, nside=nside)
    streams, stream_pixels = mask_streams(
        stream_widths,
        nside,
        width_factor,
        wide_stream_deg,
        wide_stream_factor,
        exclude_tracks,
        tracks=tracks,
    )
    clusters, clusters_in = get_GC_within_footprint(footprint=footprint, nside=nside)
    dwarfs, dwarfs_in = get_dwarf_within_footprint(footprint=footprint, nside=nside)
    cluster_mask, cluster_radii = mask_objects(
        clusters, clusters_in, nside, object_radius_factor, object_min_radius_deg
    )
    dwarf_mask, dwarf_radii = mask_objects(
        dwarfs, dwarfs_in, nside, object_radius_factor, object_min_radius_deg
    )
    objects = cluster_mask | dwarf_mask
    return {
        "footprint": inside,
        "streams": streams,
        "objects": objects,
        "usable": inside & ~streams & ~objects,
        "stream_pixels": stream_pixels,
        "clusters": (clusters, clusters_in, cluster_radii),
        "dwarfs": (dwarfs, dwarfs_in, dwarf_radii),
    }


def spatial_fold(ra, stripe_deg=20.0, n_folds=2):
    """Which fold of a spatial cross-validation a sky position belongs to.

    Stripes of `stripe_deg` in right ascension, dealt out in turn. On real
    data there is one sky, and a model scored on the pixels it trained on
    has already been taught what their background looks like, which flatters
    its false-alarm rate; a model trained on one fold and run on another has
    not. Stripes in right ascension rather than a single cut give every fold
    the whole range of galactic latitudes DES spans. At 20 degrees a stripe
    is about 13 degrees across at Dec -50, wider than an 11-degree window, so
    most training windows sit inside one fold.

    Returns:
        integer array, the fold of each position.
    """
    stripe = np.floor(np.mod(np.asarray(ra, dtype=float), 360.0) / stripe_deg)
    return stripe.astype(np.int64) % n_folds
