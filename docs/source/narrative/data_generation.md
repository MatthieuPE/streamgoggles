# Data generation

This is the largest subsystem: everything involved in turning a sky
footprint and a set of stream parameters into one labeled training sample
(a {py:class}`~streamgoggles.sample.Sample`). {doc}`notebooks` walks
through it interactively; this page explains the pieces and why they're
shaped the way they are.

## Configuration ({py:mod}`streamgoggles.config`)

`StreamConfig` describes a *distribution* over stream parameters (each a
`ParameterSpec`: fixed, uniform, log-uniform, or discrete), plus
`background_fraction` (what fraction of training samples should be pure
background, no stream at all) and `richness_kind` (which unit the
`"richness"` parameter is expressed in). `build_eval_grid` turns every
non-fixed parameter into an exhaustive grid — one evaluation point per
combination — with a stable per-point seed, so an evaluation set is
reproducible across runs.

## Storage ({py:mod}`streamgoggles.storage`)

Three parameter-addressed stores, all built on canonical parameter-dict
hashing (`ParameterStore`):

- `SimulationStore` — persisted `Sample`s (used by eval mode, optionally by
  training mode).
- `BackgroundMapStore` — cached raw/finalized background maps, keyed per
  `(filter, distance)`, so re-running with an unchanged background
  configuration is a cache hit.
- `ModelStore` — model checkpoints (`state_dict.pt`), a config snapshot
  (the exact kwargs to reconstruct the model class), and training metrics.

## Pixelization and matched filters ({py:mod}`streamgoggles.matched_filter`)

`PixelizationSpec` fixes a HEALPix `nside` and an output image size/scale;
`project`/`crop_window` handle the HEALPix → 2D gnomonic projection.
`MatchedFilter` is a small protocol with one method, `select(catalog,
bands, distance_modulus) -> bool array`:

- `StreamobsSplineFilter` wraps `streamobs.match_filter.build_match_filter`
  — an isochrone-shaped color-magnitude polygon at a given age/metallicity/
  distance.
- `ShiftedColorBoxFilter` is the deliberately "bad" companion: a plain
  box cut sharing the reference filter's magnitude range at the same
  distance, shifted off the isochrone locus in color. It still lets
  background contamination through without preferring real stream stars —
  exactly the negative example the network needs to see.

`finalize_full` optionally applies HEALPix-sphere Gaussian smoothing
(`healpy.smoothing`) and a polynomial background subtraction before a map
gets projected/cropped.

## Windows ({py:mod}`streamgoggles.windows`)

A `Window` is a sky patch: center, rotation, width/height. Two ways to get
one: `sample_random_window` (uniform over the footprint, for background-only
samples) and `sample_stream_window` (rejection-sampled to guarantee at
least a configurable minimum length of the injected stream's track falls
inside it — a floor, not a guarantee of *full* inclusion; the label is
still cropped to the same window, so stream stars outside it are correctly
excluded from the count).

## Background sources ({py:mod}`streamgoggles.background_sources`)

`StudyRegion` is a plain RA/Dec box, independent of and larger than any one
sample's window — background gets generated once for the whole region and
reused. Three interchangeable sources:

- `StreamObsLightBackgroundSource` — `streamobs`'s fast synthetic
  field-star population. No per-star photometric error columns, so
  `"snr"`-based cuts silently can't be applied against it (see the
  `"mag"`-based default in `config/background.yaml`).
- `StreamObsCatalogueBackgroundSource` — `streamobs`'s slower,
  injection-based method, which does carry per-star errors.
- `DataFileBackgroundSource` — a real data skim from disk.

## Cuts and clipping ({py:mod}`streamgoggles.data_preparation`)

`Cut`/`apply_cuts`/`apply_magnitude_clipping` live here — deliberately
*not* inside `background.py` — because they need to apply identically to
background stars and injected stream stars. Putting them in a shared,
catalog-agnostic module is what makes that guarantee structural rather
than a convention someone has to remember to follow at every call site.

## Background maps ({py:mod}`streamgoggles.background`)

`Background.load_or_cache` is the one call that ties source loading, region
restriction, cuts/clipping, and per-`(filter, distance)` raw-map caching
together. Before touching the catalog at all, it checks the cache for every
`(filter, distance)` pair; if everything is already cached, the catalog
never gets loaded. Each matched filter is guaranteed to touch the
background catalog exactly once per distance, regardless of how many times
`load_or_cache` gets called.

## Richness conversions ({py:mod}`streamgoggles.inject_utils`)

A stream's richness can be specified as a literal star count, a stellar
mass (M☉), or a surface brightness (mag/arcsec²) — `convert_N_to_Mass`,
`convert_Mass_to_N`, and the surface-brightness equivalents convert between
them via the isochrone's stellar population, so
`resolve_richness_to_nstars` can always reduce whichever unit was given to
an integer star count before realization.

## Injection ({py:mod}`streamgoggles.injector`)

`StreamInjector.inject_single_stream` is the orchestrator: resolve
richness → realize the stream population (true, noiseless magnitudes) →
place it along a random great circle in the footprint
(`place_stream_in_footprint`, which builds its own great-circle frame
rather than `streamobs`'s own rejection search, since that search fails
against footprints this narrow) → run it through the survey's
noise/detection model → apply the same cuts/clipping as the background →
sample a window → for every `(distance, filter)` channel: select, make a
raw count map, combine with the cached background, finalize, crop to the
window — and separately crop the *stream-only* raw map as that channel's
label. `inject_background_only` produces the shape-compatible negative
case: no realization, no label computation, just the cached background
cropped to a random window (used whenever
`StreamConfig.background_fraction > 0`).

## Alternate labels ({py:mod}`streamgoggles.rasterize`)

The pre-pivot label mechanism — `rasterize_binary`, `rasterize_density`, a
`soft_distance` stub, dispatched via `StreamInjector.label_policy` — remains
implemented and selectable, but is no longer the default (`"stream_count"`
is, computed directly in `injector.py`, not routed through this module at
all).
