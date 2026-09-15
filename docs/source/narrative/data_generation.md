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

### From a catalog to a HEALPix count map ({py:func}`~streamgoggles.matched_filter.make_raw_map`)

Before any projection happens, a selected set of catalog rows (background
*or* stream stars — the same function serves both) gets binned onto a
full-sky HEALPix grid at `pix.nside`: `raw_map[p]` is the count of selected
stars whose `(ra, dec)` falls in HEALPix pixel `p`. A separate `valid_mask`
marks which HEALPix pixels the *catalog itself* has any coverage at all
(selected or not) — this is what makes "zero selected stars in a covered
pixel" distinguishable from "no survey data here at all": both read as
`0.0` in `raw_map`, and only `valid_mask` tells them apart. For the dense
background catalog this traces the real survey footprint; a stream-only
catalog only touches a handful of pixels, so a stream's own `valid_mask`
is never used for validity downstream — only the background's (computed
once, cached, and shared by every sample) is.

### HEALPix → 2D image: the gnomonic projection ({py:func}`~streamgoggles.matched_filter.project`)

Every full-sky HEALPix map (`raw_map`, or the combined/finalized map
downstream) eventually needs to become a small 2D image centered on a
particular `Window` — that's what `project`/`crop_window` do. The
projection is a standard **gnomonic (TAN) projection**, following
Calabretta & Greisen (2002)'s conventions — the same projection FITS/WCS
astronomy tools use for "tangent plane" images, chosen because it correctly
handles projection effects (RA stretching near the celestial poles) rather
than naively treating RA/Dec as a flat Cartesian grid.

**Building the output grid.** `_tangent_plane_radec` lays out a regular
grid of tangent-plane offsets `(xi, eta)` — one pair per output pixel,
spaced by `pixel_scale_deg`, centered on `(0, 0)` — then **deprojects**
each offset back to real sky coordinates `(ra, dec)` around the window's
`(center_ra, center_dec)`, using the standard inverse-gnomonic spherical
trig (`rho = sqrt(xi^2 + eta^2)`, `c = arctan(rho)`, then `dec`/`ra` from
`arcsin`/`arctan2` of `sin(c)`/`cos(c)` combined with the tangent point).
The forward direction (`world_to_tangent_plane`) is the exact mathematical
inverse, sharing the identical rotation convention, and is used elsewhere
(`windows.py`) purely as an "is this sky point inside this window?" test —
a point is inside a `size_deg × size_deg` window iff both `|xi_deg|` and
`|eta_deg|` are within `size_deg / 2`.

**Rotation is part of the projection, not a post-processing step.** A
window's `rotation_deg` (position angle) is applied by rotating the
tangent-plane offsets `(xi, eta)` *before* deprojecting them to `(ra,
dec)` — not by deprojecting first and then rotating the resulting 2D
image. Rotating pixels after the fact would need its own resampling/
interpolation pass (softening the image and needing to reconcile two
separate validity masks); folding the rotation into the tangent-plane
coordinates themselves means there is exactly one resampling step for the
whole operation, and `world_to_tangent_plane`'s inverse stays exact.

**Sampling the HEALPix map at those sky positions.** Once every output
pixel has a real `(ra, dec)`, `PixelizationSpec.interpolate` picks how the
HEALPix map gets sampled there:

- `interpolate=True` (the default): `healpy.get_interp_weights` finds the
  4 surrounding HEALPix pixels for each output position and their bilinear
  weights; the output value is their weighted sum. Smoother, but an output
  pixel is only marked valid if **all four** contributing HEALPix neighbors
  are themselves valid — otherwise the interpolation would silently blend
  the invalid-pixel `0.0` fill value in as if it were real data, quietly
  corrupting counts near the footprint edge.
- `interpolate=False`: plain nearest-neighbor (`healpy.ang2pix`) — one
  HEALPix pixel per output pixel, validity copied directly from that one
  pixel. Blockier, but avoids any averaging across the footprint edge or
  between the "good" and "decoy" filters' differently-shaped selections.

Either way, any output pixel that ends up invalid is then forced to
exactly `0.0` in the returned image, regardless of whatever value the
(vestigial) interpolation/lookup computed for it — the single source of
truth for "is this pixel real" is always `valid_mask`, never the image
value alone (see {doc}`datasets_and_models`'s note on why that matters
downstream, past normalization).

**Pixel scale is tied to `nside`, on purpose.** `PixelizationSpec.
__post_init__` auto-derives `pixel_scale_deg` from `nside`
(`native_pixel_scale_deg`, `healpy.nside2resol`) when it isn't given
explicitly, and **rejects** an explicitly-given scale that's much finer
than that native HEALPix resolution — requesting `pixel_scale_deg` finer
than the map was ever sampled at would silently manufacture angular detail
that was never actually observed. Pass a *coarser* `pixel_scale_deg`
deliberately if you want that; you cannot ask for a finer one without
increasing `nside` first.

### Matched filters

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
gets projected/cropped. When disabled (the common case; `finalize_cfg=
{"enabled": False}` or `None`) it's not merely a no-op copy of the raw map
— `Background.finalized_map_full_dict[filter][dm]` is literally `None` in
that case, not the raw map. Code reading from it needs the same
`raw_map_full_dict` fallback `injector.inject_background_only` already
uses (`chosen = finalized.get(...) or raw[...]`) — `crop_window`ing a
`None` directly is a real mistake it's easy to make writing new code
against this (see {doc}`tutorial`'s detection walkthrough).

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

### Dust correction is source-dependent, for different reasons each time

`BackgroundConfig.dust_correction` defaults to disabled for the two
synthetic sources, and that's deliberate, not an oversight — each is
already dust-handled internally, just via a different mechanism:

- `StreamObsCatalogueBackgroundSource` (`method="injection"`) subtracts
  the per-band extinction from observed magnitudes as part of `streamobs`'s
  own injection step (`dust_correction=True` there by default) — applying
  `utils.deredden` again on top would double-correct.
- `StreamObsLightBackgroundSource` (`method="light"`) draws magnitudes from
  a precomputed color-magnitude-diagram grid built at *uniform*
  (dust-free) survey conditions, and instead reduces the *effective*
  magnitude limit used to sample that grid, per pixel, by the local
  extinction (`maglim_eff = maglim_obs - A_band`) — dust changes which
  grid point gets sampled (faint stars behind dust don't get drawn),
  not the magnitude of a star after the fact. Re-dereddening after the
  fact would correct a magnitude that was never artificially reddened to
  begin with.

Only `DataFileBackgroundSource` (a real observed skim) needs
`utils.deredden_dataframe` actually applied — it's the only source whose
magnitudes reflect real dust extinction in the first place.

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

## Stream sources ({py:mod}`streamgoggles.stream_sources`)

`StreamSource` is the protocol for *realizing* a stream's population —
`realize(params, rng) -> DataFrame` of `phi1`/`phi2` (stream-frame
coordinates), a `dist` column (TRUE distance modulus, despite the name —
`streamobs`'s own column convention, kept as-is), the two true-magnitude
columns for whichever bands were requested, and `is_stream=True`. No sky
position yet, no survey noise — purely the population in its own frame.

- `StreamObsSource` wraps `streamobs.model.StreamModel` for two
  morphologies: `"uniform"` (Gaussian cross-track width, uniform
  along-track density over a fixed `length`) and `"spline"` (a piecewise
  track defined by control points, with density still uniform along it —
  non-uniform density profiles aren't exposed, by design, decision 5).
  Both use the plain `StreamModel` class rather than `SplineStreamModel`,
  confirmed by reading `streamobs`'s own source: `SplineStreamModel`
  unconditionally injects a `stream_name` kwarg meant only for its
  *file*-backed interpolation classes, which the inline classes this
  wrapper actually uses don't accept.
- `ExternalSimSource` is a stub for plugging in external (e.g. N-body)
  stream realizations without reshaping them into `streamobs`'s format
  first. Its file schema is already decided even though the
  implementation isn't: **one Parquet file per realization**
  (`data/external_sims/stream_{id}.parquet`), not one directory per
  realization — required columns `phi1`/`phi2`; optional true-magnitude
  columns (used directly if present) and constant-valued `age`/`z`
  columns (repeated per row, rather than a separate sidecar metadata
  file) so one realization stays fully self-contained in one file with
  one read path.

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
window — and separately crop the *stream-only* raw map (`stream_raw`) as
the basis for that channel's label. `inject_background_only` produces the
shape-compatible negative case: no realization, no label computation, just
the cached background cropped to a random window (used whenever
`StreamConfig.background_fraction > 0`) — `label_stack` is all zeros, which
is already correct under every label policy below (nothing ever passes a
positive threshold at zero count).

### Label policies (`StreamInjector.label_policy`)

- `"stream_count"` (the original 2026-09-09 pivot): `label_stack[c]` is
  exactly that channel's cropped `stream_raw` — a literal, non-negative
  star count. Computed directly in `injector.py`, not routed through
  `rasterize.py` at all.
- `"stream_detection"` (2026-09-15, current default): the same `stream_raw`
  crop, hard-thresholded into a binary `{0, 1}` target
  (`stream_raw > count_threshold`, a constructor argument, default 10
  stars). Retargets the network at the actual detection goal instead of the
  literal count — see {doc}`ml_concepts` and {doc}`datasets_and_models` for
  why (count regression under a heavy-tailed target reliably localized
  streams but badly under-recovered their peak amplitude). Both policies
  share the same `stream_raw` computation; only the last step (return it
  directly vs. threshold it) differs.
- The pre-pivot mechanism ({py:mod}`streamgoggles.rasterize`) —
  `rasterize_binary`, `rasterize_density`, a `soft_distance` stub — remains
  implemented and selectable via the same `label_policy`, for a single
  distance-and-filter-independent 2D label computed purely from
  `phi1`/`phi2` geometry (no `stream_raw` involved at all). Not the default
  under either pivot above: it throws away the filter-dependence
  `label_policy="stream_count"`/`"stream_detection"` were specifically
  designed to keep (a "bad" decoy filter should score near zero even at
  pixels a real stream geometrically crosses, since it didn't actually
  *select* real members there).
