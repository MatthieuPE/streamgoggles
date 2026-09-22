# Stream parameters

**Question.** How well can streams be detected across the range of properties
of the known DES streams — distance, width, length, surface brightness,
distance gradient — by a model that is told which trial distance to look at?

**Status.** In progress: the design below is fixed and tested; the generation
is being made faster before the first trainings.

This page records every choice made for this experiment, with the reason for
it. Where a choice was made by the project lead rather than derived from a
measurement, it says so.

## What changed from the earlier experiments

- **Survey:** DES year 6 (`survey="des"`, `release="yr6"`), instead of LSST year
  1.
- **Stream parameters** vary over ranges, instead of one fixed shape at one
  distance.
- **The model answers for one queried trial distance** and sees the matched
  filter at that distance and its two neighbours, instead of one fixed
  distance.

Everything else is the configuration selected by {doc}`hyperparameters`: batch
Dice loss, batch size 8, background fraction 0.05, U-Net of depth 2 and base
width 12, the fixed decoy box, magnitudes 16 to 24.5 in g and r, a count
threshold of one star per pixel for the label.

## The sky

**Survey model.** DES year 6 from streamobs: magnitude-limit maps, completeness,
photometric errors and extinction. Its g depth is 25.0 at the median of the
footprint and between 24.5 and 25.2 over 95% of it, so the 24.5 magnitude cut
is reached almost everywhere.

**Background (decision).** The fast "light" background of streamobs samples
stars from colour-magnitude histograms tabulated as a function of the
magnitude limit. Such tables exist only for LSST. Since the generator uses them
only through the magnitude limit, and DES and LSST g and r are close, the LSST
tables are used for DES: they were copied into
`streamobs/data/background/des/`. This is an approximation of the DES stellar
and galaxy populations, not a DES measurement.

**Study region: RA 0, Dec −50, 25 × 18 degrees.** Chosen as the fully covered
DES region with the lowest extinction (median E(B−V) 0.012) and a typical
depth (g limit about 25.1). The region used before (RA 0, Dec −30) is only 41%
inside DES.

**Two silent defaults were wrong for DES and are fixed** (both tested):

- a stream's true magnitudes are now computed in the photometric system of the
  survey that observes it — it fell back to LSST before;
- the isochrone of the matched filter is drawn in the bands of the catalog it
  selects, read from the column names (`des_yr6` → DES).

## Model input and output

For a queried trial distance modulus `dm`, the model receives seven maps of the
same window:

| channel | content |
|---|---|
| 0 | matched-filter map at `dm − 0.5` |
| 1 | matched-filter map at `dm` |
| 2 | matched-filter map at `dm + 0.5` |
| 3 | decoy map (fixed colour-magnitude box, colour 1.2-1.5, g 18-24.5) |
| 4-6 | the distance moduli of maps 0-2, each as a constant map, scaled `(d − 17) / 2` |

and returns **one** map: the probability that each pixel belongs to a stream
seen through the matched filter at `dm`.

- **Why three distances (decision).** The model answers for one distance but
  sees the stream's signal at the neighbouring ones, so it can use how a stream
  changes with distance — along a distance gradient, for instance.
- **Why the decoy once.** The fixed box does not depend on the trial distance:
  its map is the same at every `dm`.
- **Why constant maps for the distances.** They are the simplest way to give a
  convolutional network a number: the same value everywhere, scaled so that
  distance modulus 17 reads 0 and 15 and 19 read ∓1. They are added after the
  normalization of the maps (next section), so a given distance always reads
  the same.

**Trial distances.** The model can be queried at distance moduli **15, 15.5,
... 19** (9 values). The maps at 14.5 and 19.5 are computed only as neighbours
for the queries at 15 and 19. The scanned range stops at the edges of the
training range on purpose (decision), so that some streams lie at the edge of
what is scanned.

**Which distance is queried in training.** The grid point nearest the stream's
true distance, shifted by `k × 0.5` with `k` drawn uniformly from −2 to +2,
among the points that stay inside 15-19. The model therefore sees windows
queried at the stream's distance and at up to one magnitude from it. A window
without a stream is queried at a distance drawn uniformly on the grid.

**Evaluation** queries each stream at its own distance (the evaluated
distances are grid points).

## The label, and what it means

The label at `dm` is **the stream stars that the matched filter at `dm`
selects, counted per pixel and thresholded at one star** — the same definition
as in every earlier experiment, applied at the queried distance. This was a
decision, taken after the measurement below.

**The matched filter is not blind to other distances, and not symmetrically.**
Measured on stream stars after the survey model:

| stream at 15, filter at | 15 | 16 | 17 | 19 |
|---|---|---|---|---|
| stars selected | 77% | 25% | 9% | 0.2% |

A stream is therefore seen, weaker, by filters at *larger* distances. The other
direction is much cleaner: on a stream half at 16 and half at 18, the filter at
16 selects 518 stars of its own half and 15 of the far half, but the filter at
18 selects 146 of its own half and **69 of the closer half**. A closer stream
has about three times more stars bright enough to be detected, and some of
them fall inside the farther isochrone's selection.

**Consequence of the decision.** The model learns "stars compatible with this
distance", not "a stream at exactly this distance": queried at a distance
larger than a stream's, it can still flag that stream, more weakly. The
alternative considered, and not chosen, was a distance-resolved label that
counts only stars whose own distance is within ±0.25 of `dm`.

Two tests pin this down on the full generation path: a stream at 15 seen by the
filter at 19 leaves the label almost empty (at most 5% of the label at 15), and
the two-distance stream behaves as in the table above.

## Stream parameters in training

Each training stream is drawn from ranges that **bracket** the DES 2018 streams
without matching them (decision: the model should not be tuned to the streams
it will be asked to recover).

| parameter | training draw | DES 2018 streams |
|---|---|---|
| distance modulus | uniform 15-19 | 15.6-18.5 (13-50 kpc) |
| width | log-uniform 0.1-1.5 deg | 0.16-1.16 |
| length | uniform 4-30 deg | 4.8-29.2 (Palca 57) |
| surface brightness | uniform 32-34.5 mag arcsec⁻² | 31.9-34.3 |
| distance gradient | uniform ±0.2 mag/deg, at most 1.5 mag end to end | 0.16 mag/deg for Tucana III |
| age, metallicity | fixed, 12.5 Gyr and Z = 0.0002 | old, metal-poor |

DES 2018 values from Shipp et al. (2018), Tables 1 and 2 (14 streams with
measured parameters).

- **Distance gradient (decision).** Drawn per degree, but the total change
  across the stream is capped at 1.5 mag: a gradient drawn independently of the
  length would otherwise let a 30-degree stream span 6 magnitudes, more than the
  whole scanned range. The distance modulus is the value at the stream's
  centre.
- **Minimum length in a window: 3 degrees** instead of 5, since the shortest
  DES stream is 4.8 degrees long. (Before this experiment the setting was
  fixed at 5 degrees in the code whatever the notebook said; it is now an
  injector setting.)
- **Surface brightness** sets the number of stars from each stream's own width,
  length and central distance.

## Normalization

Each (filter, distance) map is normalized as `(x − mean) / std`, with one mean
and one standard deviation per map, **fitted once** and applied to every
window. It is not a per-window normalization: the absolute density level of the
sky is kept, so a denser patch looks denser to the model.

The statistics are currently estimated from the first 8 training windows. With
query-first generation (below) they will be computed once from the full-sky
background maps built at start-up, which covers the whole footprint and
contains no stream stars.

## Two streams in one window

{doc}`two_streams` showed that single-stream training is enough for neighbours
2 degrees or more apart and for crossing streams, while a faint stream running
parallel within about 1 degree of a brighter one is suppressed. Training here
keeps one stream per window. Wider streams (up to 1.5 degrees) will touch their
neighbours sooner, which a later check should cover.

## Generation cost

Before any optimization, generating one training window takes 0.73 s on one
process. Where it goes:

| stage | time | share |
|---|---|---|
| full-sky maps for 22 channels (11 distances × 2 filters) | 0.34 s | 46% |
| survey observation of the stream (streamobs) | 0.22 s | 30% |
| projecting and cropping 22 maps and 22 labels | 0.06 s | 8% |
| matched-filter selection at 11 distances | 0.05 s | 7% |
| stream realization, number of stars, placement | 0.08 s | 10% |

The model uses 4 of the 22 maps and 1 of the 22 labels. The planned changes, in
this order:

1. **Crop first, one projection per window:** no full-sky arrays, the window's
   geometry computed once and shared by every channel. Output identical.
2. **Decoy once:** one decoy channel instead of 11. Output identical.
3. **Query first:** draw the queried distance before generating the window and
   build only its three matched-filter maps and the decoy; normalization from
   the background maps.

The survey observation (0.22 s) remains as a floor.
