# Stream parameters

**Question.** How well can streams be detected across the range of properties
of the known DES streams — distance, width, length, surface brightness,
distance gradient — by a model that is told which trial distance to look at?

**Status.** First results: two quick (4800-window) and two long
(19200-window) models, each scored at the stream's known position on a grid of
distance × width × surface brightness. The four other seeds the protocol asks
for are not run yet.

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
- **Why the decoy once.** The fixed box does not depend on the trial distance,
  so its map is the same at every `dm`. The decoy map is, like every channel,
  the box applied to **stream and background** stars together: the stream
  stars that fall in the box are added to the background ones. Because the box
  sits redder than any star of the isochrone, the stream contributes almost
  nothing to it, which is what makes it a background reference.
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

## Normalization: what changed

Both the old and the new normalization standardize each channel `c` of a
window, pixel by pixel, over the window's **valid** pixels `V` (those whose
HEALPix neighbours all lie inside the footprint); invalid pixels keep their
fill value 0, and a channel with `σ = 0` is only centred. `x_c(p)` is the star
count of channel `c` in window pixel `p`. What changed is **where the mean and
the standard deviation come from**.

**Before (every earlier experiment and the notebooks, `RobustNormalizer`).**
One mean and one standard deviation per channel were fitted **once**, on the
valid pixels of a fixed set `F` of training windows pooled together (the
first 8 training windows), and then applied unchanged to every window —
training, validation and evaluation alike:

$$
\bar\mu_c = \frac{1}{N_F}\sum_{w\in F}\sum_{p\in V_w} x_{c,w}(p), \qquad
\bar\sigma_c = \sqrt{\frac{1}{N_F}\sum_{w\in F}\sum_{p\in V_w}\big(x_{c,w}(p)-\bar\mu_c\big)^2}, \qquad
N_F = \sum_{w\in F} |V_w|,
$$

$$
x'_{c,w}(p) = \frac{x_{c,w}(p)-\bar\mu_c}{\bar\sigma_c} \quad \text{for every window } w.
$$

A denser patch of sky therefore stayed denser after normalization: the model
could see the absolute density level, as it was on the training background.

**Now (this experiment, `WindowNormalizer`).** The mean and standard deviation
are computed **for each window from that window alone**:

$$
\mu_{c,w} = \frac{1}{|V_w|}\sum_{p\in V_w} x_{c,w}(p), \qquad
\sigma_{c,w} = \sqrt{\frac{1}{|V_w|}\sum_{p\in V_w}\big(x_{c,w}(p)-\mu_{c,w}\big)^2},
\qquad
x'_{c,w}(p) = \frac{x_{c,w}(p)-\mu_{c,w}}{\sigma_{c,w}}.
$$

Every channel of every window then has mean 0 and standard deviation 1 over
its valid pixels. Nothing is fitted or stored, and nothing comes from other
windows or from the background maps.

**Why the change (decision).** On real data the background level is not known
in advance, and the model must not rely on having seen it in training: with
the fitted statistics, the input depends on how the real sky's density
compares with the simulated training background. The per-window version
removes that dependence — doubling every count of a window, or adding a
constant sky level to it, gives exactly the same input (tested).

**What it costs.** The absolute density of the sky is no longer visible to the
model, only each channel's contrast within the window. A stream still changes
the statistics of the window it is in (it adds to both `μ` and `σ`), more so
for bright or wide streams filling a larger part of the window.

The three distance channels are appended after this step and are never
normalized; the label is never normalized.

## Two streams in one window

{doc}`two_streams` showed that single-stream training is enough for neighbours
2 degrees or more apart and for crossing streams, while a faint stream running
parallel within about 1 degree of a brighter one is suppressed. Training here
keeps one stream per window. Wider streams (up to 1.5 degrees) will touch their
neighbours sooner, which a later check should cover.

## First results

Two models of each length (seeds 42 and 43), each scored on 20 injected
streams per grid point, on a background none of them saw: distance modulus
15, 16, 17, 18 and 19 × width 0.2, 0.6 and 1.2 degrees × surface brightness 32,
33 and 34, length 15 degrees, no gradient, each stream queried at its own
distance. Every stream at SB 32 is detected, so the figure shows SB 33 and 34.

```{image} figures/stream_parameters/detection.png
:alt: Fraction of streams detected against distance modulus, for three widths, at surface brightness 33 and 34, for quick and long models
:width: 100%
```

*Fraction of injected streams detected at their known position, pooled over
the two trainings of each length (40 streams per point), with Wilson 68%
bars. Solid: 4800 training windows; dashed: 19200. Each model is thresholded
just above its own background level (see "The operating point" below).*

### At fixed surface brightness, closer streams are harder

**Statement.** Detection rises with distance: at SB 34 and a width of 0.2
degrees, from 2% of streams at distance modulus 15 to 100% at 19. This is a
property of the streams, not of the model.

Surface brightness is light per unit solid angle. A stream of given surface
brightness, angular width and angular length is physically larger the farther
away it is, so it holds more stars (about ∝ distance²). Far fewer of them are
bright enough to be detected, but the survivors still outnumber the closer
stream's, while the background the matched filter picks up at the fainter
magnitudes of a distant isochrone is slightly lower. Measured for SB 33, 0.2
degrees wide, 15 degrees long:

| distance modulus | stars | detected | selected by the filter | stream density in its 1σ band | background in the filter | contrast |
|---|---|---|---|---|---|---|
| 15 | 3,710 | 777 | 597 | 68 /deg² | 573 /deg² | 0.12 |
| 16 | 9,453 | 1,312 | 1,011 | 115 /deg² | 673 /deg² | 0.17 |
| 17 | 24,325 | 2,174 | 1,552 | 177 /deg² | 636 /deg² | 0.28 |
| 18 | 64,066 | 3,561 | 2,262 | 257 /deg² | 511 /deg² | 0.50 |
| 19 | 176,687 | 4,837 | 2,531 | 288 /deg² | 483 /deg² | 0.60 |

The contrast rises five times from distance modulus 15 to 19. This agrees with
the DES 2018 streams: the faintest found (Elqui and Chenab, SB 34.1-34.3) are
the most distant, at 40-50 kpc. **The hard case for this model is a close,
narrow, faint stream**: at distance modulus 15-16, 0.2 degrees wide and SB 34,
2-5% are detected. Among the DES streams, Wambelong (distance modulus 15.9,
0.40 degrees wide, SB 33.7) is the closest to that regime.

**Wider streams are easier at the same surface brightness** for a similar
reason: the density per pixel is the same, but a wider stream covers more
pixels, so it holds more stars in total and gives the network more to add up.
At SB 34 and distance modulus 16: 5% at 0.2 degrees, 30% at 0.6, 52% at 1.2.

### Training longer brings nothing here

**Statement.** The 19200-window models detect no more streams than the
4800-window ones: over every SB 33 and 34 cell, 930 of 1200 streams against 945
(Fisher p = 0.49), and at a fixed threshold of 0.5, 878 against 879. In the
hardest cells the long models are, if anything, slightly lower. Their
validation loss is clearly better (about 0.39 against 0.46), so they fit the
label better without finding more streams.

This differs from {doc}`hyperparameters`, where 19200 windows beat 4800 at a
matched false-alarm rate. The setting has changed on three counts at once —
per-window normalization, a much wider range of streams, and the
query-distance input — so the two results are not in contradiction, but this
one does not say which change removed the gain. For now it supports exploring
with the quick model (two-tier policy), at a quarter of the training time.

### The operating point

On stream-free sky, both kinds of model output an almost constant value — a
background floor, about 0.007 for the quick models and 0.001 for the long ones
— and all of their discrimination happens near streams. Their false-alarm
curve is therefore a step rather than a slope: below the floor every
stream-free pixel is flagged, just above it almost none. "Thresholded to flag
at most 1e-3 of the stream-free sky" therefore means, here, "just above the
model's background floor", where the quick models flag 7e-4 of the stream-free
sky at most and the long ones none at all in a quarter of the cells. Both sit
at essentially zero false alarms, so the comparison above is fair, but the
number 1e-3 describes the target, not the rate actually reached. At a fixed
threshold of 0.5 the models flag 6e-5 (quick) and 2e-6 (long) of the
stream-free sky.

### Caveats

- Two trainings per length, not the six of the protocol.
- One length (15 degrees) and no distance gradient in the evaluation, although
  training covers 4-30 degrees and gradients.
- Detection at the stream's known position, queried at its own distance.
- The background uses LSST colour-magnitude tables (see "The sky").

## Generation cost

Generating one training window originally took **0.73 s** on one process. The
profile showed where:

| stage | before | after |
|---|---|---|
| building the 22 channels (11 distances × 2 filters) | 0.34 s | 0.06 s |
| survey observation of the stream (streamobs) | 0.22 s | 0.22 s |
| matched-filter selection at 11 distances | 0.05 s | 0.05 s |
| stream realization, number of stars, placement | 0.08 s | 0.08 s |
| **total per window** | **0.73 s** | **0.37 s** |

Two changes, both with **identical output** (a test compares every map, label
and validity mask with the previous computation, pixel for pixel):

1. **Crop first.** A window used to be cut out of two full-sky maps per
   channel (about 3 million pixels each at nside 512): one for the stream's
   selected stars, one for stream plus background. Now the window's
   projection — for each window pixel, the HEALPix pixels around it and their
   interpolation weights — is computed once and shared by every channel, the
   stream's stars are placed on HEALPix pixels once, and each channel's
   selected stars are counted only on the pixels the window reads, then added
   to the background map (built once at start-up) there.
2. **Decoy once.** The fixed box's selection of the stream's stars is computed
   once per window instead of at each of the 11 distances, and its background
   map once at start-up instead of 11 times.

What remains is dominated by streamobs observing each stream (60%), which a
window cannot avoid. Building only the three distances a query needs would
save about 15-20% more; it is set aside for now, since each window then
carries every distance and could instead be reused for several queries.

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/experiments/stream_parameters/run.py --seeds 42 43 --windows 4800   # then --windows 19200
python scripts/experiments/stream_parameters/figures.py --contrast                 # figure, tables, contrast
```

`run.py` resumes where it stopped (trained models are reloaded, scored grid
points skipped). Models and results are written to
`data/experiments/stream_parameters/`.
