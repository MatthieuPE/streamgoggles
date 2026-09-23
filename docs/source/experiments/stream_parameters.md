# Stream parameters

**Question.** How well can streams be detected across the range of properties
of the known DES streams — distance, width, length, surface brightness,
distance gradient — by a model that is told which trial distance to look at?

**Status.** First results: six quick (4800-window) models, as the protocol
asks, and two long (19200-window) ones, each scored at the stream's known
position on a grid of distance × width × surface brightness.

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

Six quick models (seeds 42-47) and two long ones (42-43), each scored on 20
injected streams per grid point, on a background none of them saw: distance
modulus 15, 16, 17, 18 and 19 × width 0.2, 0.6 and 1.2 degrees × surface
brightness 32, 33 and 34, length 15 degrees, no gradient, each stream queried
at its own distance. Every stream at SB 32 is detected, so the figure shows
SB 33 and 34.

```{image} figures/stream_parameters/detection.png
:alt: Fraction of streams detected against distance modulus, for three widths, at surface brightness 33 and 34, for quick and long models
:width: 100%
```

*Fraction of injected streams detected at their known position, pooled over
the trainings of each length (120 streams per point for the quick models, 40
for the long ones). **The bars are the Wilson 68% interval of that pooled
fraction: sampling only** — how precisely 120 injections measure a rate (about
±4.5 points at 50%). They say nothing about how much the answer changes from
one training to another, which in the transition cells is far larger and is
the subject of the next section. Solid: 4800 training windows; dashed: 19200. Each model is thresholded
just above its own background level (see "The operating point" below).*

### At fixed surface brightness, closer streams are harder

**Statement.** Detection rises with distance: at SB 34 and a width of 0.2
degrees, from 1% of streams at distance modulus 15 to 98% at 19 (six
trainings, 120 streams per point). This is a property of the streams, not of
the model.

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
1-3% are detected. Among the DES streams, Wambelong (distance modulus 15.9,
0.40 degrees wide, SB 33.7) is the closest to that regime.

**Wider streams are easier at the same surface brightness** for a similar
reason: the density per pixel is the same, but a wider stream covers more
pixels, so it holds more stars in total and gives the network more to add up.
At SB 34 and distance modulus 16: 3% at 0.2 degrees, 30% at 0.6, 57% at 1.2.

### Training longer brings nothing here

**Statement.** The 19200-window models detect no more streams than the
4800-window ones: over every SB 33 and 34 cell, 930 of 1200 streams (2
trainings) against 2814 of 3600 (6 trainings), that is 77.5% against 78.2%,
Fisher p = 0.63; on the two seeds they share, 77.5% against 78.8%. At a fixed
threshold of 0.5, 878 against 879 of 1200 on those seeds. In the hardest cells
the long models are, if anything, slightly lower. Their
validation loss is clearly better (about 0.39 against 0.46), so they fit the
label better without finding more streams.

This differs from {doc}`hyperparameters`, where 19200 windows beat 4800 at a
matched false-alarm rate. The setting has changed on three counts at once —
per-window normalization, a much wider range of streams, and the
query-distance input — so the two results are not in contradiction, but this
one does not say which change removed the gain. For now it supports exploring
with the quick model (two-tier policy), at a quarter of the training time.

### Between trainings, the transition cells swing wildly

**Statement.** Where a cell is neither always nor never detected, the same
configuration retrained gives very different answers: at SB 34, distance
modulus 16 and a width of 1.2 degrees, the six trainings range from **5% to
100%** (pooled 57%).

```{image} figures/stream_parameters/training_spread.png
:alt: One point per trained model at surface brightness 34, against distance modulus, for three widths
:width: 100%
```

*One point per trained model (six 4800-window trainings), each scored on its
own 20 injections; the horizontal bar is the mean of the six. **No error bars
here on purpose**: the quantity of interest is the scatter of the points
themselves, the training-to-training spread, which is much larger than the
±11 points of sampling uncertainty on each single point. Points are offset
horizontally by width.*

| SB 34 | 0.2 deg | 0.6 deg | 1.2 deg |
|---|---|---|---|
| distance modulus 15 | 1% (0-5) | 15% (5-40) | 25% (5-50) |
| 16 | 3% (0-10) | 30% (0-75) | 57% (5-100) |
| 17 | 33% (5-60) | 93% (70-100) | 97% (85-100) |
| 18 | 92% (55-100) | 100% | 100% |

Pooled percentage, with the range over the six trainings in brackets. Away
from the transition every training agrees; inside it, a single model tells you
almost nothing, which is why the numbers on this page are pooled over six. The
same effect dominated {doc}`hyperparameters`, and it is the reason that
experiment recommends comparing configurations only with several seeds each.

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

### Conclusions so far

1. **At fixed surface brightness, distance helps rather than hurts.** A stream
   of given surface brightness and angular size holds more stars the farther
   it is (about ∝ distance²), and the matched filter's background is slightly
   lower at the fainter magnitudes of a distant isochrone: the contrast rises
   from 0.12 at distance modulus 15 to 0.60 at 19. Detection follows: at SB 34
   and 0.2 degrees wide, 1% of streams at distance modulus 15 and 98% at 19.
2. **Wider streams are easier at the same surface brightness**, since the
   density per pixel is unchanged but more pixels carry it: at SB 34 and
   distance modulus 16, 3% at 0.2 degrees, 30% at 0.6, 57% at 1.2.
3. **The hard regime is close, narrow and faint** — distance modulus 15-16,
   0.2 degrees, SB 34 — where 1-3% of streams are found. Everything at SB 32
   is found everywhere, and everything beyond distance modulus 18 is found at
   SB 33 and 34 as well.
4. **Training four times longer brings nothing here** (77.5% against 78.2%
   over the SB 33-34 cells, p = 0.63), although the long models fit the label
   better (validation loss 0.39 against 0.46). Explore with the quick model.
5. **In the transition cells, one training tells you almost nothing**: the six
   trainings of the same configuration range from 5% to 100% at SB 34,
   distance modulus 16 and 1.2 degrees. Pool several trainings before reading
   any number there.
6. **These models say "background" with one almost constant value**, so their
   false-alarm curve is a step and the usable operating point is just above
   that floor, at essentially no false alarms. Threshold tuning will have to
   work with that, not with a smooth trade-off curve.

7. **Simulated with their own parameters, 12 of the 14 DES 2018 streams are
   recovered in at least 97% of injections** at their known positions, and the
   two exceptions (Wambelong 49%, Aliqa Uma 80%) are the ones the grid points
   to. See the next section for what that does and does not mean.

## The DES 2018 streams, simulated

The grid above varies one parameter at a time; the known streams differ in
several at once. So each of the 14 DES streams with measured parameters
(Shipp et al. 2018) is simulated **with its own width, length, distance and
surface brightness**, injected at random positions in the study region, and
scored at its known track by the six quick models — 20 injections per model,
120 per stream. Training never sees these values: it draws from the ranges
above, so the model is not tuned to the streams it is asked to recover.

```{image} figures/stream_parameters/des_streams.png
:alt: Fraction of injections recovered for each of the 14 DES 2018 streams, with the range over the six trainings
:width: 100%
```

*Bars: the fraction recovered, pooled over the six trainings (120 injections
per stream). **The horizontal lines are not error bars: they span the lowest
and the highest of the six trainings' own rates** (20 injections each). So
Wambelong's line from 15% to 85% means one training found 3 of its 20
injections and another 17, not that the pooled 49% is uncertain by that much —
sampling alone would give about ±4.5 points on 120 injections. Each stream is
queried at the grid distance nearest its own.*

| stream | m−M | width | SB | recovered (range over trainings) |
|---|---|---|---|---|
| Tucana III, Molonglo, Indus, Ravi, Chenab, Turbio, Willka Yaku | 16.1-18.0 | 0.18-0.83 | 31.9-34.1 | **100%** |
| Elqui | 18.5 | 0.54 | 34.3 | 99% (95-100) |
| ATLAS | 16.8 | 0.24 | 33.0 | 98% (95-100) |
| Phoenix | 16.4 | 0.16 | 32.6 | 98% (95-100) |
| Jhelum | 15.6 | 1.16 | 33.3 | 97% (85-100) |
| Turranburra | 17.2 | 0.60 | 34.0 | 97% (85-100) |
| Aliqa Uma | 17.3 | 0.26 | 33.8 | 80% (35-95) |
| **Wambelong** | **15.9** | **0.40** | **33.7** | **49% (15-85)** |

**Statement.** Simulated with their own parameters, **12 of the 14 streams are
recovered in at least 97% of injections**, including the two faintest ones
(Chenab at SB 34.1 and Elqui at SB 34.3), which are also the most distant. The
two that fall short are the ones the grid predicts: **Wambelong** (49%), close
at distance modulus 15.9, narrow at 0.40 degrees and faint at SB 33.7, sits in
the hard corner; **Aliqa Uma** (80%) is next to it. Both also vary most
between trainings (15-85% and 35-95%), so a single trained model is not enough
for them.

**What this does and does not say.** It says that, in these simulations and at
known positions, a model trained on ranges rather than on the streams
themselves recovers almost all of them. It does not yet say they will be
recovered in DES data: the background here is simulated (from LSST
colour-magnitude tables, see "The sky"), every stream is given this
experiment's isochrone rather than its own population, the five streams longer
than the study region are evaluated on a 15-degree segment, and nothing here
accounts for the extinction, depth variations, crowding or the real
overdensities of the DES footprint.

### Caveats

- Six trainings for the quick models, two for the long ones.
- The DES streams are simulated with this experiment's isochrone, not their
  own populations, and the longest five on a 15-degree segment.
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
