# Training on the real DES background

Every model so far was trained on a simulated sky. This experiment trains the
adopted configuration on the **real DES Y6 sky**, with every known stream,
globular cluster and dwarf galaxy masked out of it, then runs it over the whole
DES footprint with the known streams **put back**, to see whether it finds them.

**Status: trained and run** (2026-09-25): twelve models, the nine maps, and a
first test of the DES 2018 streams. The design comes first on this page, the
results after it, from "Results: the maps" on.

**In short: 5 of the 14 DES 2018 streams are detected** — Elqui, Tucana III,
Willka Yaku, ATLAS and Chenab, each peaking at its own distance. Three of the
nine others are not in the matched-filter maps at all; the rest are, but lie
nearer than m−M 16.5, where the network is least sensitive, as the
simulations had warned.

What it is meant to deliver:

1. a HEALPix map of the network's output over the whole DES footprint, one
   per queried distance modulus from 15 to 19 (and, later, an animation
   through them);
2. then, whether each DES 2018 stream is detected in those maps.

## The two skies

Both come from `scripts/real_data/background.py`, from the same downloaded
DES Y6 Gold stars with the same cuts (S/N > 5, 16 ≤ g, r ≤ 24.5; see
{doc}`../narrative/real_des_background`). They differ only in what is masked:

| | training sky (`des_yr6_background`) | inference sky (`des_yr6_inference`) |
|---|---|---|
| DES 2018 streams | masked, along their DES tracks | **left in** |
| Sagittarius | masked | masked — no search is run inside it |
| globular clusters, dwarf galaxies | masked | masked |
| area | 4,017 deg² | 4,690 deg² |
| stars | 31,649,940 | 38,058,093 |

Training on a sky that still held the known streams would teach the model to
call them background. Inference on a sky without them could not find them.

## Two folds, so no pixel is predicted by a model that trained on it

### Why

**A detection is a comparison.** The network's output is never read on its
own: a stream counts as detected when the output along its track stands out
from the output on **stream-free sky**. The threshold is the level that fewer
than 10⁻³ of stream-free pixels exceed, and the signal-to-noise compares the
track with stream-shaped bands drawn on stream-free sky. A detection therefore
means "the network responds more to this stream than to sky without one", and
it is only as fair as that comparison.

**Training bends the sky it trains on.** Every training window is labelled
zero everywhere except the injected stream. One training shows the model
4,800 windows of 96 × 96 pixels, about 44 million pixel-views, over some
300,000 sky pixels: **each background pixel is shown to the model about 150
times, each time labelled "nothing here".** Whatever real structure sits in
that background — a survey artefact, a depth pattern, an unmasked stream no
one has found yet — the model is trained, repeatedly, to answer zero on.

**Without folds, the comparison is tilted.** At inference the known streams
are new to the model, since they were masked in training. But the stream-free
sky they are compared against is exactly the sky the model was drilled on. It
looks quieter than sky the model has never seen, so the threshold comes out
lower and the signal-to-noise higher than they should be: a stream could be
declared detected against a background made artificially calm. With two
folds, a stream and the background it is compared with are **both** predicted
by models that never saw them.

**And unknown streams would be taught away.** Without folds, a real stream
nobody has found yet, sitting in the training sky, is taught to the model as
background about 150 times over, so the model learns to miss exactly what a
search is for. With folds it is taught as background only to its own fold's
models, and predicted by the other fold's, which never saw it. That matters
less for recovering the DES 2018 streams, which are masked either way, than
for any discovery later.

### How

- **Alternating stripes of right ascension**, 20° wide
  (`objects_overlap.spatial_fold`):

  ```
  RA   0°──20°──40°──60°──80°──100° ...   and 340°–360°, 300°–320° ... the other way
       │ A  │ B  │ A  │ B  │ A  │
  ```

  The six models of fold A train on the A stripes only and predict the B
  stripes; the six of fold B train on B and predict A. Every pixel of the final
  map comes from the ensemble that never trained on it.
- **Stripes rather than two halves**, because the sky is not uniform: the star
  density climbs steeply toward the galactic plane at the east and west ends
  of DES. With a left-right split one ensemble would train only on the dense
  side and be asked about the sparse one. Stripes give both folds the whole
  range of galactic latitude. At Dec −50° a stripe is about 13° across, wider
  than an 11° window. Fold A holds 163,107 pixels (2,140 deg²) of the training
  sky, fold B 143,139 (1,880 deg²).
- **By pixel, not by star.** Each star goes with the HEALPix pixel it falls
  in, and the pixel with its centre. A first version split the stars by their
  own positions, which left 1,177 pixels on the stripe edges partly in each
  fold — pixels that the model later asked to predict them had half-trained
  on. A test holds stars on both sides of an edge in one pixel and checks they
  land in the same fold.
- **In training**, streams are injected on the fold's own pixels; any part of
  a stream crossing into the other fold falls on empty pixels, zeroed in input
  and label alike, exactly like a hole in the footprint.

One nuance. To predict a pixel, a model reads the whole 11° window around it,
so for pixels near a stripe edge the window reaches into stripes the model
*did* train on. What is guaranteed is that the pixel being judged was never
trained on, not that nothing in its neighbourhood was. It concerns a band a
few degrees wide along each edge, and stream and background pixels alike, so
it does not tilt the comparison.

### What it costs, and how big the effect is

Twice the training, and each model sees half the sky. Training windows near a
stripe edge see an artificial straight hole. And if the two ensembles end up
calibrated slightly differently, the output map could show the stripe pattern
itself; that is checked on the maps, and a threshold can be set per fold if it
appears.

How much a model's output on its own training sky actually differs is not
known in advance: it may be small, since each pixel is seen under many
rotations and window positions. The folds measure it for free. Both ensembles
run on every tile anyway, so inference also writes each ensemble's output on
its **own** stripes (`maps/in_fold/`), and the summary compares the two on
stream-free sky. If the difference proves negligible, later runs can train
once on the whole sky.

## The configuration

Everything but the sky is what the experiments adopted:

| | value | from |
|---|---|---|
| matched filter | 13 Gyr, Z = 0.0002, Marigo2017, **DES Y6 error model at one sigma** | {doc}`stream_parameters`, {doc}`matched_filter_errors` |
| decoy channel | fixed box, g − r 1.2-1.5, g 18-24.5 | {doc}`hyperparameters` |
| channels built | 11 distances, 14.5 to 19.5 by 0.5, × (filter, decoy) | {doc}`stream_parameters` |
| model input | the filter maps at dm − 0.5, dm, dm + 0.5, the decoy, the three distances | {doc}`stream_parameters` |
| normalization | each window standardized from itself, channel by channel | {doc}`stream_parameters` |
| queried distances | 15 to 19 by 0.5 | {doc}`stream_parameters` |
| training streams | SB 32-34.5, width 0.1-1.5° (log), length 4-30°, m−M 15-19, gradient ±0.2 mag/deg (≤ 1.5 mag in all) | {doc}`stream_parameters` |
| stream population | **age 9-13.5 Gyr and Z 0.0001-0.001 drawn** | {doc}`stream_parameters`, conclusion 10 |
| network | U-Net, depth 2, base width 12, sigmoid head | {doc}`hyperparameters` |
| loss, batch, learning rate | batch Dice, 8, 2e-3; 40 epochs | {doc}`loss_selection` |
| background-only windows | 5% | {doc}`loss_selection` |
| training length | 4800 windows (the quick tier) | {doc}`hyperparameters` |
| deployment | six trainings per fold, outputs averaged | {doc}`stream_parameters`, conclusion 9 |

Two things change beyond the sky itself. The matched filter is now the
calibrated one: the stream-parameters experiment kept the LSST-error filter
because its trained models depend on it. And the models see the whole
footprint rather than a 25° × 18° patch. The training loop, the grids and the
ensemble are the stream-parameters experiment's own code, imported, so the two
experiments differ only where this section says.

## Inference

The inference sky is tiled with 11° windows at half a window's stride: 167
tiles over 4,690 deg². Each tile's 22 channels are cut out once and shared by
all nine queried distances and both ensembles. Each ensemble's tiles are
stitched back onto HEALPix keeping, for every pixel, the tile in which it sits
furthest from an edge, and the two stitched maps are then combined by fold as
above.

For each queried distance this writes
`data/experiments/real_des/maps/prediction_dm<dm>.fits`, an nside 512 map of
the ensemble's output, with UNSEEN outside the inference sky, plus a summary.
The tiles cover 98.6% of the inference sky (352,500 of 357,639 pixels); the
rest sit at the edge of the footprint or of a hole, where interpolating reads
a neighbour outside the sky and the pipeline marks the pixel invalid.

## Checks before running

An end-to-end trial on 400-window models (meaningless as models) exercised
every step: both folds' skies build, with no pixel outside its fold; training
runs on each; inference writes all nine maps.

Cost: a window takes 0.60 s to generate on this sky in one process, about
0.15 s with the four data-loading workers. A 4800-window training should take
13-15 minutes, like the 12.9 minutes the same configuration took on the
simulated sky, so about three hours for the twelve, and five minutes for
inference. The background maps are built once per fold and cached, about a
minute and a half each. Each data-loading worker receives a 297 MB copy of the
dataset, after dropping the catalogue (nothing downstream reads it) and
storing the count maps as float32, which is exact for counts.

## Results: the maps

```{image} figures/real_des/prediction.gif
:alt: The network's output over the DES footprint, stepping through the queried distance moduli from 15 to 19
:width: 100%
```

*The out-of-fold ensemble output over the inference sky, one frame per
queried distance modulus from 15 to 19; the DES 2018 tracks are outlined in
orange. Sagittarius, the clusters and the dwarfs are masked (white).*

```{image} figures/real_des/prediction_dm16.5.png
:alt: Network output at m-M 16.5
:width: 100%
```

*m−M 16.5 (20 kpc). ATLAS is the dark line on its track near RA 15-30°,
Dec −25° to −33°. The strips along the edge of the Sagittarius mask and at the
footprint's edges are real structure the masks left, taken up below.*

```{image} figures/real_des/prediction_dm18.5.png
:alt: Network output at m-M 18.5
:width: 100%
```

*m−M 18.5 (50 kpc): the Magellanic Clouds' distance, and the southern edge of
the footprint lights up toward them. Elqui and Chenab are at their own
distances here; rings surround the masks of the Sculptor and Fornax dwarfs.*

### How much the folds mattered

The ensembles also predicted their own training stripes, which is what a
model trained on the whole sky would have done. On stream-free sky, the
fraction of pixels above 0.5:

| m−M | from the ensemble that never saw the pixel | from the one that trained on it | ratio |
|---|---|---|---|
| 15.0 | 0.42% | 0.15% | **2.8×** |
| 15.5 | 0.52% | 0.18% | **2.9×** |
| 16.0 | 1.13% | 0.69% | 1.6× |
| 16.5 | 2.16% | 1.78% | 1.2× |
| 17.0 | 3.18% | 2.83% | 1.1× |
| 19.0 | 3.21% | 2.95% | 1.1× |

A model scored on its own training sky would have shown a false-alarm rate
**nearly three times too low at m−M 15-15.5**, and about 10% too low beyond
m−M 17. The folds were worth their cost.

### What the stream-free sky still holds

About 3% of the training mask's sky was above 0.5 beyond m−M 17 — far more
than any simulated sky gave. The maps show why: it is not noise but real
structure the masks left. The fraction of flagged pixels around each
structure, at its own distance, against distance from it:

| structure | flagged pixels by distance | radius in the training mask | calibration radius |
|---|---|---|---|
| LMC | 67% at 9-12°, 52% at 12-15°, 9% at 15-18°, 3% (background) at 18-21° | not masked | **20°** |
| SMC | 90% at 6-9°, 23% at 9-12°, 1% at 12-15° | not masked | **12°** |
| Sagittarius | 45% at 6-7° from its track, 8% at 8-9°, background at 9-10° | 6° | **9°** |
| Sculptor, Fornax | 37% and 18% at 1-2°, nothing beyond | 0.94°, 1.5° (5 half-light radii) | **12 half-light radii** |
| NGC 1904, NGC 1261 | 72% and 86% within 1°, about 4% at 1-2° | not selected: their centres sit in Gold's foreground holes | **2°**, for any cluster within 2° of the footprint |

The Magellanic Clouds light up at exactly their distance, and the dwarfs'
stars extend past their masks — the network is finding real stellar
structure. For measuring false alarms, these are removed: the **calibration
sky** is the training mask's sky minus these regions, 3,331 of its 4,017 deg².
On it, the fraction above 0.5 falls to 0.2-0.8% at every distance, while the
removed 17% of sky held up to 18% flagged pixels: **80-85% of the apparent
false alarms beyond m−M 16.5 were these structures**. The training itself
still saw them, taught as background; whether masking them in training too
changes the models is left for a later run.

## Results: the DES 2018 streams

### The test

The criterion of the simulated experiments, adapted to real tracks
(`evaluation.footprint`: `false_alarm_map`, `track_band`,
`real_track_statistics`):

- **A false-alarm-rate map per distance.** Each pixel gets the fraction of
  calibration pixels *of its own fold* that the network scored at least as
  high. The two folds' ensembles are calibrated separately, so one cut then
  means the same thing everywhere; ties count against the pixel.
- **Flagged**: a false-alarm rate of 10⁻³ or less.
- **The band**: pixels within one width of the stream's DES track. One track
  per stream: Shipp et al. (2018, 2019) where `galstreams` carries it — for
  Chenab its DES segment, not the whole Orphan-Chenab stream its mask
  covers — else the reference matching DES's length (ATLAS: Li et al. 2021,
  23.6° against 22.6°).
- **The null bands**: the band's pixels moved rigidly onto 200 random places
  and orientations of the calibration sky, kept when at least 90% land on it.
- **Detected**: at least 20 flagged pixels in the band, and a flagged density
  standing out from the null bands at S/N ≥ 2 (`band_snr`), at the queried
  distance nearest the stream's own.

### The result: 5 of 14

| stream | m−M | queried | flagged / band pixels | S/N | |
|---|---|---|---|---|---|
| Elqui | 18.5 | 18.5 | 283 / 952 | 96 | ✔ |
| Tucana III | 17.0 | 17.0 | 43 / 104 | 43 | ✔ |
| Willka Yaku | 17.7 | 17.5 | 64 / 215 | 39 | ✔ |
| ATLAS | 16.8 | 17.0 | 146 / 843 | 21 | ✔ |
| Chenab | 18.0 | 18.0 | 48 / 1201 | 7.5 | ✔ |
| Molonglo, Wambelong, Turbio, Aliqa Uma, Turranburra, Phoenix, Indus, Ravi, Jhelum | 15.6-17.3 | | 0 flagged | < 0 | ✘ |

```{image} figures/real_des/snr_by_distance.png
:alt: S/N along each stream's track at every queried distance
:width: 100%
```

*S/N along each stream's track at every queried distance; dashed, its
catalogued distance. The five detections peak at or next to it — Elqui at
18.5 exactly, ATLAS at 16.5 for 16.8, Chenab at 18.5 for 18.0 — which is what
ties each detection to its stream. Turbio and Turranburra light up only at
18.5-19, far from their own distances: Turbio's southern end runs within about
13° of the SMC, and neither excess is the stream.*

### Why nine are missed

Two measurements separate a stream the network misses from one the input
does not contain: the classic matched-filter significance of the stream in the
counts the network reads (the band against side bands two to four widths
away), and the network's own S/N.

```{image} figures/real_des/input_vs_network.png
:alt: The network's S/N against each stream's S/N in the matched-filter counts, coloured by distance
:width: 90%
```

*Each stream's S/N in the input counts (x) against the network's (y), coloured
by its distance. Every detected stream is at m−M 16.8 or beyond (dark); every
stream visible in the input but missed is nearer than 16.5 (pale).*

They fall into three groups:

- **Not in the input at all**: Ravi (input S/N −1.0), Aliqa Uma (−1.8) and
  Jhelum (−2.8). Along their DES tracks, with this filter and these cuts, the
  matched-filter counts show no excess; no model reading these maps could find
  them. Why — the tracks, the filter, the depth — is open.
- **An excess, but not at one distance**: Turbio's excess is 6-9% at every
  channel distance from 14.5 to 19.5, and Molonglo's 8-11% from 14.5 to 17.5.
  A stream is concentrated at its distance; a flat excess is a density
  difference across the band, and the network declining to call it a stream
  is arguably right. Wambelong and Turranburra show marginal inputs (S/N 3.8
  and 4.9).
- **In the input, and missed**: Phoenix (input S/N 11.7, contrast 22%) and
  Indus (12.4, 6%). Phoenix is the clearest case: its excess peaks at its own
  distance and falls off on both sides, the same profile as ATLAS, and the
  counts show it as a thin line — yet the network responds only in patches,
  reaching false-alarm rates of about 10⁻², not 10⁻³. It is not the
  per-pixel statistic: averaging the output over the whole band gives Phoenix
  S/N 1.1 and Indus 0.3, against 11-28 for the detected streams.

**The pattern is distance**: everything detected lies at m−M ≥ 16.8,
everything visible but missed at m−M ≤ 16.4. It is the weakness the
simulations measured ({doc}`stream_parameters`, conclusion 3: at fixed surface
brightness, closer streams are harder, and m−M 15-16 is the hard regime), now
seen on real streams.

## Conclusions so far

1. **5 of the 14 DES 2018 streams are detected** on real DES Y6 data by a model
   that never saw them — Elqui, Tucana III, Willka Yaku, ATLAS, Chenab — each
   peaking at its own distance.
2. **The fold design mattered**: scored on its own training sky, the model's
   false-alarm rate is nearly three times too low at m−M 15-15.5.
3. **The masks were too small for real structure**: the Magellanic Clouds'
   outskirts, Sagittarius beyond 6°, the Sculptor and Fornax dwarfs, and
   bright clusters hidden in Gold's holes. They made most of the apparent
   false alarms; a calibration mask removes them for measurement.
4. **Of the nine missed**, three (Ravi, Aliqa Uma, Jhelum) are absent from the
   matched-filter input itself, and two (Turbio, Molonglo) show excesses that
   are not concentrated at one distance. **Phoenix and Indus are in the input
   but missed**: both lie nearer than m−M 16.5, where the network is least
   sensitive.

## What comes next

- **Nearby streams**: the network's sensitivity below m−M 16.5 is now the
  limit on the streams that are in the data. Options: a training that weights
  nearby streams more, the longer (19200-window) tier there, or pairing the
  network with the plain matched-filter test at short distances, where the
  input alone already finds Phoenix and Indus.
- **The masks in training**: retrain with the calibration mask's regions
  removed from the training sky too, so the Magellanic outskirts and dwarf
  rings are no longer taught as background.
- **Ravi, Aliqa Uma, Jhelum**: why the filter counts show nothing along their
  DES tracks.

## Reproducing

From the repository root, in the `streamml` environment, with both skies built
(`python scripts/real_data/background.py --write`):

```bash
python scripts/experiments/real_des/run.py train --fold 0    # six models, ~1.5 h
python scripts/experiments/real_des/run.py train --fold 1
python scripts/experiments/real_des/run.py infer             # nine maps, ~5 min
python scripts/experiments/real_des/run.py figures           # a map per distance, the GIF
python scripts/experiments/real_des/run.py calibration       # the calibration mask
python scripts/experiments/real_des/run.py detect            # the DES 2018 test, its figures
```

Each finished model is saved as it completes, so a run resumes where it
stopped.
