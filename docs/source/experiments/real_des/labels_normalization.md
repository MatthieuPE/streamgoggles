# Labels and normalization, on patches of the real sky

The first real-data models found 5 of the 14 DES 2018 streams, answered with
compact blobs, and saw nothing nearer than m−M 16 ({doc}`recovery`). Before
training on the whole sky again, this page goes back to quick models on two
patches of the real DES sky, to find out which part of the data preparation
teaches the model blobs, and whether normalizing by the decoy channel helps.

**Status: done (2026-09-29, branch `des-clean-training`).**

**In short.** Keep the per-window normalization: normalizing by the decoy
fails (the decoy's density follows Galactic latitude, the isochrone channels'
does not), and by counting noise is 14-17 points worse. The band label does
what it was designed to — every label one elongated band, and the model's own
false alarms elongated rather than blobs — but it does not beat the count
label overall (79-80% of the DES 2018 copies against 82%): it wins on wide,
near streams (Jhelum 88% against 12%) and loses on narrow ones (Phoenix 38%
against 88%), and it fires on empty sky forty times more often. Train on a
wide sky: one 600 deg² patch does not transfer.

## Why the label: an audit of the first training's windows

The first training's label marks a pixel as "stream" when the stream's stars
selected by the matched filter, interpolated onto the window, exceed one per
pixel. For a dense or distant stream that is a clean band. For a near or
sparse one, only where the stars happen to clump: at a fixed surface
brightness, a nearer stream puts fewer stars into the filter's magnitude range.

400 windows drawn exactly as the first training drew them (fold 0 of the real
training sky, the population training set, the query transform):

| | |
|---|---|
| labelled windows whose largest labelled piece is < 2° | **33%** (< 1°: 24%) |
| at m−M 15-17 / 17-18 / 18-19 | **58%** / 17% / 7% |
| streams below 1,000 stars per deg² | 80-100% |
| median number of separate labelled pieces | 10 |
| windows at least 90% unmasked: largest piece < 2° | 30% |

So the model was shown, a third of the time and more than half the time for
near streams, a few scattered clumps labelled "stream". That is both the blobs
it answers with and its blindness to near streams. Masked sky, by contrast,
changes little: windows with almost no masked sky have as many clumpy labels.

## What changed

All of it optional, so the first training stays reproducible:

- **A band label** (`StreamInjector(label_policy="stream_band")`):
  - *shape*: every valid pixel within one width of the stream's track, along
    its length, in the frame it was placed with — always one elongated band;
  - *visibility, per channel, from the data*: S = the stream's selected stars
    in the band, B = the real sky's stars in the same pixels and channel; the
    channel is labelled with the band when S / √B ≥ 2, and left empty
    otherwise. Unlike the per-pixel threshold, this judges the stream as a
    whole, which is how a stream is seen, so a faint wide stream spread below
    one star per pixel can still be labelled;
  - *a window is accepted* when, at the channel nearest the stream's distance,
    S / √B ≥ 2, and the band covers at least 4° of track on valid sky
    (all of the stream if it is shorter);
  - *a stream visible nowhere* after several placements raises
    `StreamInvisible`, and training draws another stream
    (`StreamMapDataset.invisible_draws` counts them).
  One distance per stream, as in DES 2018: the input already carries the
  filter maps at ±0.5 mag of the queried distance, enough to follow the small
  distance gradients real streams have.
- **A decoy normalizer** (`DecoyNormalizer`): every channel of a window
  standardized with the decoy channel's mean and spread, instead of its own.
  The decoy box holds background stars only, so a stream neither inflates the
  spread it is divided by nor is centred away, and the isochrone channels keep
  their level relative to the background. Still scale-free: doubling every
  count changes nothing.
- **Masks of nearby massive galaxies and two artefacts**
  (`objects_overlap.contaminant_mask`): 2MASS galaxies of Tully (2015) with
  v < 3,500 km/s and log L_K > 10.8, masked to 1° (30 in the footprint:
  the Fornax core, NGC 1407, NGC 1332…), and the two survey artefacts, whose
  excess is equal in every colour and magnitude selection. Their catalogue is
  `data/others/nearby_galaxies_tully2015.csv`. Tucana III stays masked along
  its DES 2018 track only.
- **Patches of the real sky** (`PreparedCatalogBackgroundSource`): a
  `StudyRegion` keeps only the stars whose pixel lies inside it, and an
  `exclude` mask drops the stars of masked pixels, which makes that sky
  invalid.

## The experiment

- **Sky**: the real DES Y6 training sky (known streams, clusters, dwarfs
  masked) minus the galaxy and artefact mask. Patch **A**, 30° × 20°
  centred on (60°, −45°), for training; patch **B**, 30° × 20° centred on
  (40°, −20°), for evaluation — different stars, different sky.
- **Configurations**: label (count, band) × normalization (window, decoy,
  and in round 2 Poisson), each trained twice (seeds 42, 43) on the quick
  tier (4,800 windows of 96 × 96 pixels, 11°), everything else as in
  {doc}`first_training`.
- **Scoring on patch B**, as the real data are scored: each model's output on
  the stream-free patch sets its false-alarm rates; a stream is detected with
  ≥ 20 pixels at a false-alarm rate ≤ 10⁻³ within one width of its track and
  S/N ≥ 2 against 200 null bands, at the queried distance nearest its own.
  Two sets of injected streams, 8 random placements each:
  - the 14 DES 2018 streams with their Table 1 parameters and populations
    (lengths capped at 15° to fit the patch);
  - a distance scan: m−M 15 to 19, width 0.3°, 10° long, surface brightness 33.
  Each seed's model is scored, and so is their average.

`scripts/experiments/real_des/patches.py`; results in
`data/experiments/real_des/patches/`.

## The labels compared

On 300 training windows of patch A for each label:

| | count label | band label |
|---|---|---|
| labels whose largest piece is < 2° | 34% (58% at m−M < 17) | **0%** |
| median number of pieces | 10 | **1** |
| empty labels, query at the stream's distance / 0.5 mag off / 1 mag off | 1% / 2% / 1% | 0% / 6% / 23% |
| streams redrawn as invisible | — | 4 of 300 |

```{image} ../figures/real_des_patches/label_clumps.png
:alt: Fraction of labels made of clumps against distance, count label and band label
:width: 60%
```

```{image} ../figures/real_des_patches/label_examples.png
:alt: Four injected streams, their input channel, their count label and their band label
:width: 80%
```

*The same four streams, the same windows: input at the stream's distance, the
count label, the band label. At m−M 15.4 the count label is a cloud of
scattered pixels; the band label is the stream.*

The band label follows the query as intended: empty when the queried distance
is too far for the stream to stand out, which the count label, marking any
pixel with a stray selected star, almost never is.

## Round 1: one patch is too narrow a sky

Models trained on patch A and scored on patch B recovered at most 15-20% of
the DES 2018 copies, and none with the decoy normalization:

| | DES 2018 copies recovered (ensemble) | distance scan |
|---|---|---|
| count / window | 15% | 32% |
| count / decoy | 1% | 0% |
| band / window | 19% | 22% |
| band / decoy | 0% | 0% |

These numbers do not compare the labels: **the models do not transfer from
patch A to patch B**. On patch B's stream-free sky their output is above 0.5
on 9-13% of pixels with the window normalization and on **60%** with the
decoy normalization, against 0.1-4% on patch A, the sky they trained on — so
the false-alarm cut on patch B sits at an output of 1.0, and nothing passes
it (the saturation seen in {doc}`recovery`, now everywhere).

| | patch A (trained on) | patch B |
|---|---|---|
| decoy stars per pixel | 36 | 21 |
| isochrone channel at m−M 17 over the decoy | 0.23 | 0.37 |
| stream-free pixels above 0.5, window normalization | 1-4% | 9-13% |
| stream-free pixels above 0.5, decoy normalization | 0.1-1% | 60% |

The decoy box counts red disk dwarfs, whose density follows Galactic latitude
(patch A at b ≈ −48°, patch B at b ≈ −65°); the isochrone channels count a
different population. Standardizing them by the decoy gives them a level that
changes with the sky, and a model trained on one level fires wherever the
level differs. Two lessons, both carried into round 2:

- **train on a wide sky**: the cost of a quick model is its number of windows,
  not the area it is drawn from, so round 2 trains on all of fold 0's
  training sky (2,094 deg² after the galaxy mask) and is scored on fold 1's
  calibration sky;
- **normalization**: the decoy is kept in round 2, to see whether a wide sky
  teaches the model the level's variation; and a third one is added,
  `PoissonNormalizer`: each channel's excess over its window mean in units of
  its own counting noise, (x − μ)/√μ — an absolute significance per pixel,
  which a stream cannot inflate (unlike the window's spread) and which
  borrows no other population's level (unlike the decoy).

```{image} ../figures/real_des_patches/recovery_A.png
:alt: Round 1 recovery on patch B per configuration
:width: 90%
```

*Round 1, for the record: recovery on patch B, where every model's cut is
saturated.*

## Round 2: trained on fold 0, scored on fold 1

The same configurations — plus the Poisson normalization, and in a round 2b
the band label with a stricter visibility cut, S/N ≥ 5 — trained on all of
fold 0's training sky and scored on fold 1's calibration sky. Now the
false-alarm cuts of the count-label models are no longer saturated, and the
comparison measures the models.

```{image} ../figures/real_des_patches/recovery_fold0.png
:alt: Recovery of the DES 2018 copies per configuration, and against distance
:width: 100%
```

*Left: DES 2018 copies recovered on fold 1 (bar: the two seeds' average;
dots: each seed). Right: the distance scan (ensemble). The band/window line
lies under the band5/window one: their scans are identical.*

```{image} ../figures/real_des_patches/false_alarms_fold0.png
:alt: How often each configuration fires on stream-free sky, and how blob-like those answers are
:width: 100%
```

*On fold 1's stream-free calibration sky (medians over the queried
distances): the share of pixels above 0.5, and the share of those in
connected groups shorter than 2°.*

| configuration | DES 2018 copies (seeds) | m−M 15 / 16 / 17 / 18 / 19 | empty sky > 0.5 | of which blobs | saturated cuts |
|---|---|---|---|---|---|
| **count / window** | **82%** (83, 81) | 12 / **100** / **100** / 100 / 100% | **0.1%** | 100% | 1 of 9 |
| band / window | 79% (71, 79) | **25** / 62 / 88 / 100 / 100% | 4.1% | **41%** | 9 of 9 |
| band5 / window (S/N ≥ 5) | 80% (72, 80) | **25** / 62 / 88 / 100 / 100% | 3.7% | 45% | 9 of 9 |
| count / poisson | 68% (72, 67) | 0 / 38 / 100 / 100 / 100% | 0.4% | 80% | 2 of 9 |
| band / poisson | 62% (46, 62) | 25 / 38 / 75 / 100 / 100% | 3.5% | 36% | 9 of 9 |
| band / decoy | 31% (0, 24) | 25 / 12 / 62 / 38 / 12% | 11% | 13% | 9 of 9 |

*"Saturated cuts": queried distances at which the output at a false-alarm
rate of 10⁻³ is above 0.98 — where the model's false alarms are as confident
as a stream can be.*

**Stream by stream** (ensemble), the two labels are complementary:

| | count / window | band5 / window |
|---|---|---|
| Jhelum (1.16° wide, m−M 15.6) | 12% | **88%** |
| Wambelong (m−M 15.9) | 12% | 25% |
| Phoenix (0.16° wide) | **88%** | 38% |
| Aliqa Uma (0.26° wide) | **75%** | 38% |
| Willka Yaku (0.21° wide) | **100%** | 75% |
| ATLAS, Chenab, Elqui, Indus, Molonglo, Ravi, Tucana III, Turbio, Turranburra | same (88-100%) | same |

### What this says

1. **Normalization: keep the window's own.** The decoy normalization fails
   on a wide sky too (a third of the copies, 11% of empty sky above 0.5): the
   decoy box counts red disk dwarfs, whose density follows Galactic latitude,
   so standardizing the isochrone channels by it gives them a level that
   changes across the sky. Normalizing by counting noise keeps an absolute
   significance but is 14-17 points worse, most at m−M 15-16.
2. **The band label teaches elongation, and pays for it in specificity.** Its
   models' confident answers on empty sky are mostly elongated (blobs are
   36-45% of them, against 80-100% for the count label) — the blob problem is
   gone from the false alarms — but those answers are forty times more
   frequent, so its false-alarm cut is saturated at every distance.
   Tightening the visibility cut from S/N 2 to 5 changes nothing: the extra
   false alarms come from the label's geometry — it marks every pixel within
   one width, including stretches with no stream star in them — not from
   labelling faint streams.
3. **Overall, the count label stays ahead** at a fixed false-alarm rate (82%
   against 79-80%), and ahead at m−M 16-17; the band label is ahead for wide
   and near streams (Jhelum 88% against 12%, m−M 15 25% against 12%), behind
   for narrow ones (Phoenix, Aliqa Uma, Willka Yaku). Neither solves the
   nearest distances.
4. **Train on a wide sky.** One patch of 600 deg² does not transfer (round
   1); fold 0 does, at the same cost.

### What to try next

- **A label between the two**: the band's shape, narrowed to where the
  stream's stars are (e.g. the band intersected with the count label dilated
  along the track, or a band of half a width), to keep the elongation without
  labelling empty stretches.
- **Longer training** (the 19200-window tier) for the two labels, since the
  quick tier's seed spread is up to 8 points here.
- **Both models together**: count-label models for narrow streams and
  band-label models for wide ones are complementary on these fourteen.

## Reproducing

```bash
python scripts/experiments/real_des/patches.py audit            # the label audit (~5 min)
python scripts/experiments/real_des/patches.py label-figures    # its two figures
# round 2 (the default sky, --train-sky fold0): train on fold 0, score on fold 1
python scripts/experiments/real_des/patches.py train --config band/window --seed 42   # ~15-20 min each
python scripts/experiments/real_des/patches.py evaluate --config band/window          # ~8 min
python scripts/experiments/real_des/patches.py figures
# round 1: add --train-sky A (train on patch A, score on patch B)
```
