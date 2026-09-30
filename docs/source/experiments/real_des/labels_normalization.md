# Labels and normalization, on patches of the real sky

The first real-data models found 5 of the 14 DES 2018 streams, answered with
compact blobs, and saw nothing nearer than m−M 16 ({doc}`recovery`). Before
training on the whole sky again, this page goes back to quick models on two
patches of the real DES sky, to find out which part of the data preparation
teaches the model blobs, and whether normalizing by the decoy channel helps.

**Status: done (2026-09-29, branch `des-clean-training`).**

**In short.** Keep the count label and the per-window normalization, and use
four models: four quick count-label models recover 86% of the DES 2018 copies
on sky they never saw; six, longer training or larger windows add nothing,
and none lowers the detection limit for distant streams below an input S/N
of about 9. Keep the per-window normalization: normalizing by the decoy
fails (the decoy's density follows Galactic latitude, the isochrone channels'
does not), and by counting noise is 14-17 points worse. The band label does
what it was designed to — every label one elongated band, and the model's own
false alarms elongated rather than blobs — but it does not beat the count
label overall (79-80% of the DES 2018 copies against 82%): it wins on wide,
near streams (Jhelum 88% against 12%) and loses on narrow ones (Phoenix 38%
against 88%), and it fires on empty sky forty times more often. Train on a
wide sky: one 600 deg² patch does not transfer.

**But along a known track the matched filter alone is twice as sensitive as
the network** (half of the copies found at an input S/N of 5, against 9-12
for the network, whether its output is thresholded per pixel or averaged
along the band): the network loses information, and adds none where the
track is known.

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

The same configurations — plus the Poisson normalization; in a round 2b the
band label with a stricter visibility cut, S/N ≥ 5; and in a round 2c the
**segmented band**: the band cut into 1° segments along the track, each
labelled only where the stream stands out locally (S/N ≥ 1 in the segment,
`band_segment_deg`) — trained on all of
fold 0's training sky and scored on fold 1's calibration sky. Now the
false-alarm cuts of the count-label models are no longer saturated, and the
comparison measures the models.

| configuration | DES 2018 copies (seeds) | m−M 15 / 16 / 17 / 18 / 19 | empty sky > 0.5 | of which blobs | saturated cuts |
|---|---|---|---|---|---|
| **count / window** | **82%** (83, 81) | 12 / **100** / **100** / 100 / 100% | **0.1%** | 100% | 1 of 9 |
| band / window | 79% (71, 79) | **25** / 62 / 88 / 100 / 100% | 4.1% | **41%** | 9 of 9 |
| band5 / window (S/N ≥ 5) | 80% (72, 80) | 25 / 62 / 88 / 100 / 100% | 3.7% | 45% | 9 of 9 |
| segmented band / window | 73% (78, 72) | **50** / 62 / 100 / 100 / 100% | 2.4% | 46% | 7 of 9 |
| count + band / window (4 models) | 86% | 25 / 100 / 100 / 100 / 100% | 1.6% | 60% | 1 of 9 |
| **count / window, 4 models** | **86%** | 38 / **100** / **100** / 100 / 100% | **0.2%** | 100% | 1 of 9 |
| count / window, 128-pixel windows | 80% (79, 77) | 38 / 75 / 100 / 100 / 100% | 0.4% | 66% | 1 of 9 |
| count / poisson | 68% (72, 67) | 0 / 38 / 100 / 100 / 100% | 0.4% | 80% | 2 of 9 |
| band / poisson | 62% (46, 62) | 25 / 38 / 75 / 100 / 100% | 3.5% | 36% | 9 of 9 |
| band / decoy | 31% (0, 24) | 25 / 12 / 62 / 38 / 12% | 11% | 13% | 9 of 9 |

*"Saturated cuts": queried distances at which the output at a false-alarm
rate of 10⁻³ is above 0.98 — where the model's false alarms are as confident
as a stream can be.*

*Each figure: left, the DES 2018 copies recovered on fold 1 (bar: the ensemble of the models, dots: each model); middle, the distance scan (ensemble); right, the share of fold 1's stream-free calibration sky where the ensemble's output is above 0.5 (median over the queried distances). The reference, the first training's configuration, is in dark grey.*

### Normalization

```{image} ../figures/real_des_patches/compare_normalization_fold0.png
:alt: Window, Poisson and decoy normalizations compared
:width: 100%
```

**Normalization: keep the window's own.** The decoy normalization fails
on a wide sky too (a third of the copies, 11% of empty sky above 0.5): the
decoy box counts red disk dwarfs, whose density follows Galactic latitude,
so standardizing the isochrone channels by it gives them a level that
changes across the sky. Normalizing by counting noise keeps an absolute
significance but is 14-17 points worse, most at m−M 15-16.

### Labels

```{image} ../figures/real_des_patches/compare_labels_fold0.png
:alt: Count, band, band with S/N 5, and segmented band labels compared
:width: 100%
```

**The band label teaches elongation, and pays for it in specificity.** Its
models' confident answers on empty sky are mostly elongated (blobs are
36-45% of them, against 80-100% for the count label) — the blob problem is
gone from the false alarms — but those answers are forty times more
frequent, so its false-alarm cut is saturated at every distance.
Tightening the visibility cut from S/N 2 to 5 changes nothing: the extra
false alarms come from the label's geometry — it marks every pixel within
one width, including stretches with no stream star in them — not from
labelling faint streams.

**Overall, the count label stays ahead** at a fixed false-alarm rate (82%
against 79-80%), and ahead at m−M 16-17; the band label is ahead for wide
and near streams (Jhelum 88% against 12%, m−M 15 25% against 12%), behind
for narrow ones (Phoenix, Aliqa Uma, Willka Yaku). Neither solves the
nearest distances.

**Labelling only where the stream stands out locally does not rescue the
band.** The segmented band keeps its labels elongated (1% with a largest
piece under 2°, one piece at the median) and halves the band's false
alarms (2.4% of empty sky), and it is the best at m−M 15 (50%) — but it
is the worst of the three overall (73%), losing the narrow streams
(Tucana III, Willka Yaku, Aliqa Uma), whose short segments rarely reach
S/N 1 on their own.

**Stream by stream** (ensemble), the two labels are complementary:

| | count / window | band5 / window | segmented band / window |
|---|---|---|---|
| Jhelum (1.16° wide, m−M 15.6) | 12% | **88%** | 75% |
| Wambelong (m−M 15.9) | 12% | 25% | 12% |
| Phoenix (0.16° wide) | **88%** | 38% | 50% |
| Aliqa Uma (0.26° wide) | **75%** | 38% | 25% |
| Willka Yaku (0.21° wide) | **100%** | 75% | 50% |
| Tucana III (0.18° wide) | **100%** | 100% | 50% |
| ATLAS, Chenab, Elqui, Indus, Molonglo, Ravi, Turbio, Turranburra | same (88-100%) | same | same |

### Number of models

```{image} ../figures/real_des_patches/compare_models_fold0.png
:alt: Two and four count-label models, and the count and band mix
:width: 100%
```

**Combining the labels adds nothing that more models do not.** Averaging
the two count-label and the two band-label models recovers 86% of the
DES 2018 copies, and gains Jhelum (62%) without losing the narrow
streams — but four count-label models do exactly as well (86%, Jhelum
62%, 38% at m−M 15) with eight times fewer confident answers on empty sky
(0.2% against 1.6%). The gain was the number of models, as the simulated
experiments had found for the quick tier ({doc}`../stream_parameters`).

### Window size

```{image} ../figures/real_des_patches/compare_window_size_fold0.png
:alt: 96-pixel and 128-pixel windows compared
:width: 100%
```

**Larger windows do not help, at this training length.** 128 × 128
pixels (14.7°, `image_pix`) instead of 96 (11°), the same 4,800 windows:
80% of the DES 2018 copies against 82%, a tie within the seeds' spread.
Better on the nearest streams (38% against 12% at m−M 15, Wambelong 50%
against 12%), worse at m−M 16 (75% against 100%), on Turranburra (62%
against 100%) and Aliqa Uma (50% against 75%) — per-stream differences
below about 15 points are within the noise of two seeds and eight
placements. A larger window also means fewer windows' worth of sky per
pixel for the same training length; it may need a longer training to pay.

### Sensitivity: more models, longer training

The detection limit ({doc}`recovery`) is the bar to clear, so the scoring
also measures it: a **fainter** set, the DES 2018 copies made 1 and 1.5
mag/arcsec² fainter (near the real streams' strength), and from it and the
full-brightness copies, the input S/N at which half the copies are found, for
near and distant streams (`sensitivity`). Two levers, with the count label and
the per-window normalization: more quick models (4 and 6 instead of 2), and
two models trained four times longer (19,200 windows).

```{image} ../figures/real_des_patches/compare_sensitivity_fold0.png
:alt: Recovery of DES 2018 copies against their input S/N, per configuration, near and distant streams
:width: 100%
```

```{image} ../figures/real_des_patches/compare_models_and_length_fold0.png
:alt: Two, four and six quick models, and two long ones
:width: 100%
```

| | DES 2018 copies | m−M 15 | half-recovery input S/N, near | far |
|---|---|---|---|---|
| 2 quick models (reference) | 82% | 12% | 14.6 | 8.9 |
| **4 quick models** | **86%** | 38% | **12.1** | 8.9 |
| 6 quick models | 86% | 38% | 12.3 | 8.9 |
| 2 long models (19,200 windows) | 80% | **62%** | 13.9 | 11.3 |
| 128-pixel windows (2 quick models) | 80% | 38% | 15.2 | 12.7 |

**Averaging helps up to four models, then stops**: four lower the limit for
near streams from 14.6 to 12.1 and recover 86% of the copies; six add
nothing. **Training four times longer does not help**: better on the bright
m−M 15 scan (62%), worse on faint distant streams (limit 11.3). And **for
distant streams no lever moves the limit below an input S/N of about 9**: the
curves of two, four and six models lie on one another. The limit is not the
models' spread; it is what this network, scored this way, can do.

### Does the network add sensitivity? Three tests on the same copies

The detection test asks for at least 20 pixels above the false-alarm cut
within one width of the track — a test of the *map*, blind to the track. To
see whether the network's limit is the network's or the test's, the same
copies (four quick models, DES 2018 set and the fainter set) are scored three
ways, all along the known track:

- **network, per pixel**: the test above (at least 20 pixels at a false-alarm
  rate of 10⁻³).
- **network, integrated**: the mean of the network's output over the band
  (± one width along the track) against the same band shape moved to 200
  random places of the calibration sky (`null_band_placements`,
  `band_mean_statistics`); found when the band beats all 200.
- **matched filter alone**: the same integrated test on the network's
  *input* — the isochrone channel's counts at the stream's distance, minus a
  smooth local background (the mean over 1.8° pixels, nside 32, interpolated).
  No network at all.

The matched-filter test is honest: on 150 stream-free bands of the fold-1
calibration sky, for a narrow distant track (0.25°, 10°, m−M 17) and a wide
near one (0.8°, 15°, m−M 16), none is found, and the S/N of empty bands
scatters by 0.9 — the null bands measure the noise correctly.

```{image} ../figures/real_des_patches/detection_tests_fold0.png
:alt: Fraction of copies found against their input S/N, by the network per pixel, the network integrated along the band, and the matched filter alone
:width: 100%
```

| four quick models | half-recovery input S/N, near | far | copies found, near | far |
|---|---|---|---|---|
| network, per pixel (current test) | 12.1 | 8.9 | 41% | 50% |
| network, integrated along the band | 10.4 | 8.0 | 48% | 55% |
| **matched filter alone, integrated** | **5.2** | **5.0** | **72%** | **78%** |

With two models the matched filter's limit is the same (5.2 / 5.0, it does
not depend on the models) and the network's is 14.6 / 8.9 per pixel, 15.3 /
8.2 integrated. At full brightness, per stream: the matched filter finds
every stream at every place except Wambelong (88%); the network finds
Jhelum at 62% and Wambelong at 12%.

**Along a known track, the matched filter alone is about twice as sensitive
as the network** (half-recovery S/N 5 against 9-12). Integrating the
network's output along the band gains little (10.4 against 12.1 near): the
information is lost *inside* the network, not by the per-pixel test. The
network does not add sensitivity where the track is known; its use is to
find tracks that are *not* known — where the matched-filter test cannot be
run, since it needs the track. The fair comparison for discovery is
therefore the matched filter searched over all tracks, with the look-elsewhere
cost that brings; that has not been measured.

### The sky

**Train on a wide sky.** One patch of 600 deg² does not transfer (round
1); fold 0 does, at the same cost.

### What to try next

- ~~A label between the two~~: tried as the segmented band (round 2c); it
  keeps the elongation and fewer false alarms, but loses the narrow streams.
- **Longer training** (the 19200-window tier) for the two labels, since the
  quick tier's seed spread is up to 8 points here.
- ~~Both models together~~: tried (round 2d); four count-label models match
  it with fewer false alarms.
- ~~More count-label models, or longer ones~~: tried; four quick models are
  the best (86%, near limit 12.1); six and longer add nothing.
- **Why the network loses half the input's sensitivity**: the per-window
  normalization (a faint stream in a window with a bright feature is
  compressed), the count label (near, faint streams get speckled labels), or
  the loss. The matched-filter integrated test is the ceiling to aim for.
- **A matched-filter search over tracks** (great circles through each
  window, the same integrated test), with its look-elsewhere cost: the
  baseline the network has to beat for discovery.

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
