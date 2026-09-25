# Training on the real DES background

Every model so far was trained on a simulated sky. This experiment trains the
adopted configuration on the **real DES Y6 sky**, with every known stream,
globular cluster and dwarf galaxy masked out of it, then runs it over the whole
DES footprint with the known streams **put back**, to see whether it finds them.

**Status: set up and tested end to end; not run yet.** This page records the
design. Results go below it once the trainings have run.

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

There is only one real sky. A model predicting pixels it trained on has
already been taught what their background looks like, which flatters its
false-alarm rate, and that rate is what a detection threshold is set from. So
the footprint is split in two, and each half is predicted only by models that
never saw it:

- **The split**: 20° stripes of right ascension, alternating between the
  folds (`objects_overlap.spatial_fold`). Stripes rather than one cut give
  both folds the whole range of galactic latitude DES spans, so neither trains
  on an easier sky. At Dec −50° a stripe is about 13° across, wider than an
  11° window. Fold 0 holds 163,107 pixels (2,140 deg²) of the training sky,
  fold 1 holds 143,139 (1,880 deg²).
- **By pixel, not by star**: each star goes with the HEALPix pixel it falls
  in, and the pixel with its centre. A first version split the stars by their
  own positions. That left 1,177 pixels on the stripe edges partly in each
  fold, so a pixel could be predicted by a model that had trained on part of
  it. A test now holds stars on both sides of an edge in one pixel and checks
  they land in the same fold.
- **Training**: six models per fold, each seeing only its fold's stars.
  Injected streams are placed on the fold's own pixels, and any part of a
  stream crossing into the other fold falls on empty pixels, zeroed in both
  input and label, exactly like a hole in the footprint.
- **Inference**: both ensembles run on every tile of the inference sky, and
  each pixel keeps the output of the ensemble trained on the *other* fold.

The known streams are masked in both folds' training, so detecting them is
never a model recognising something it was taught.

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

## What comes after the maps

- **A threshold**: the false-alarm rate measured on the inference sky away from
  the known streams, which the fold split makes an honest one.
- **The DES 2018 streams**: each queried near its own distance and judged along
  its DES track, with the same criterion as the simulated streams: enough
  flagged pixels near the track, and a signal-to-noise against stream-shaped
  bands of the stream-free sky.
- An animation of the maps through the distances.

## Reproducing

From the repository root, in the `streamml` environment, with both skies built
(`python scripts/real_data/background.py --write`):

```bash
python scripts/experiments/real_des/run.py train --fold 0    # six models, ~1.5 h
python scripts/experiments/real_des/run.py train --fold 1
python scripts/experiments/real_des/run.py infer             # nine maps, ~5 min
```

Each finished model is saved as it completes, so a run resumes where it
stopped.
