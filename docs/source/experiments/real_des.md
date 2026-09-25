# Training on the real DES background

Every model so far was trained on a simulated sky. This experiment trains the
adopted configuration on the **real DES Y6 sky**, with every known stream,
globular cluster and dwarf galaxy masked out of it, then runs it over the whole
DES footprint with the known streams **put back**, to see whether it finds them.

**Status: running** (started 2026-09-25). This page records the design;
results go below it once the trainings have run.

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
