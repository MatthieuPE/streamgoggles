# Real DES Y6 data

Every model before this section was trained on a simulated sky. Here the
configuration the simulated experiments adopted is trained on the **real DES
Y6 sky**, with every known stream, globular cluster and dwarf galaxy masked
out of it, and then run over the whole DES footprint with the known streams
**put back**, to see whether it finds the fourteen streams of DES 2018
(Shipp et al. 2018). The work runs over several rounds; each has its page.

| page | what it holds | status |
|---|---|---|
| {doc}`first_training` | two spatial folds, the configuration, inference | done (2026-09-25) |
| {doc}`results` | the output maps; the DES 2018 test: 5 of 14 | done, re-scored along the paper's tracks (2026-09-29) |
| {doc}`des2018_reproduction` | the paper's own analysis on our data, without the network: which streams are in the data, and what our cuts cost | done (2026-09-29) |
| {doc}`recovery` | each DES 2018 stream simulated and injected into the real sky 54 times; the fold imbalance traced to saturated false alarms | done (2026-09-29) |
| {doc}`labels_normalization` | quick models: the training label (count or band) and the normalization (window, decoy, Poisson); galaxy masks; patches of the sky | done (2026-09-29, branch `des-clean-training`) |
| {doc}`window_level` | a network that answers with lines (a Hough transform built in), scored without the track; the matched filter's own line search alongside | done, quick models (2026-10-01, branch `hough-window-model`) |
| {doc}`line_sky` | the line model over the whole DES sky with the real streams in it: on-sky maps of the detected lines against the DES 2018 tracks | done (2026-10-01, branch `line-model-sky`) |
| {doc}`line_followup` | more models and longer training of the 2° line model; the catalogue's leads looked at one by one (maps, distance profiles, Hess differences) | in progress (2026-10-01, branch `line-model-followup`) |

How the real catalogue and its masks are built is in
{doc}`../../narrative/real_des_background`; the code is
`scripts/experiments/real_des/run.py`, one step per command (see Reproducing).

## The three skies

The same DES Y6 Gold stars with the same cuts (`0 <= EXT_XGB <= 1`, S/N > 5,
16 ≤ g, r ≤ 24.5) are used three ways, nested one inside the other. They
differ only in what is masked, and each answers a different question.

| | **training sky** | **calibration sky** | **inference sky** |
|---|---|---|---|
| what it is for | the background the models learn from: streams are injected into it | the stream-free reference: how often the network fires with no stream | the sky the models are run on, to find streams |
| used by | `train` | `detect`, `recovery`: false-alarm rates and null bands | `infer`, `detect`, `recovery` |
| the DES 2018 streams | masked, 3 widths along their tracks | masked | **left in** |
| Sagittarius | masked to 6° | masked to **9°** | masked to 6° |
| LMC, SMC outskirts | left in | masked to **20°, 12°** | left in |
| dwarfs, globular clusters | masked to 5 half-light radii | dwarfs to **12**, clusters to **2°** (also those hidden in Gold's holes) | masked to 5 half-light radii |
| area | 4,017 deg² | 3,331 deg² | 4,690 deg² |
| defined by | `get_footprint("des_yr6_background")` | `calibration_mask_nside512.fits.gz` (`run.py calibration`) | `get_footprint("des_yr6_inference")` |

- **Training ⊂ inference**: the inference sky is the training sky plus the
  known streams (673 deg²). Training on a sky that held them would teach the
  model to call them background; inference on a sky without them could not
  find them.
- **Calibration ⊂ training**: the calibration sky is the training sky minus
  the real structure the first maps showed the training masks had left —
  the Magellanic outskirts, Sagittarius beyond 6°, the rings of Sculptor and
  Fornax, clusters in Gold's holes (686 deg²; {doc}`results`, "What the
  stream-free sky still holds"). Measuring false alarms there counts only
  what the network does on sky that truly holds no stream. **The models
  still trained on those 686 deg²**, taught as background.
- **Each is split into two folds** of 20° stripes of right ascension
  ({doc}`first_training`): each ensemble trains on one fold's training sky,
  predicts the other fold's inference sky, and is calibrated on that fold's
  calibration sky.

```{image} ../figures/real_des/skies.png
:alt: The inference, training and calibration skies over the DES footprint, nested
:width: 100%
```

*The three skies. Dark: calibration; light blue: training but not
calibration; grey: inference only (the known streams). White: masked in all
three (Sagittarius, clusters, dwarfs, Gold's holes). Orange: the regions of
saturated false alarms found in {doc}`recovery` — the Fornax cluster, the
Eridanus group, an excess on Tucana III's extension, two artefacts —
removed from the calibration sky only as a diagnostic. `run.py folds`.*

## Conclusions so far

1. **5 of the 14 DES 2018 streams are detected** by models that never saw
   them — Elqui, ATLAS, Willka Yaku, Tucana III, Chenab — each peaking at its
   own distance ({doc}`results`).
2. **The folds mattered**: scored on its own training sky, a model's
   false-alarm rate is nearly three times too low at m−M 15-15.5.
3. **Twelve of the fourteen streams are in the data**, with the paper's
   selection or ours; only Ravi and Molonglo are not. Our cuts and filter
   cost little, except for Turranburra ({doc}`des2018_reproduction`).
4. **Simulated copies of the DES 2018 streams, injected into the real sky,
   are recovered at 63% of places** — 83% once a few regions of saturated
   false alarms are left out of the calibration sky: the Fornax galaxy
   cluster and the Eridanus group, whose globular-cluster systems pass the
   matched filter, and two artefacts. They alone made the fold imbalance
   (50% against 75%). Below m−M 16 almost nothing is recovered anywhere
   ({doc}`recovery`).
5. **The missed real streams that are in the data are fainter than their
   simulated copies** — partly because the copies read Table 2's V-band
   surface brightness as g-band (about 0.35 mag too bright, 1.5 times the
   stars; flagged, not converted), and for Indus, Turranburra and Molonglo by
   more than that.

6. **The training label taught blobs**: a third of the first training's
   labels, and more than half for streams nearer than m−M 17, were scattered
   clumps. A band label removes them and makes the model's false alarms
   elongated, but it is not better overall; the per-window normalization is
   the best of three ({doc}`labels_normalization`).
7. **The network finds the real streams it can**: its 50% detection limit
   is an input S/N of about 9 for distant streams and 20 for near ones, and
   at their real input S/N the copies predict 4.5 of the 14 real streams to
   be found; 5 are ({doc}`recovery`, "The detection limit"). Every missed
   stream is below that limit.
8. **Along a known track the network is half as sensitive as its input**:
   the matched-filter counts along the band, against the same band placed
   elsewhere, find half of the copies at an input S/N of 5; the network
   needs 9-12, per pixel or averaged along the band. Four quick count-label
   models with the per-window normalization are the best network
   ({doc}`labels_normalization`).
9. **No change of the network closes that gap**: a depth-4 U-Net, a
   cross-entropy loss, the band label, training on streams down to 36
   mag/arcsec², and summing logits rather than probabilities along the band
   all leave the limit at 9-15. The network acts as a local detector; the
   integration along the stream is what it lacks
   ({doc}`labels_normalization`, "Why is the network half as sensitive?").
10. **A network that answers with lines finds near, faint streams best** —
    once its inputs are matched-filter S/N maps. Without knowing the track
    it finds half of the near copies at an input S/N of 7.9 (the per-pixel
    network needs 12.1, and its test knows the track; the matched filter's
    own line search 12.5), and Wambelong, which no other method finds. It
    misses the two brightest short streams, and for distant ones the
    matched filter's line search stays ahead (10.0 against 12.8)
    ({doc}`window_level`).
11. **On the real sky, two line searches together find eight of the
    fourteen DES 2018 streams beyond chance**, without knowing any track, at
    the false-alarm rate of one: the line network seven (ATLAS, Phoenix,
    Indus, Jhelum, Chenab, Elqui — the near, wide ones the per-pixel network
    missed — and Turranburra by one line), the matched filter's line search
    six once the bright dwarfs and clusters are masked (with Tucana III and
    Willka Yaku, the short, bright ones the line network misses), against
    five for the per-pixel network. The line network missed short streams
    because its label (4° minimum) taught it to; with a 2° minimum it finds
    them, and Tucana III on the sky. Joined into tracks, the lines also
    follow the Magellanic Clouds' outskirts; of the tracks found elsewhere,
    one runs along Leiptr (Ibata et al. 2021) and its Hess difference shows
    a main sequence — the others are edges and depth changes
    ({doc}`line_sky`, {doc}`line_followup`).

## What comes next

- **Mask known galaxy clusters and groups** in training and calibration,
  from a catalogue rather than from the model's output.
- **Train only where the background is contiguous enough**: a stream placed
  across masked sky is cut into fragments, and a model trained on fragments
  learns blobs rather than elongated structures. Restrict training windows
  (and calibration) to sky where enough of a window, and of an injected
  stream, is unmasked.
- **The normalization per window**, which lets each window's background set
  its own scale: normalize by the decoy box instead, keeping the count level.
- **Nearby streams**: nothing is recovered below m−M 16 — weighting nearby
  streams in training, or the longer (19200-window) tier.
- **Measure the input with the profile fit** of {doc}`des2018_reproduction`
  rather than side bands, so the input-against-network figure is right for
  Jhelum.

## Reproducing

From the repository root, in the `streamml` environment, with the skies built
(`python scripts/real_data/background.py --write`):

```bash
python scripts/experiments/real_des/run.py train --fold 0    # six models, ~1.5 h
python scripts/experiments/real_des/run.py train --fold 1
python scripts/experiments/real_des/run.py infer             # nine maps, ~5 min
python scripts/experiments/real_des/run.py figures           # a map per distance, the GIF
python scripts/experiments/real_des/run.py calibration       # the calibration sky
python scripts/experiments/real_des/run.py detect            # the DES 2018 test, its figures
python scripts/experiments/real_des/run.py recovery          # recovery efficiency (~15 min), its figures
python scripts/experiments/real_des/run.py infer-models      # one sky map per model (~2 min)
python scripts/experiments/real_des/run.py recovery --per-model             # each model alone
python scripts/experiments/real_des/run.py recovery --without-false-alarms  # the diagnostic
python scripts/experiments/real_des/run.py folds             # the skies and folds figures
```

The DES 2018 reproduction is `notebooks/des2018_reproduction.ipynb`. Each
finished model is saved as it completes, so a run resumes where it stopped.

```{toctree}
:maxdepth: 1
:hidden:

first_training
results
des2018_reproduction
recovery
labels_normalization
window_level
line_sky
line_followup
```
