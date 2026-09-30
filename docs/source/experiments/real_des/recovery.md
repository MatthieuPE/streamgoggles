# Recovery efficiency

## The test

A single real sky gives one yes or no per stream. To tell a stream that is hard
from a place that is hard, each DES 2018 stream was simulated with its own
Table 1 parameters — width, length, distance, surface brightness, and its
fitted age and Z (`objects_overlap.DES2018_POPULATIONS`, all inside the
training range) — and injected into **the real inference maps**:

- at **50 random places** of the calibration sky, at random orientations (at
  least 90% of the band on calibration sky);
- at **4 places parallel to its own track**, 3° and 5° to either side.

Each copy is scored by exactly the test the real streams get: the tiles over
its band re-predicted by the ensemble of the fold that did not train on each
pixel, false-alarm rates against the original maps' calibration pixels of the
same fold, ≥ 20 flagged pixels and S/N ≥ 2 against 200 null bands, at the
queried distance nearest its own. The copy's own S/N in the matched-filter
input (its stars in the band over the square root of the background's) is
recorded too. 756 trials, about 15 minutes (`run.py recovery`, results in
`data/experiments/real_des/recovery.csv`).

```{image} ../figures/real_des/recovery_by_stream.png
:alt: Fraction of simulated copies recovered per stream, overall and per fold, with the real outcome above
:width: 100%
```

*Fraction of the 50 random copies recovered (dark), and split by the fold the
copy fell in (fold 0: ▼, fold 1: ▲); bars are 1σ binomial intervals. Streams
ordered by distance modulus. Above, whether the real stream was detected.*


## What decides recovery

**Three things decide recovery, and the place is not one of them.**

1. **The model's fold.** Copies in fold 0 — predicted by the ensemble trained
   on fold 1 — are recovered **50%** of the time, copies in fold 1 **75%**,
   over the same streams. The two folds' copies sit at similar latitudes (median
   b −57° and −63°) and background densities (7.2 and 8.0 stars per pixel). Within a stream, found and missed copies have the same
   input S/N, latitude and background density; the fold is what separates
   them (for ATLAS, all 8 misses are in fold 0). **The two ensembles did not
   learn the same thing**: six models averaged per fold do not wash out what
   one training sky teaches. **Why is taken apart below**: a few regions of
   fold 0's calibration sky saturate the fold-1 ensemble and switch its
   detections off there; without them both folds recover 83-84%.

2. **Distance and width.** Jhelum (m−M 15.6, 1.16° wide) is recovered in 1
   copy of 50 although its input S/N is 26, and Wambelong (15.9) in 2. Below
   m−M 16 the network does not see these streams whatever the place — the
   near-distance weakness of the simulations, now measured on the real sky.
3. **The stream's strength.** Past that, recovery follows the input. And the
   copies are **stronger than the real streams**:

```{image} ../figures/real_des/recovery_input.png
:alt: Input S/N of the simulated copies against the real streams' input S/N
:width: 100%
```

*The simulated copies' S/N in the matched-filter input (blue: median and
10-90%) against the real stream's (orange, side bands along the paper's
track; Jhelum's is biased negative, see {doc}`results`). The two measure the same
thing — excess stars over the background's noise in the band — by different
means.*

Taking each real stream's band as it lies across the two folds, the
simulations predict **about 10 real detections; 5 are made**. The five found
all have an expected recovery of at least 0.83, and inputs close to their
copies'. The missed streams the network should have caught — Indus (expected
0.80), Turbio (0.84), Aliqa Uma (0.66), Phoenix, Turranburra, Ravi (about 0.6)
— all have real inputs **two to eight times weaker than their simulated
copies** (Indus 12 against 92, Turranburra 4.5 against 18). So:

- the network recovers the simulated DES 2018 streams it is expected to, on
  the real sky, at most places: it has learned something, and the fold
  imbalance is the false-alarm cut, not the learning (below);
- the real streams it misses are, in our data, fainter than Table 1's surface
  brightness makes their copies — the simulation of a stream from its
  published parameters overestimates what DES Y6 with our cuts contains —
  except Jhelum and Wambelong, which are too near.

## The fold asymmetry: saturated false alarms

**Which ensemble, or which sky?** Each copy was predicted by both
ensembles, so each can be scored alone on every band (calibrated on its
own output: its out-of-fold map on the other fold, its in-fold map on its
own):

| | copies in fold 0 | copies in fold 1 |
|---|---|---|
| ensemble trained on fold 0 | 74% (its training sky) | 76% |
| ensemble trained on fold 1 | **54%** | 81% (its training sky) |

The fold-0 ensemble recovers the same fraction everywhere, on the sky it
trained on and on the sky it never saw. **The fold-1 ensemble does well
only on its own training sky**, and loses a third of the streams on the
other — Molonglo 85% → 22%, Ravi 91% → 41%, Turbio 100% → 56%. So the
asymmetry is one ensemble that does not generalize, not a harder half of
the sky. It is not the obvious difference between the skies: fold 0's
training sky is the larger (2,139 against 1,877 deg²) and holds *more*
unmasked real structure (20% of it outside the calibration sky, against
14%).

**All six fold-1 models share it**, so it is what they learned from their
sky, not bad seeds. Each of the twelve models scored alone (its own sky
map, `run.py infer-models`, then `recovery --per-model`):

| | on its own training sky | on the other fold's sky |
|---|---|---|
| fold-0 models (6) | 0.68 ± 0.09 | 0.75 ± 0.05 |
| fold-1 models (6) | 0.76 ± 0.08 | **0.43 ± 0.03** |

**The mechanism is the cut, not the streams.** On fold 0's calibration
sky the fold-1 models answer at full confidence on 885 pixels — more than
the 10⁻³ false-alarm budget allows — so their threshold there sits at an
output of 1.000, and no stream can pass it however well it is seen. On
their own sky the threshold is 0.60-0.80, like the fold-0 models'. The
fold-0 models give the same pixels a median of 0.27: they trained on them
as background. The saturated pixels form a few regions, all in fold 0:

| region | isochrone filter, m−M 17 | faint stars (23-24.5) | red decoy | bright stars | what it is |
|---|---|---|---|---|---|
| Fornax cluster (54.7, −35.4) | 1.77 | 1.23 | 0.98 | 1.03 | galaxy cluster |
| NGC 1407 (55.0, −18.5) | 1.90 | 1.26 | 1.03 | 1.02 | Eridanus group |
| NGC 1332 (51.6, −21.3) | 1.40 | 1.15 | 0.97 | 0.98 | Eridanus group |
| (7.4, −58.7), 9° × 2° | 1.24 | 1.09 | 1.04 | 1.05 | on Tucana III's eastward extension |
| (0.5, −4.2) | 2.06 | 2.10 | 1.96 | 1.99 | survey artefact? |
| (12.7, 2.2) | 1.41 | 1.30 | 1.22 | 1.23 | survey artefact? |

*Density inside each region over a 2.5-4° ring around it, in each
selection.* The galaxy groups add faint point sources in exactly the
filter's colours and nothing in red or bright stars — most likely the
globular-cluster systems of their giant ellipticals, classified as stars:
to a matched filter, a population at the right distance. The Tucana III
excess is stream-like, the last two equal in every selection.

**Without them, the imbalance is gone.** Taking these regions, grown by 1°,
out of the calibration sky (101 deg², 3% of it; `recovery
--without-false-alarms`) — a diagnostic, since they were found from the
model's own output:

| recovery | before | without the regions |
|---|---|---|
| all copies | 63% | **83%** |
| copies in fold 0 / fold 1 | 50% / 75% | 84% / 83% |
| fold-1 ensemble alone, fold 0 | 54% | 87% |

Phoenix 34% → 76%, Aliqa Uma 50% → 86%, Turranburra 44% → 88%, Molonglo
58% → 98%, Jhelum 2% → 22%. **The real streams do not change**: the same
five are detected, more strongly, and the nine missed still have no
flagged pixel at any cut. For them the output itself is low — they are
fainter than their copies — not held back by the threshold.

## The surface brightness of the copies

The paper
derives μ_V from the stream's observed main-sequence stars (Table 1, N*):
through a Dotter isochrone and a Chabrier IMF to a stellar mass and M_V, then
μ_V assuming 68% of the light within ±w over length × 2w. Our injection goes
the other way (`inject_utils.convert_SurfaceBrightness_to_N`), with the same
geometry — 68% within ±w, length × 2 sin w — but **reads the V-band μ_V of
Table 2 as a g-band surface brightness**. For these old, metal-poor
populations g is 0.32 mag fainter than V (0.27-0.36; V from g and r after
Jester et al. 2005), so each copy is injected about 0.35 mag too bright.
Carried forward to the paper's own quantities, our copies hold:

| | ours / paper, median (range) |
|---|---|
| observed main-sequence stars, 16 < g < 23.5 (N*) | 1.54 (1.14-1.79) |
| stellar mass | 1.54 |
| M_V | 0.39 mag brighter (0.31-0.48) |

The band accounts for a factor of about 1.35; the rest, about 1.15, is the
isochrone family and mass function. **This is flagged, not converted**: the
pipeline's surface brightness stays a g-band quantity. Where the DES 2018
values are compared with ours, read them as about 0.35 mag brighter than the
paper's; for training, it only means the range reaches that much fainter than
it seems. So the copies carry about 1.5 times the
real streams' stars — for a faint stream, about 1.5 times the input S/N. That
explains part of the gap to the real streams, not all of it: for Indus,
Turranburra and Molonglo the real input is still three to five times weaker
than a corrected copy's would be.

## The detection limit: the real streams are where the copies say

The copies are stronger than the real streams, so their recovery rate does not
say what to expect of the real ones. Measured instead: recovery against the
S/N a copy actually has in the matched-filter input. Each stream's copies
were made fainter by 0, 0.5, 1, 1.5 and 2 mag/arcsec² of surface brightness,
20 random places each (1,400 copies), and scored by the same test, on the
calibration sky without the saturated regions (the masks now adopted;
`run.py detection-limit`). The real streams are placed on the same axis at
their input S/N with our selection, from the DES 2018 reproduction's profile
fit ({doc}`des2018_reproduction`).

```{image} ../figures/real_des/detection_limit.png
:alt: Recovery of dimmed copies against their input S/N, near and far, with the real streams
:width: 100%
```

*Copies recovered against their input S/N, for streams nearer (light) and
farther (dark) than m−M 16.5. Diamonds: the fourteen real streams at their
input S/N, at the top if the first training detected them, at the bottom if
not.*

| input S/N | copies recovered, m−M ≥ 16.5 | copies recovered, m−M < 16.5 |
|---|---|---|
| 2-6 | 4-15% | 0-6% |
| 6-8 | 23% | 7% |
| 8-13 | 51-52% | 5-20% |
| 13-22 | 88-89% | 27-69% |
| 22-45 | 100% | 42-95% |

- **The network's detection limit** — half the copies found — is an input
  S/N of **about 9 for distant streams and about 20 for near ones**. Near
  streams need twice the signal: the near-distance weakness, in the input's
  own units.
- **The real streams follow it.** The five detected are all distant and at
  or above the limit (Willka Yaku 8.7, Tucana III 11.1, Chenab 11.7, Elqui
  15.7, ATLAS 23.1). Every missed one is below it: Indus, Jhelum and Phoenix
  are near, at 11-13, where copies are found 20% of the time; Aliqa Uma is
  distant but at 7.9 (23%); Turranburra, Wambelong and Turbio are at 4-5;
  Molonglo and Ravi at nothing. Summing each real stream's chance at its
  input S/N gives **4.5 expected detections; 5 are found**.
- **So the network does on real streams what it does on simulated ones.**
  What separates the missed streams from the found ones is their strength in
  the input, and for three of them their distance — not a failure on real
  data. The gap between copies and real streams earlier on this page was the
  copies' brightness (their V-band surface brightness read as g, and more).
- **The bar to clear is sensitivity.** The DES 2018 paper reported its
  streams at significances down to 5-7 with a targeted fit along a known
  track; the network, searching blindly at a false-alarm rate of 10⁻³ per
  pixel, needs about 9, and about 20 for near streams. Lowering that limit —
  most of all at short distance — is what would find more streams.

## Caveats

One realization of the stream's stars per place; 50 places give
rates to about ±7%. The home placements often fall partly on masked sky (22-88%
of their band on calibration sky) and are few; they agree with the random
ones and are not analysed separately. The input S/N of copies and real streams
come from different estimators.

