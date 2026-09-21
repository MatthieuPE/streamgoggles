# Loss selection

**Question.** Which loss, batch size and background fraction should the
detection U-Net be trained with, given that what matters is finding stream
pixels while flagging as little background as possible?

**Decision.** Batch Dice, batch size 8, background fraction 0.05, with the
probability threshold left to tune for the final maps.

## Setup

Nine configurations, 4 seeds each (36 trained models):

| Loss | Background fraction | Batch size |
|---|---|---|
| Dice | 0.05, 0.3 | 2 |
| Dice | 0.3 | 8 |
| Dice+BCE | 0.05, 0.3 | 2 |
| batch Dice | 0.05, 0.3 | 2 |
| batch Dice | 0.05, 0.3 | 8 |

Everything else is `train_model.ipynb`'s setup: U-Net with base width 12 and
depth 2, sigmoid output, 1200 training windows (40 epochs of 30), learning
rate 2e-3, detection label with `count_threshold=1`, training surface
brightness 31-34 mag arcsec⁻², distance modulus 16, one isochrone filter plus
one colour-shifted decoy. The number of training windows is the same for
every configuration, so batch 8 takes 4 times fewer optimizer steps than
batch 2.

- **Dice** computes one overlap ratio per window and averages them.
- **Dice+BCE** adds a per-pixel binary cross-entropy (weight 1:1, only tested
  here, not a library loss).
- **Batch Dice** pools the sums over the whole batch before taking the ratio,
  so a false alarm in a window without a stream is penalized like one next to
  a stream. Per-window Dice gives such a window essentially no gradient.

Exact formulas and the gradient argument are in {doc}`../narrative/datasets_and_models`
("Formulas").

**Evaluation**, following the {doc}`shared protocol <index>`: surface
brightness 31 to 35, 10 injected streams per surface brightness (50 in total
per model), on the training background and an independent one; completeness $C = S_s/S_t$, contamination $F = B_s/B_t$,
contrast $C/F$; models compared at threshold 0.5, cross-checked at 0.9.
Curves show $C$ and $F$ averaged over the 4 seeds and the contrast as the
ratio of those averages; bars are the per-seed minimum and maximum. Hollow
markers flag points where the median seed finds fewer than 20 stream pixels:
too few counts to measure a completeness or a contrast. The shaded band is the
training range.

## 1. Comparison at a fixed threshold

```{image} figures/loss_selection/fixed_threshold_comparison.png
:alt: Completeness, contamination and contrast against surface brightness for the nine configurations, at thresholds 0.5 and 0.9
:width: 100%
```

At threshold 0.5, independent background:

| Configuration | $C$ at SB 32 | $C$ at SB 33 | $F$ at SB 33 | $C/F$ at SB 32 | $C/F$ at SB 33 |
|---|---|---|---|---|---|
| Dice, 0.05, batch 2 | 0.91 | 0.50 | 1.6e-2 | 62 | 31 |
| Dice, 0.3, batch 2 | 0.93 | 0.58 | 3.3e-2 | 30 | 18 |
| Dice, 0.3, batch 8 | 0.76 | 0.38 | 8.6e-3 | 103 | 44 |
| Dice+BCE, 0.05, batch 2 | 0.68 | 0.22 | 2.1e-3 | 169 | 109 |
| Dice+BCE, 0.3, batch 2 | 0.66 | 0.20 | 2.0e-3 | 172 | 97 |
| batch Dice, 0.05, batch 2 | 0.87 | 0.13 | 7.3e-4 | 330 | 179 |
| batch Dice, 0.3, batch 2 | 0.88 | 0.15 | 1.0e-3 | 212 | 154 |
| batch Dice, 0.3, batch 8 | 0.87 | 0.20 | 1.4e-3 | 270 | 140 |
| **batch Dice, 0.05, batch 8** | **0.84** | **0.14** | **8.1e-4** | **405** | **166** |

- **Batch Dice has the highest contrast** at every surface brightness where
  the stream is still detected: at SB 32, 2 to 13 times that of Dice and 1.2
  to 2.4 times that of Dice+BCE. The ranking is the same at 0.9.
- **Dice flags the most background.** It finds more of the faint stream at
  0.5 (up to half of it at SB 33), but flags 0.9 to 3% of the background, 6 to
  45 times more than batch Dice. Its false alarms are also over-confident: some
  background pixels get a probability so close to 1 that no threshold removes
  them.
- **Dice+BCE is unreliable.** One of its 4 seeds flags no pixel at all at
  0.5, for both background fractions: its probabilities all sit below the
  threshold. That seed drags its mean completeness down (the bars reaching 0).
- **The background fraction has no consistent effect** for any loss.
- **Every loss fails at the same depth.** At SB 34, batch Dice and Dice+BCE
  find at most 2% of the stream at 0.5, and Dice at most 18% while flagging
  0.7 to 3% of the background; the contrasts there rest on too few stream
  pixels to be measured (hollow points). The loss decides how cleanly a
  stream is found, not how faint a stream can be found.

## 2. Effect of the threshold

The same model gives different maps depending on the threshold. Each panel
below is one configuration; each line is one threshold, from 0.1 (flag
generously) to 0.999 (flag only near-certain pixels).

```{image} figures/loss_selection/threshold_sensitivity_completeness.png
:alt: Completeness against surface brightness, one panel per configuration, one line per threshold
:width: 100%
```

```{image} figures/loss_selection/threshold_sensitivity_contrast.png
:alt: Contrast against surface brightness, one panel per configuration, one line per threshold
:width: 100%
```

Moving the threshold from 0.1 to 0.999, independent background:

| Configuration | $C$ at SB 32 | $C/F$ at SB 32 | $C$ at SB 33 |
|---|---|---|---|
| Dice, 0.05, batch 2 | 0.95 → 0.71 | 41 → 255 | 0.58 → 0.22 |
| Dice+BCE, 0.3, batch 2 | 0.96 → 0.22 | 74 → 6850 | 0.43 → 0.007 |
| batch Dice, 0.05, batch 2 | 0.92 → 0.62 | 191 → 1650 | 0.18 → 0.03 |
| batch Dice, 0.3, batch 8 | 0.95 → 0.33 | 106 → 4300 | 0.34 → 0.01 |
| **batch Dice, 0.05, batch 8** | **0.94 → 0.20** | **149 → 14200** | **0.26 → 0.004** |

- **Raising the threshold always trades completeness for contrast**, in every
  configuration: fewer stream pixels are found, but the ones found are much
  more reliable.
- **Configurations differ in how far that trade goes.** Batch Dice at batch 2
  outputs probabilities mostly close to 0 or 1, so the threshold has a
  limited effect (completeness at SB 32 only drops from 0.92 to 0.62). Dice
  varies moderately. Batch Dice at batch 8 and Dice+BCE span the
  widest range: at SB 32, from finding 94% of the stream with a contrast of
  150, to finding 20% with a contrast above 10,000.
- **This range is useful.** A model whose output responds to the threshold
  lets the final maps be tuned after training: a generous threshold for a
  complete catalogue of candidates, a strict one for a clean map of the most
  secure pixels. A model whose output is nearly binary fixes that choice at
  training time.

## 3. Effect of the background

```{image} figures/loss_selection/background_impact.png
:alt: Completeness, contamination and contrast on the training background and on an independent background, for batch Dice and Dice
:width: 100%
```

Two models, each scored on the background catalog it was trained with (solid)
and on an independently seeded catalog (dashed), at threshold 0.5:

| Model | Background | $C$ at SB 32 | $C$ at SB 33 | $F$ at SB 32 | $C/F$ at SB 32 |
|---|---|---|---|---|---|
| batch Dice, 0.05, batch 8 | training | 0.85 | 0.17 | 2.1e-3 | 409 |
| batch Dice, 0.05, batch 8 | independent | 0.84 | 0.14 | 2.1e-3 | 405 |
| Dice, 0.05, batch 2 | training | 0.91 | 0.56 | 1.5e-2 | 63 |
| Dice, 0.05, batch 2 | independent | 0.91 | 0.50 | 1.5e-2 | 62 |

- **Contamination and contrast are the same on a background the model never
  saw.** The false alarms are not memorized features of the training
  catalogue: the model reacts to background fluctuations in general.
- **Completeness drops slightly at the detection edge** (SB 33: 0.17 → 0.14
  and 0.56 → 0.50). At SB 31-32 it is unchanged.

## 4. Selected model

```{image} figures/loss_selection/selected_model.png
:alt: Completeness, contamination and contrast of batch Dice, batch 8, background fraction 0.05, at threshold 0.5, with each seed shown
:width: 100%
```

Batch Dice, batch 8, background fraction 0.05, threshold 0.5, independent
background (mean over 4 seeds):

| Surface brightness | $C$ | $F$ | $C/F$ |
|---|---|---|---|
| 31 | 0.92 | 1.8e-3 | 524 |
| 32 | 0.84 | 2.1e-3 | 405 |
| 33 | 0.14 | 8.1e-4 | 166 |
| 34 | < 0.01 | 2.3e-4 | not measurable |

**Why this configuration:**

1. **Batch Dice over Dice**: at the same threshold, 11 to 40 times less
   background flagged at SB 33 and 4 to 13 times the contrast at SB 32. Dice
   finds more of the faintest streams at 0.5, but with contamination that no
   threshold can fully remove.
2. **Batch Dice over Dice+BCE**: higher contrast at 0.5 (405 against 170 at
   SB 32, 166 against about 100 at SB 33), higher completeness at bright
   surface brightness (0.84 against 0.68 at SB 32), and no seed collapses. Dice+BCE lost 1 seed in 4 at both background fractions.
3. **Batch 8 over batch 2**: the widest usable range of thresholds, so the
   balance between completeness and contamination can be set when making the
   final maps; the highest contrast at SB 32 at threshold 0.5; and one of
   the smallest spreads between seeds.
4. **Background fraction 0.05 over 0.3**: no measurable difference between the
   two (contrast at SB 33: 166 against 140 at batch 8), and 0.05 keeps
   training focused on windows that contain a stream. The intended use is to
   train on real data with the known streams masked, then unmask them and
   check that they are recovered.

**The threshold is still open.** At 0.5, this model finds only 14% of a
SB 33 stream. Lowering the threshold to 0.1 raises that to 26%, with the
contrast at SB 32 going from 405 to 149. Where to set it depends on what the
maps are used for; choosing it, and checking it on skies not used for the
choice, is the next experiment (see {doc}`index`).

## Conclusion: what this experiment fixes

**Selected, and used by every later experiment and notebook:**

| Setting | Value | Why |
|---|---|---|
| **loss** | **batch Dice** | the only loss that keeps the background clean; per-window Dice cannot learn to suppress false alarms, because an empty window's Dice is ~1 whatever it predicts |
| **batch size** | **8** | the widest usable range of thresholds and the highest contrast at SB 32, with one of the smallest spreads between seeds |
| **background fraction** | **0.05** | indistinguishable from 0.3, and it keeps training on windows that contain a stream |

These three are settled; nothing measured since has argued against them.

**What this experiment did *not* settle**, and what later work changed:

- The **threshold** is still a free parameter. This page compares at 0.5 and
  0.9, and {doc}`hyperparameters` section 6 shows that comparing two models at
  a fixed threshold can point the wrong way — models should be compared at a
  matched false-alarm rate, and the threshold chosen last from a false-alarm
  budget.
- The **training surface brightness range and training length** used here
  (SB 31-34, 1200 windows) were the starting point, not a result.
  {doc}`hyperparameters` replaces them with SB 32-34.5 and 4800 windows.
- The faint-end numbers above are **one draw from a wide seed-to-seed
  distribution** for these short trainings. {doc}`hyperparameters` finds that
  training for longer (19200 windows) both raises them and narrows that
  spread, which does more than averaging several short trainings.

So the configuration to carry forward is this page's loss, batch size and
background fraction, with the training range and length from
{doc}`hyperparameters`.

## Limitations

- **Contrast measured on one stream per sky**, in the neighbourhood tiled
  around it (about 24,000 pixels per realization at nside 512), not over the
  whole survey footprint.
- **At SB 34 and fainter the numbers are counting noise**: a few flagged
  stream pixels out of several hundred. Those points are drawn hollow.
- **Both backgrounds share the same stream realizations**, so the background
  test isolates the effect of the background, not of the streams.
- **Batch 8 took 4 times fewer optimizer steps** than batch 2 for the same
  number of training windows. It may be undertrained; the training length is
  part of the planned hyperparameter experiment.
- **Dice+BCE was tested with a single weighting (1:1).** A smaller BCE weight
  or a Dice warm-up might avoid its collapses.
- **Training windows near RA = 0 were not all checked properly.** Found
  later, during the hyperparameter experiment: the "at least 5° of stream in
  the window" check broke for streams crossing RA 0°/360° (about 13% of
  training streams in this study region, which is centred on RA 0). Those
  windows could contain less of the stream. Every configuration was trained
  the same way, so the comparison is fair, but absolute numbers may shift a
  little with the fix.
- **One set of stream parameters.** Distance modulus, width, length, age and
  metallicity were fixed; the conclusions need checking when they vary.

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/experiments/loss_selection/run.py      # trains and scores the 36 models (~2 min each)
python scripts/experiments/loss_selection/figures.py  # figures above, and their numbers
```

`run.py` resumes where it stopped and re-scores saved models without
retraining. Models and results are written to
`data/experiments/loss_selection/`.
