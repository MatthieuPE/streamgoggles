# Experiments and tests

This section records the tests run to make design decisions for the detection
model: what was asked, how it was measured, what came out, and what was
decided. Each page is self-contained, and its figures can be regenerated from
the scripts it points to.

## Shared protocol

Every experiment follows the same conventions, so results can be compared
across pages.

**Several trained models per configuration.** A configuration (loss,
hyperparameters, training settings) is trained with at least 4 random seeds.
One training run is a single draw: two seeds of the same configuration can
differ more than two configurations do. Curves show the mean over seeds, and
bars the range between seeds.

**The same evaluation skies for every model.** Models are scored with
{py:func}`~streamgoggles.evaluation.footprint.evaluate_footprint_realizations`
on full-sky injections with a fixed evaluation seed: for each surface
brightness, the same 10 stream realizations (population, placement,
orientation, survey noise). Each sky is tiled, predicted, stitched back to
HEALPix and scored per pixel. Differences between models therefore come from
the models, not from the skies they were shown.

**Two backgrounds.** Each model is scored on the background catalog it was
trained with, and on an independently seeded catalog it never saw. The
second one tests whether the model learned what a stream looks like, or the
particular background it was trained on.

**Metrics independent of the area.** At a probability threshold $t$ (a pixel
is classified as stream when the model's output is above $t$), with counts
summed over the realizations of one surface brightness:

| | classified as stream | classified as background | total |
|---|---|---|---|
| true stream pixels | $S_s$ | $S_b$ | $S_t$ |
| true background pixels | $B_s$ | $B_b$ | $B_t$ |

$$
C = \frac{S_s}{S_t}\ \ \text{(completeness)},\qquad
F = \frac{B_s}{B_t}\ \ \text{(contamination)},\qquad
\frac{C}{F}\ \ \text{(contrast)}.
$$

$C$ is the fraction of the stream found, $F$ the fraction of the background
wrongly flagged, and $C/F$ how many times more likely a stream pixel is to be
flagged than a background pixel: $C/F = 1$ means the stream cannot be told
apart from the background, whatever $C$ is. The purity
$S_s/(S_s + B_s) = 1/\big(1 + (B_t/S_t)/(C/F)\big)$ is not used to compare
models, because it depends on how much background area was scored
($B_t/S_t$), which is a property of the evaluation, not of the model.

**One reference threshold.** Models are compared at the same threshold, 0.5
(the natural cut for a sigmoid output trained on a 0/1 label), with 0.9 as a
cross-check. The threshold is not tuned per model when comparing them: it is a
free parameter of the final product, chosen afterwards depending on whether
completeness or low contamination matters more, and each page shows how much
it moves the results.

**Where things live.**

- Scripts: `scripts/experiments/<experiment>/` (`run.py` trains and scores,
  `figures.py` makes the page's figures and prints its numbers).
- Trained models and raw results: `data/experiments/<experiment>/`
  (git-ignored, so kept locally only).
- Figures: `docs/source/experiments/figures/<experiment>/`.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| {doc}`loss_selection` | Which training loss, batch size and background fraction? | done: batch Dice, batch 8, background fraction 0.05 |
| Threshold tuning | Which probability threshold for the final maps, and does it hold on skies not used to choose it? | planned |
| Network hyperparameters | Width, depth, learning rate, training length and batch size with the selected loss. Batch 8 used 4x fewer optimizer steps than batch 2 in the loss selection, so it may still be undertrained. | planned |
| Wider stream parameter space | Detection as a function of surface brightness and distance modulus (2-D), then width, length, age and metallicity. | planned |
| Generic matched filter | One filter swept over trial distance modulus, with the filter's parameters (age, metallicity, trial distance) given to the network as inputs. | planned |
| Stream populations | Several streams injected in the footprint; per-stream (object-level) metrics next to the per-pixel ones. | planned |
| Survey-wide contamination | False alarms over a fully tiled footprint, rather than around one injected stream. | planned |
| Real data | Train with the known streams masked, then unmask and check that they are recovered. | planned |

```{toctree}
:maxdepth: 1
:hidden:

loss_selection
```
