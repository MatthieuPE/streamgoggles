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
brightness, the same stream realizations (population, placement, orientation,
survey noise). Each sky is tiled, predicted, stitched back to HEALPix and
scored per pixel. Differences between models therefore come from the models,
not from the skies they were shown.

**Realizations are counted per grid point, and they are an evaluation
quantity.** `n_realizations` is the number of independent injections used to
*score* a trained model, *per parameter set* — per surface brightness, or per
cell when a grid varies two parameters — not a total, and never a description
of what the model was trained on (that is counted in training windows). Raising
it tightens the error bar on a model's measured performance; it cannot change
the model. An experiment scoring 20
realizations over 6 surface brightnesses injects 120 streams, with 20 behind
each plotted point, and that 20 is what the point's error bar is computed from.
The realization seed is (base seed, parameter-set index, realization index), so
adding realizations extends a run rather than redrawing it: scoring 100 keeps
the first 20 identical, which makes the two directly comparable.

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

**Is the stream detected? Signal-to-noise along the track.** Per-pixel
completeness can be low while a stream is still plainly visible: a search
finds a stream as a *line* of flagged pixels denser than the background's
false alarms, not by finding every pixel. So each injected stream is also
scored as a whole, in its own frame ($\phi_1$ along the track, $\phi_2$
across it, $\sigma$ the stream's Gaussian width):

- **band**: the pixels within $1\sigma$ of the track
  ($|\phi_2| < \sigma$, $|\phi_1| \le L/2$), $N_\text{band}$ of them, of which
  $n_\text{band}$ are flagged;
- **side bands**: the pixels between $1\sigma$ and $2\sigma$ on both sides.
  They are reported, but they are not background: about 27% of a Gaussian
  stream's stars lie there, and the prediction spreads slightly beyond the
  track;
- **background bands**: the same band shape placed at 200 random positions and
  orientations on the same sky with the stream removed. Their flagged
  densities have mean $\langle\rho_\text{bg}\rangle$ and scatter
  $\sigma_\text{bg}$. Using the real band shape on the real prediction map keeps
  the fact that false alarms come in clumps, which a Poisson formula would
  ignore.

$$
\mathrm{SNR} = \frac{n_\text{band}/N_\text{band} - \langle\rho_\text{bg}\rangle}
                   {\max\!\big(\sigma_\text{bg},\ \sqrt{\max(\langle\rho_\text{bg}\rangle N_\text{band},\,1)}\,/\,N_\text{band}\big)}
$$

Densities are compared, so the SNR is normalized by area. The noise is never
taken below the Poisson noise of the expected background count, or one pixel
when the background is clean, so a clean background cannot give an infinite
SNR. A stream is **detected** at a threshold if

$$
n_\text{band} \ge 20 \quad\text{and}\quad \mathrm{SNR} \ge 2 .
$$

At least 20 pixels, because a real search would not follow up a handful of
pixels; 2 sigma, because the question is whether the stream can be seen. The
reported number is the **fraction of injected streams detected** at each
surface brightness, with a Wilson 68% interval
({py:func}`~streamgoggles.evaluation.footprint.stream_detection`).

This is the significance of one band placed where the stream is known to be.
A blind search tries a huge number of positions and orientations across the
survey, and will need a stricter threshold (the look-elsewhere effect); that
is the survey-wide contamination experiment below.

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
| {doc}`hyperparameters` | Which training length, training surface brightness range, network depth and width, learning rate and batch size detect most streams at SB 33 and some at SB 34, with a clean background? | done for training range, length and network size: train on SB 32-34.5 and average several trainings; learning rate and batch size not run yet |
| Wider stream parameter space | Detection as a function of surface brightness and distance modulus (2-D), then width, length, age and metallicity. | planned |
| Generic matched filter | One filter swept over trial distance modulus, with the filter's parameters (age, metallicity, trial distance) given to the network as inputs. | planned |
| Stream populations | Several streams injected in the footprint; per-stream (object-level) metrics next to the per-pixel ones. | planned |
| Survey-wide contamination | False alarms over a fully tiled footprint, rather than around one injected stream. | planned |
| DES survey model | Switch the simulations from LSST year 1 to the DES Y6 survey model (`survey="des"`, `release="yr6"` in streamobs), so that detection limits can be compared with the DES 2018 stream search, whose known streams reach surface brightness 34-34.3. Includes the magnitude cut: g < 23.5 as in DES Y3 (chosen against depth and galaxy-contamination fluctuations) versus 24 or 24.5, judged on real data since the simulated background has depth variations only at 27' scale and unclustered galaxies. Needs the DES light-background resources built in streamobs first. | planned |
| Real data | Train with the known streams masked, then unmask and check that they are recovered. | planned |

```{toctree}
:maxdepth: 1
:hidden:

loss_selection
hyperparameters
```
