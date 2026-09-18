# Hyperparameters

**Question.** The streams of interest are between surface brightness 33 and
34 mag arcsec⁻². Which training length, training surface brightness range,
network depth and width give a model that **detects most streams at SB 33**
(missing a few is acceptable) **and declines gradually through SB 34** rather
than dropping from nearly all to nothing, while keeping the background as
clean as the current model's?

A first real-data target is to reproduce the DES 2018 stream search: its known
streams reach surface brightness 34 to 34.3, so detections at SB 34 and just
beyond matter, not only at 33. How detectable a real stream is will also
depend on where it lies (survey depth, extinction, footprint edges), which
these simulations do not vary.

**Status.** Training range, training length, network size and model averaging
are done: 58 trainings, up to 6 seeds per configuration, each scored on 20
injected streams at each of 6 surface brightnesses. Learning rate and batch
size are not varied yet.

## Starting point

The model from {doc}`loss_selection`: batch Dice loss, batch size 8,
background fraction 0.05, U-Net with depth 2 and base width 12, 1200 training
windows (40 epochs of 30), learning rate 2e-3, trained on surface brightness
31, 32, 33 and 34. Its background is clean, but it misses many faint streams.

Since the loss selection, the magnitude range was tightened to 16-24.5 in g
and r (it was 16-25). Every model on this page uses the new range.

## How to read the figures

**Window.** A window is one 96 × 96 pixel cut-out of the count maps (about
11° on a side), which is what the network sees at once. In training, 95% of
the windows are drawn around an injected stream and 5% are stream-free
background windows (`background_fraction = 0.05`). Windows are redrawn every
epoch from newly injected streams, so "4800 windows" means 40 epochs of 120
freshly simulated windows, not one set of 120 seen 40 times. Training cost is
proportional to this number: about 1.5, 3.5, 6 and 11 minutes for 1200, 4800,
9600 and 19200 windows on this machine.

**Training, seed.** Every configuration is trained several times with
different seeds. A seed changes the training data drawn, the augmentation and
the initial weights all at once, so two seeds are two independent trainings of
the same configuration, not two random initializations of the same data.

**Detection.** A stream counts as detected at a probability threshold when at
least 20 pixels are flagged within 1σ of its track *and* the signal-to-noise
of their density against 200 stream-shaped background bands is at least 2 (see
"Is the stream detected?" in {doc}`index`). Every figure uses threshold 0.5.
That value is not optimized: it is the natural threshold of a sigmoid output,
and fixing one value keeps the models comparable. But a fixed threshold does
not put two models at the same false-alarm rate, so figure 6 redoes every
comparison at matched background instead — and it changes some of the
conclusions, which is why it is worth reading before acting on figures 1 to 3.

**Two different uncertainties**, which the figures draw differently:

- **Error bars are sampling uncertainty.** Each model is scored on 20 injected
  streams per surface brightness, so a configuration with 6 seeds contributes
  120 streams. "k of n streams detected" is binomial — n is fixed and k ≤ n —
  so the bars are a **Wilson 68% interval**: the set of detection rates p for
  which the observed k is within one standard deviation, solved for p rather
  than centred on k/n. Unlike a √k bar it stays inside [0, 1] and keeps a
  sensible width at k = 0 and k = n, where √k would be zero or reach past 1.
  The bars are asymmetric for the same reason.
- **Spread between trainings.** The comparison figures (1, 2, 3, 6) show only
  the mean over a configuration's trainings, to stay readable. The spread
  itself is the subject of figure 4, where each point is one trained model,
  and of figure 5, where each thin line is one trained model.

**Background density** is the fraction of pixels flagged inside stream-shaped
bands placed on a sky with no stream in it — the false-alarm rate at the scale
a search actually looks at. A density of 1e-3 means one pixel in a thousand,
so of order 40 false pixels in a band the size of a stream's track.

## What was varied

Loss, batch size, background fraction and magnitude range stay fixed. Every
model is scored on surface brightness 32, 33, 33.5, 34, 34.5 and 35, with 20
streams each, on a background it never saw during training, with the
per-threshold counts and track bands saved so any threshold can be analysed
later without retraining.

| Varied | Values | Seeds | Why |
|---|---|---|---|
| training surface brightness | 31-34 (starting point); 32-34.5 | 6 each, except 2 for 9600 windows on 31-34 | more examples near the detection edge |
| training length | 1200, 4800, 9600, 19200 windows | 6 | the batch-8 models look undertrained |
| network depth and base width | depth 2, 3, 4; width 12, 24 | 6 at width 12, 2-4 at width 24 | a faint stream is only visible by adding up pixels along its track (~70 px long), while a depth-2 network sees ~30-40 px at once |
| models averaged into one prediction | 1, 2, 3, 4, 6 | — | trainings differ a lot; does averaging them recover the faint end? |

## Results

### 1. The training surface brightness range sets where the model operates

**Statement.** At a fixed threshold, which surface brightnesses the model is
trained on decides where its detections stop, far more than anything else
tested. Moving the training range from 31-34 to 32-34.5 raises detection at
SB 34 from 5-8% to 20-36% and at SB 33.5 from 16-20% to 48-67% — and makes the
background 10 to 30 times dirtier at the same time. Section 6 shows that this
is mostly a change of operating point rather than a better model: at a matched
false-alarm rate the two ranges perform almost the same.

```{image} figures/hyperparameters/1_training_range.png
:alt: Fraction of streams detected against surface brightness, for 1200 and 4800 training windows on each of the two training ranges
:width: 100%
```

*Fraction of injected streams detected at threshold 0.5. Each curve is the
mean over 6 trainings of that configuration (120 streams per point). Error
bars are the Wilson 68% sampling interval on that pooled fraction; the spread
between the 6 trainings is not shown here — see figure 4. The shaded band is
the SB 33-34 target region. Points are offset horizontally so the bars stay
readable.*

At threshold 0.5, pooled over 6 trainings each:

| Training windows | Training SB | SB 33 | SB 33.5 | SB 34 | Background density |
|---|---|---|---|---|---|
| 1200 | 31-34 | 62% | 16% | 5% | 2.7e-4 |
| 4800 | 31-34 | 79% | 20% | 8% | 1.9e-4 |
| 1200 | 32-34.5 | 94% | 67% | 36% | 5.4e-3 |
| 4800 | 32-34.5 | 92% | 48% | 20% | 1.6e-3 |

This is the whole point of the experiment, and also its main warning: the
model responds to the faint streams it was *shown*, not to faint streams in
general. Nothing here says the network learned a general notion of "stream";
it learned where the decision boundary should sit for the population it was
trained on.

### 2. Training longer is a real improvement

**Statement.** Training longer buys a cleaner background at the same detection
rate: from 1200 to 19200 windows the fraction of background flagged drops by a
factor 4 while detection at SB 33.5 stays between 48% and 67%. Read at a
matched false-alarm rate, where the two effects are separated, it is a plain
gain: SB 33.5 goes from 31% to 60%.

```{image} figures/hyperparameters/2_training_length.png
:alt: Detection fraction and background density against surface brightness for 1200 to 19200 training windows on the fainter range
:width: 100%
```

*Both panels: models trained on SB 32-34.5, at threshold 0.5, colour by
training length, 6 trainings per configuration. Left: fraction of streams
detected, mean over those trainings, with Wilson 68% sampling bars — the four
lengths sit within about 15 points of each other, which is the size of the
spread between trainings of a single configuration (figure 4). Right: fraction
of pixels flagged inside stream-shaped bands on a stream-free sky, same models,
log scale.*

At SB 33.5, with the last two columns from figure 6's construction:

| Training windows | detected at 0.5 | background at 0.5 | detected at 1e-4 | detected at 1e-3 |
|---|---|---|---|---|
| 1200 | 67% | 5.4e-3 | 7% | 31% |
| 4800 | 48% | 1.6e-3 | 21% | 44% |
| 9600 | 57% | 2.0e-3 | 25% | 47% |
| 19200 | 62% | 1.3e-3 | 32% | 60% |

The first two columns are why a fixed threshold misleads: the 1200-window
model has the most detections *and* four times the false alarms, so nothing
can be concluded by reading detections alone. The last two columns hold the
threshold's effect fixed and are monotone in training length, at both
false-alarm budgets. A longer training is more confident, so fewer marginal
pixels pass any given threshold — but the pixels it does keep are better
chosen.

An earlier version of this section, written when 9600 and 19200 windows still
had 3-4 seeds, reported a clean "longer training trades detections for a
cleaner background". With 6 seeds the detection differences at 0.5 shrank into
the training-to-training spread and the background ordering stopped being
monotone (9600 is dirtier than 4800). The matched-background columns were
stable throughout, which is another reason to prefer them.

### 3. Depth and width change nothing

**Statement.** Network depth (2, 3, 4) and base width (12, 24) do not move
detection outside the spread between trainings of the same configuration.

```{image} figures/hyperparameters/3_architecture.png
:alt: Detection fraction against surface brightness for depth 2, 3 and 4 and base width 12 and 24
:width: 100%
```

*All models: 4800 windows, training SB 32-34.5, threshold 0.5. Each curve is
the mean over that configuration's trainings (6 seeds for the width-12
configurations, 2-4 for width 24), with Wilson 68% sampling bars.*

The motivation was that a faint stream is only visible by adding up pixels
along a track ~70 pixels long, while a depth-2 U-Net's receptive field covers
only 30-40 pixels, so a deeper network should see the line as a line. The
data does not support it: at SB 33.5 the depth-2, 3 and 4 models sit at 48%,
41% and 40%, within the seed-to-seed scatter of a single configuration
(σ ≈ 0.2-0.3 at that surface brightness, from figure 4).

This is also where an earlier version of this page was wrong. Screened with
2 seeds, depth 4 looked like it softened the drop between SB 33 and 34; with
6 seeds that difference disappeared. Two seeds are not enough to compare
configurations here, and the width-24 rows above, which still have 2-4 seeds,
should be read the same way.

### 4. The same configuration, retrained, gives very different models

**Statement.** Training-to-training scatter is the largest effect on this
page at the faint end. Retraining one configuration with a different seed can
change the SB 33.5 detection rate from about 15% to about 90%.

```{image} figures/hyperparameters/4_variability.png
:alt: Detection fraction of each individual trained model, for three configurations, against surface brightness
:width: 100%
```

*One point per trained model (6 per configuration), scored on its own 20
streams per surface brightness; the horizontal bar is their mean. Threshold
0.5. Sampling bars are omitted so the spread stays visible — the scatter of
the points is the quantity of interest, and it is several times larger than
the sampling uncertainty on each point (±0.1 for 20 streams).*

Standard deviation over trainings, at threshold 0.5:

| Configuration | SB 33 | SB 33.5 | SB 34 |
|---|---|---|---|
| 1200 w, SB 32-34.5 | 0.10 | 0.32 | 0.27 |
| 4800 w, SB 32-34.5 | 0.05 | 0.33 | 0.17 |
| 4800 w, SB 31-34 | 0.07 | 0.13 | 0.04 |
| 19200 w, SB 32-34.5 | 0.00 | 0.07 | 0.07 |

Two consequences. First, any comparison of two configurations that differ by
less than ~15 points at SB 33.5 is not established, however clean the curves
look. Second, a single trained model is not a reliable product: the
configuration says what the *typical* model does, and the one you actually
trained may sit anywhere in that range. That is the problem the next figure
addresses.

### 5. Averaging several trainings is what recovers the faint end

**Statement.** Averaging the per-pixel probabilities of N independently
trained models, then thresholding once, detects more faint streams than any of
the models alone, and removes the seed lottery: four or more averaged models
take SB 33.5 from 20% to 55-60%. At equal training cost, averaging four short
trainings matches a single long one almost exactly.

```{image} figures/hyperparameters/5_ensemble.png
:alt: Detection fraction for ensembles of 1 to 6 averaged models, and the equal-cost comparison between four averaged 4800-window models and single 19200-window models
:width: 100%
```

*Left: N models of 4800 windows (SB 32-34.5) whose probability maps are
averaged into one prediction, which is then thresholded at 0.5 — the curves
are model averages, one prediction each, scored on the same 20 streams per
surface brightness, with Wilson 68% sampling bars. Right: the same total
training cost spent two ways. Thin blue lines are individual 19200-window
trainings, one per seed, and the thick blue line is the average of those
curves — a summary of several models, not a model you could deploy. The
orange curve is the model average of four 4800-window trainings (4 × 4800 =
19200 windows), the same object as the left panel's, with its sampling bars.*

Averaging four or six models takes SB 33.5 from 20% to 60% and 55%, and SB 34
from 0% to 20%, while the background density stays at 1.1e-3 and 4.8e-4 —
below the 1.6e-3 of a typical single 4800-window model. Two and three models
are not enough: the gain appears between three and four. The mechanism is the
one figure 4 exposes: each training flags a different, partly random subset of
the marginal pixels, so the false alarms average down while the pixels that
many models agree on survive. This costs N trainings and N forward passes per
prediction — expensive, but the forward passes are cheap compared with
training, and they parallelize.

Each ensemble curve is one prediction scored on 20 streams per surface
brightness, so its sampling bars are wide (±0.11 at 50%) and the non-monotonic
wiggles between sizes 2, 3 and 4 are noise. The step from ≤3 to ≥4 models,
0.20 to 0.55-0.60, is larger than that; section 6 shows the same step is
weaker once the false-alarm rate is held fixed, so the honest summary is that
averaging clearly helps at a fixed threshold and probably, but not yet
measurably, at a fixed false-alarm rate.

The equal-cost comparison matters for how to spend a fixed budget. Four
averaged trainings of 4800 windows and one training of 19200 windows use the
same number of simulated windows. The single long training does well on
average — and the two curves agree within their bars at every surface
brightness — but its individual curves scatter (thin lines), and you get one
draw from that scatter. The average of four gives one prediction, whose
remaining spread is much smaller. At this budget the choice is therefore about
reproducibility rather than sensitivity.

### 6. Which of those effects are real, and which are the threshold moving

**Statement.** Compared at the threshold where each model flags the same
fraction of stream-free sky, the training range's large advantage nearly
disappears, while training length and model averaging keep theirs. The
training range mostly moves the operating point; training length and averaging
move the model.

```{image} figures/hyperparameters/6_matched_background.png
:alt: Detection fraction against surface brightness for four configurations and for one versus six averaged models, each thresholded at its own background density of 1e-4 and 1e-3
:width: 100%
```

*Each model's threshold is chosen per panel so that it flags the target
fraction of stream-free sky (1e-4 left, 1e-3 right) instead of using 0.5
everywhere; curves are then means over that configuration's trainings. The two
star curves are a single 4800-window model and an average of six, treated the
same way — one prediction each, not a mean over trainings, which is why the
single-model star sits slightly below the 4800-window curve above it. Detection here is the ≥ 20 flagged pixels condition only: the
SNR compares the band with the background, which this construction has already
fixed. The panels are two operating points, not two experiments — 1e-4 is a
stricter false-alarm budget than 1e-3, so every curve sits lower on the left.*

A fixed threshold puts these models at false-alarm rates that differ by a
factor of 30, so this is the fair comparison, and the one a survey cares about:
the false-alarm rate is the resource being spent. At a matched rate, at SB 33.5:

| | 1e-4 | 1e-3 |
|---|---|---|
| 4800 w, SB 31-34 | 20% | 35% |
| 4800 w, SB 32-34.5 | 21% | 44% |
| 1200 w, SB 32-34.5 | 7% | 31% |
| 19200 w, SB 32-34.5 | 32% | 60% |
| 1 model (4800 w) | 20% | 35% |
| 6 models averaged | 30% | 55% |

The two training ranges are within a few points of each other — nothing like
the 48% against 20% that figure 1 shows at threshold 0.5. Training on fainter
streams mainly makes the model less conservative, which a threshold can do
too. What survives the matched comparison is training length: 31% → 60% at 1e-3
and 7% → 32% at 1e-4, monotone in both. Averaging points the same way (35% →
55% at 1e-3, 20% → 30% at 1e-4) but each ensemble is one prediction scored on
20 streams, so those are 7/20 against 11/20 and 4/20 against 6/20 — the right
direction, not yet a measurement. Confirming it needs more evaluation streams
per ensemble, not more models.

This does not make the training range irrelevant. A model trained on 31-34
cannot be pushed to the faint end by lowering its threshold indefinitely — at
1e-3 it saturates around 35% at SB 33.5 — and the faint range is what lets the
longer trainings and the ensembles reach the numbers above. But the headline
"training range doubles the SB 34 detections" is an operating-point effect,
and should be reported as one.

## What to use

For the streams this project targets, on this training population:

- **Set the threshold from a false-alarm budget, not to 0.5.** This is the
  first decision, not the last one: at 0.5 two models that differ only in
  training length sit at false-alarm rates an order of magnitude apart, and
  comparing them there points the wrong way (section 2).
- **Train on SB 32-34.5**, bracketing the target rather than matching it. Its
  advantage at a fixed threshold is mostly an operating-point shift, but the
  longer trainings and the ensembles that do help are all on this range, and
  the narrow range saturates below them (section 6).
- **Train as long as you can afford**: 19200 windows is the best model here at
  a matched false-alarm rate, and 4800 is a reasonable compromise at a quarter
  of the cost.
- **Depth 2, width 12.** Nothing larger helped, so the cheapest network wins.
- **Average 4 to 6 trainings** into one prediction. This is what turns a model
  that may or may not see SB 34 streams into one that reliably sees some of
  them, and it is the only lever that both raises detection and lowers the
  false-alarm rate.

## Caveats

These conclusions are narrower than they may look.

- **Single streams, not a population.** Every number is the fraction of
  *individually injected* streams recovered. A survey-wide search faces many
  trials at once and a false-alarm budget over the whole footprint; nothing
  here measures that.
- **Highly sensitive to the training range.** Figure 1 is the clearest
  result on this page, and it is also the evidence that the model's
  sensitivity is set by what it was shown. Retraining on a different range
  moves the detection edge, so "the model detects SB 34 streams" is a
  statement about a training choice, not about the network.
- **One stream shape.** Distance, width, length, and the stellar population
  are fixed. Varying them will change the picture, and the seed-to-seed
  spread of figure 4 warns that differences will need many trainings to
  establish — a comparison screened on 2 seeds already misled us once
  (section 3).
- **One background, one survey.** A single simulated background realization
  family and one survey model; no extinction variation, no depth
  inhomogeneity, no real data yet.

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/experiments/hyperparameters/run.py training_length   # or phase1, phase2, depth
python scripts/experiments/hyperparameters/ensemble.py              # averaged-model predictions
python scripts/experiments/hyperparameters/summary.py               # the six figures above
python scripts/experiments/hyperparameters/figures.py phase1        # per-phase diagnostics
```

`run.py` resumes where it stopped and re-scores saved models without
retraining. Models and results are written to
`data/experiments/hyperparameters/`.

## Appendix: fixed before any result was kept

The first run stopped when a long training could not find a valid window for
one training stream. The investigation found two problems in the
training-window sampler, both fixed before the experiment was restarted from
scratch:

- **RA wrap.** The length of a stream inside a candidate window was measured
  in a projection centred on the mean RA of its stars. For a stream crossing
  RA 0°/360° that mean is near 180°, on the far side of the sky, so the
  "at least 5° of stream in the window" check was meaningless for about 13%
  of training streams (lengths of millions of degrees were measured).
- **Inefficient proposals.** Candidate windows were centred up to a full
  window size away from a stream star, so about half missed the stream
  entirely. Measured on real placements, the median acceptance per attempt was
  0.18 (0.05 at worst), so failing 100 attempts happened about once per
  10,000 training streams, i.e. likely once in a 9600-window run. Centring
  within half the window diagonal keeps every valid window reachable and
  doubles the acceptance (median 0.34, worst 0.11). If no window fits at all,
  the stream is now placed again instead of stopping training.
