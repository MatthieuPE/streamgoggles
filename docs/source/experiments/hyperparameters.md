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

**Status.** Training range, training length, network size, model averaging and
the decoy channel are done: 70 trainings, up to 6 seeds per configuration, each
scored on the same 20 injected streams at each of 6 surface brightnesses; the
selected model and the averages also on 300 to 500 streams per point. Learning rate and batch
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

**Streams per point: an evaluation quantity, never a training one.** Every
count of injected streams on this page — 20, 100, 500 — describes how a trained
model was *scored*, and says nothing about what it was trained on. It is also
**per surface brightness**, not a total: "20 streams" means 20 independently
injected streams at SB 32, 20 more at SB 33, and so on over the six evaluation
brightnesses, so 120 injections for that model in total but 20 behind each
plotted point, which is what its error bar is computed from. Each one is a
fresh full-sky injection — different stream realization, placement, orientation
and survey noise — on a background catalog the model never saw.

Raising that count makes the *measurement* of a model more precise. It cannot
make the model better, and it is not what the training-length section varies.

**Counting the training set.** Training is counted in windows, and the
conversion to streams is worth spelling out because the two units appear side
by side on this page.

A training window is drawn fresh each time it is needed — the dataset is
generated on the fly, never reshuffled from a fixed pool — so the number of
windows a run consumes is

$$
\text{windows} = \text{epochs} \times \text{windows per epoch}
$$

which for every model here is 40 epochs of 30, 120, 240 or 480, giving the
1200, 4800, 9600 and 19200 of section 2. `steps_per_epoch` in the code is that
windows-per-epoch count, not a count of optimizer steps: at batch 8, 120
windows per epoch is 15 optimizer steps.

Each window is either a **stream window** (probability
$1 - f_\text{bg}$, with `background_fraction` $f_\text{bg} = 0.05$) or a
stream-free **background window**. A stream window contains exactly one freshly
injected stream, whose surface brightness is drawn uniformly at random from the
$V$ discrete values of the training range ($V = 4$ for SB 31-34, $V = 5$ for
SB 32-34.5), and the window is positioned so that at least 5° of the stream's
8° track lies inside it. So the expected number of distinct streams a model
sees at each surface brightness value is

$$
\frac{\text{windows} \times (1 - f_\text{bg})}{V}
$$

| Training windows | Stream windows | Streams per SB value (SB 32-34.5, $V=5$) |
|---|---|---|
| 1200 | 1140 | ~228 |
| 4800 | 4560 | ~912 |
| 9600 | 9120 | ~1824 |
| 19200 | 18240 | ~3648 |

The per-value counts are expectations, not exact: the surface brightness is
drawn per window, so the realized count fluctuates by about
$\sqrt{N p (1-p)} \approx 27$ streams for the 4800-window models. No stream is
ever seen twice, so "more windows" always means more distinct streams rather
than more passes over the same ones.

Those are the numbers that change the model. The 20/100/500 above are the
numbers that change the error bar on its measured performance.

**Two different uncertainties**, which the figures draw differently:

- **Error bars are sampling uncertainty.** Each model is scored on 20 injected
  streams per surface brightness — and every model is scored on **the same 20
  skies per point**, so that two configurations are always compared on
  identical skies. That makes comparisons paired and sharp, but it has a cost
  for absolute rates: a configuration curve pools 6 trainings, yet those are 6
  models × the same 20 streams, not 120 independent streams. The uncertainty
  from *which* 20 skies were drawn (about ±11 points at 50%) is shared by every
  seed and does not average down, so the pooled bars are too narrow for an
  absolute rate. Use the configuration curves to compare; section 9 measures
  the selected model's absolute rates on 300 skies per point. An ensemble or a
  deployed model — one prediction — is scored on a few hundred streams per
  point for that reason; figure 7 shows what that changes. "k of n streams detected" is binomial — n is fixed and k ≤ n —
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
| models averaged into one prediction | 1, 2, 3, 4, 6 | one prediction each, 500 streams per point | trainings differ a lot; does averaging them recover the faint end? |

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
the sampling uncertainty on each point (±0.1 for 20 streams per point).*

Standard deviation over trainings, at threshold 0.5:

| Configuration | SB 33 | SB 33.5 | SB 34 |
|---|---|---|---|
| 1200 w, SB 32-34.5 | 0.10 | 0.32 | 0.27 |
| 4800 w, SB 32-34.5 | 0.05 | 0.33 | 0.17 |
| 4800 w, SB 31-34 | 0.07 | 0.13 | 0.04 |
| 19200 w, SB 32-34.5 | 0.00 | 0.07 | 0.07 |

Two consequences. First, any comparison of two configurations that differ by
less than ~15 points at SB 33.5 is not established, however clean the curves
look. Second, a single short training is not a reliable product: the
configuration says what the *typical* model does, and the one you actually
trained may sit anywhere in that range. There are two ways out — average
several trainings, or train one for longer. The last row already hints at the
second: at 19200 windows the spread falls to 0.07. Section 5 compares the two.

### 5. Averaging several trainings helps, but training longer helps more

**Statement.** Averaging the per-pixel probabilities of several independently
trained 4800-window models, then thresholding once, detects more faint streams
than one of them alone: six averaged take SB 33.5 from 37% to 55% at a matched
false-alarm rate of 1e-3. But the same compute spent on one longer training
does better. One 19200-window training detects 55-65% whichever seed it gets,
against 36% for four averaged 4800-window trainings of the same total cost, and
it needs one forward pass instead of four.

```{image} figures/hyperparameters/5_ensemble.png
:alt: Detection fraction for ensembles of 1 to 6 averaged models, and the equal-cost comparison between four averaged 4800-window models and single 19200-window models
:width: 100%
```

*Left: N models of 4800 windows (SB 32-34.5) whose probability maps are
averaged into one prediction, which is then thresholded at 0.5 — the curves are
model averages, one prediction each, scored on 500 streams per surface
brightness, with Wilson 68% sampling bars. Right: the same total training cost
spent two ways, at threshold 0.5. Thin blue lines are individual 19200-window
trainings, one per seed, and the thick blue line is the average of those curves
— a summary of several models, not a model you could deploy. The orange curve
is the model average of four 4800-window trainings (4 × 4800 = 19200 windows),
the same object as the left panel's.*

At SB 33.5, the averages scored on 500 streams each:

| Models averaged | detected at 0.5 | background density at 0.5 | detected at 1e-4 | at 1e-3 |
|---|---|---|---|---|
| 1 | 24% | 6.3e-6 | 32% | 37% |
| 2 | 20% | 6.0e-6 | 28% | 48% |
| 3 | 28% | 4.0e-5 | 27% | 38% |
| 4 | 54% | 6.9e-4 | 27% | 36% |
| 6 | 47% | 3.1e-4 | 34% | 55% |

**What averaging does.** At threshold 0.5, four or six averaged models detect
twice as many SB 33.5 streams as one (54% and 47% against 24%). Part of that is
the operating point: the averages flag 50 to 100 times more background at 0.5.
Held at a matched false-alarm rate, the six-model average still gains at 1e-3
(55% against 37%, Fisher p = 1.5e-8), but not at the stricter 1e-4 (34% against
32%, p = 0.25). The mechanism is the one figure 4 exposes: each training flags a
different, partly random subset of the marginal pixels, so the false alarms
average down while the pixels that many models agree on survive.

**How many is not the right question.** The sizes are nested: size N averages
the first N of the same six trainings. At 1e-3 the 2-model average gains
(48%), the 3- and 4-model averages do not (38%, 36%), and the 6-model average
does (55%). A trend in N would not go up, down and up again. Which trainings
enter the average matters as much as how many, and these data cannot separate
the two.

**Equal cost: one long training wins.** Four averaged 4800-window trainings and
one 19200-window training use the same number of simulated windows. At
threshold 0.5 they look alike (54% against 62% at SB 33.5, right panel). At a
matched false-alarm rate of 1e-3 they are not, and the comparison has to be
made on identical skies, since the long trainings were scored on the shared 20
and the averages on 500 (see "Error bars" above). On the same 20 skies per
point, the long training detects 60% of SB 33.5 streams on average against 40%
for the four-model average, and the six-model average reaches 55%. One long
training is at least as good as six averaged short ones, for two thirds of the
training and a sixth of the prediction cost. Training longer also shrinks the
seed lottery that averaging was meant to fix.

### 6. Which of those effects are real, and which are the threshold moving

**Statement.** Compared at the threshold where each model flags the same
fraction of stream-free sky, the training range's large advantage nearly
disappears, while training length keeps all of its advantage and model
averaging keeps part of it. The training range mostly moves the operating
point; training length moves the model.

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
| 1 model (4800 w), 500 streams | 32% | 37% |
| 6 models averaged, 500 streams | 34% | 55% |

The two training ranges are within a few points of each other — nothing like
the 48% against 20% that figure 1 shows at threshold 0.5. Training on fainter
streams mainly makes the model less conservative, which a threshold can do
too. What survives the matched comparison is training length (31% → 60% at 1e-3
and 7% → 32% at 1e-4, monotone in both), which is the strongest effect on this
page, and averaging six trainings at the looser budget only (37% → 55% at 1e-3,
Fisher p = 1.5e-8; 32% → 34% at 1e-4, p = 0.25, no gain).

One caution on the 1e-4 column: some models never flag as little as 1e-4 of
the background at any threshold on the grid, and those are left out of that
column's mean — only 3 of the six 19200-window trainings and 4 of the six
4800-window ones reach it at SB 33.5. The 1e-3 column includes every model and
is the one to rely on.

This does not make the training range irrelevant. A model trained on 31-34
cannot be pushed to the faint end by lowering its threshold indefinitely — at
1e-3 it saturates around 35% at SB 33.5 — and the faint range is what lets the
longer trainings reach the numbers above. But the headline
"training range doubles the SB 34 detections" is an operating-point effect,
and should be reported as one.

### 7. How many evaluation streams it takes to say any of this

This section is about measurement precision, not about the models. Nothing is
retrained here: the same trained ensembles are scored three times, on 20, 100
and 500 injected streams per surface brightness.

**Statement.** 20 evaluation streams per point could not tell whether averaging
survives the matched-background test. 100 streams seemed to settle it, including
a small gain at the strict 1e-4 budget. 500 kept the 1e-3 gain and erased the
1e-4 one. The curves themselves barely moved; what changed is which claims
their precision could carry.

```{image} figures/hyperparameters/7_sampling.png
:alt: The six-model ensemble scored on 20, 100 and 500 streams per surface brightness, and the single-model against six-model comparison at each sample size
:width: 100%
```

*Left: the same six-model ensemble — identical weights in all three curves —
scored on 20, 100 and 500 evaluation streams per surface brightness, with
Wilson 68% bars. The scorings are nested: the realization seed is (base seed,
surface-brightness index, realization index), so the first 20 of the 100 and
the first 100 of the 500 are the same skies, and this is a pure sample-size
comparison. Right: the comparison that motivated the reruns, at SB 33.5 and a
matched background density of 1e-3, annotated with the raw counts.*

At SB 33.5 and a matched rate of 1e-3, single model against six averaged:

| Evaluation streams per point | 1 model | 6 averaged | Fisher p |
|---|---|---|---|
| 20 | 7/20 | 11/20 | 0.17 |
| 100 | 44/100 | 61/100 | 0.01 |
| 500 | 185/500 | 273/500 | 1.5e-8 |

The gap stays at about 18 points throughout; only its uncertainty shrinks. The
single model's own rate moved more than the gap did — 44% at 100 streams was a
high draw, and 37% at 500 is where it settles — which is exactly the kind of
shift a ±5-point interval allows.

The same reruns are what removed a claim. At the strict 1e-4 budget, 100
streams showed six averaged models ahead (43/100 against 32/100, p = 0.07,
"suggestive"); at 500 the two are level (34% against 32%, p = 0.25). A
suggestive result at 100 streams per point was noise.

The practical rule: a configuration curve pools several trainings, so 20
evaluation streams per point is enough to compare configurations that differ by
15 points or more. A single prediction — an ensemble, or the model you intend
to deploy — needs a few hundred per point before a 10-point difference means
anything. Evaluation streams are simulated on demand and cost 0.5 to 0.9 s
each, so this is a compute choice, not a limit of the data.

### 8. The fixed decoy box changes nothing

Every model in sections 1-7 was trained with the older decoy channel, a
colour-magnitude box shifted off the isochrone filter's polygon. The notebooks
now use a fixed box at absolute limits (colour 1.2-1.5, g 18-24.5), which
selects no stream star at all and is the same selection at every trial distance
(see {doc}`../narrative/data_generation`). Since that changes an input channel,
the two configurations the conclusion rests on were retrained with it, 6 seeds
each, and scored exactly like the rest of this page.

**Statement.** The fixed decoy costs no sensitivity. At 19200 windows the two
decoys give the same detection rate to the stream, and at 4800 windows the
differences go both ways and are not significant. Every conclusion on this page
holds for the fixed box.

```{image} figures/hyperparameters/8_decoy.png
:alt: Fraction of streams detected against surface brightness at a matched false-alarm rate of 1e-3, for 4800 and 19200 training windows with the shifted and the fixed decoy box
:width: 100%
```

*Fraction of injected streams detected when each trained model is thresholded
to flag 1e-3 of stream-free sky. Each point pools the streams scored by the six
trainings of a configuration (120 per surface brightness, fewer where a model
never reaches the budget), with the Wilson 68% interval on that pooled count.*

| Matched 1e-3 | SB 33 | SB 33.5 | SB 34 | training time |
|---|---|---|---|---|
| 19200 w, fixed box | 94% | 60% | 21% | ~10 min |
| 19200 w, shifted box | 93% | 60% | 22% | ~10 min |
| 4800 w, fixed box | 87% | 52% | 13% | ~3 min |
| 4800 w, shifted box | 88% | 44% | 18% | ~3 min |

These are the shared 20 skies per point, which is what makes the two decoys
directly comparable; section 9 gives the selected model's absolute rates on 300
skies. At SB 33.5 the 19200-window models detect 60 streams out of 100 with either
decoy; at 4800 windows, 62 of 120 against 53 (Fisher p = 0.30), and at SB 34, 16
against 22 (p = 0.38). The per-seed spread is also unchanged: 45% to 65% at
SB 33.5 for the fixed box at 19200 windows, 55% to 65% for the shifted one.

This also settles the discrepancy that prompted the retraining. The first
fixed-box model, trained and scored in `notebooks/train_model.ipynb`, detected
40% of SB 33.5 streams at a matched 1e-3 — one model, scored on 30 streams per
point, on different evaluation skies and with a threshold chosen per surface
brightness. With six trainings on this page's evaluation, the fixed box gives
60%. The notebook's number was the sampling of one model on a small evaluation,
not the decoy.

One practical difference: at threshold 0.5 the quick fixed-box model is more
cautious (it flags 8e-4 of stream-free sky, against 1.5e-3 with the shifted
box), so at 0.5 it would look worse than it is. As everywhere on this page, the
matched false-alarm rate is the fair reading.

### 9. The selected model, measured on 300 skies

Two questions were left about the 19200-window model once section 8 had chosen
its inputs: does averaging several long trainings help, the one combination
not yet tried; and what are its absolute detection rates, given that the
configuration curves above share their 20 skies per point. Both use the six
fixed-box trainings of section 8, with no retraining: each model on its own,
two disjoint triples averaged, and all six averaged, each scored on 300 streams
per surface brightness.

**Statement.** Averaging long trainings adds nothing: the six-model average
detects 52% of SB 33.5 streams against 49% for one model, within the sampling.
And the selected model's absolute rates are lower than the shared-sky curves
suggested: 49% at SB 33.5, not 60%.

```{image} figures/hyperparameters/9_long_models.png
:alt: Fraction of streams detected against surface brightness at a matched false-alarm rate of 1e-3, for the six single 19200-window models on 300 skies, their mean, their average as one prediction, and the same single models on the shared 20 skies
:width: 100%
```

*Each model thresholded to flag 1e-3 of stream-free sky. Thin lines: the six
trainings on their own, 300 streams per surface brightness; thick blue: their
mean; orange: the six averaged into one prediction, same 300 streams, with its
Wilson 68% bars; grey dashed: the same six trainings on the 20 skies per point
every configuration on this page shares.*

At a matched 1e-3:

| | SB 33 | SB 33.5 | SB 34 |
|---|---|---|---|
| one model, 300 skies (mean of 6) | 95% | 49% | 16% |
| one model, per seed | 90-98% | 36-56% | 8-21% |
| two disjoint triples averaged | 97-98% | 51-57% | 17% |
| all six averaged | 97% | 52% | 16% |
| one model, the shared 20 skies | 94% | 60% | 21% |

**Averaging.** On the same 20 skies, averaging all six gives 65% at SB 33.5
against 60% for one model; on 300 skies, 52% against 49%. Neither difference is
significant, and the two disjoint triples agree with each other, because long
trainings scatter much less than short ones. It is not worth the extra scoring
time.

**Absolute rates.** The same six models find 60% of SB 33.5 streams on the
shared 20 skies and 49% on 300. Those 20 skies happen to be easier than average
for this model — the six-model average shows the same drop, 65% to 52% — which
is exactly the shared sky-sampling uncertainty described in "How to read the
figures". It does not undo any comparison on this page, since every
configuration saw the same 20 skies, but the absolute rates of the selected
model are the 300-sky ones. The seed spread on 300 skies is also wider than the
shared skies showed: 36% to 56% at SB 33.5, with one weaker training (seed 46).

## Conclusion: the configuration to use from here

**The model this experiment selects**, and which the rest of the project should
start from unless a later experiment overrides it:

| Setting | Value | Set by |
|---|---|---|
| loss | batch Dice | {doc}`loss_selection` |
| batch size | 8 | {doc}`loss_selection` |
| background fraction | 0.05 | {doc}`loss_selection` |
| magnitude range (g, r) | 16-24.5 | catalog cut, fixed here |
| decoy channel | fixed box, colour 1.2-1.5, g 18-24.5 | section 8 |
| **training surface brightness** | **32, 33, 33.5, 34, 34.5** | sections 1, 6 |
| **training length** | **4800 windows while exploring (~3 min); 19200 for final results (~10 min)** | sections 2, 5, 6, 8 |
| **network** | **depth 2, base width 12** | section 3 |
| **deployment** | **one model** | section 5 |
| learning rate | 2e-3 | not varied yet |
| **threshold** | **set from a false-alarm budget**, not fixed at 0.5 | section 6 |

Written as a decision list, strongest first:

1. **Explore with 4800 windows, report with 19200.** Two tiers, because
   training time multiplies across every experiment still to come (each needs
   several seeds per parameter value). The quick model — about 3 minutes — is
   the tool for exploring, on the assumption that it ranks configurations the
   way the long one would. That assumption is only partly tested: the training
   range ranked the same at 1200 and 4800 windows, but no comparison has been
   repeated at 19200. It also understates the faint end (44% of SB 33.5 streams at a
   matched 1e-3, against 60%), and its seeds scatter twice as much, so: use
   about six seeds per quick comparison, treat its numbers as relative, and
   re-measure anything that goes into a final statement with the long model —
   or switch to the long model when a task turns out to need its sensitivity.
2. **The final model is one training on 19200 windows.** At a matched
   false-alarm rate this is the best single model on the page, and compared on
   identical skies it beats four averaged 4800-window trainings of the same
   total cost (60% against 40% at SB 33.5) and matches six of them (55%), with
   one forward pass instead of six. Measured on 300 skies it detects 95%, 49%
   and 16% of streams at SB 33, 33.5 and 34 (section 9). It costs about 10
   minutes on this laptop.
3. **Do not average — neither short nor long trainings.** Averaging six
   4800-window models helps against one of them (37% → 55% at a matched 1e-3)
   but costs more training than one long model and gets no further; averaging
   six 19200-window models adds nothing measurable (49% → 52%, section 9).
4. **Train on SB 32-34.5**, bracketing the SB 33-34 target rather than matching
   it. At a fixed threshold this looks like the biggest effect on the page; at
   a matched false-alarm rate most of it is the operating point moving. Keep it
   anyway: the narrow range saturates around 35% at SB 33.5 whatever threshold
   it is given, and every model that does better is trained on the wider range.
5. **Keep the network small**: depth 2, base width 12. Depth 3 and 4 and width
   24 changed nothing outside the seed spread, so the cheapest network wins.
6. **Choose the threshold last, from an acceptable false-alarm rate.** At 0.5,
   models that differ only in training length sit at false-alarm rates an order
   of magnitude apart, and comparing them there points the wrong way. At the
   strict budget of 1e-4, no model on this page gets past about a third of
   SB 33.5 streams, so the budget matters as much as the model.

**What the final configuration achieves**: one 19200-window training with the
fixed decoy box, mean over six trainings each scored on 300 injected streams
per surface brightness (section 9), on a background the models never saw:

| Surface brightness | detected at threshold 0.5 | detected at a background density of 1e-3 |
|---|---|---|
| 32 | 100% | 100% |
| 33 | 95% | 95% |
| 33.5 | 48% | 49% |
| 34 | 17% | 16% |
| 34.5 | 3% | 3% |

At threshold 0.5 it flags about 9e-4 of stream-free sky. One training varies:
36% to 56% at SB 33.5 between seeds. The quick 4800-window model has only been
measured on the shared 20 skies (87%, 52% and 13% there, against 94%, 60% and
21% for the long model on the same skies), so read it relative to the long
one. Against the stated goal — most streams at SB 33, a gradual decline
through SB 34 rather than a cliff — the final model reaches the first and about
half of the second: SB 33 is essentially solved, SB 33.5 is about a coin flip,
and SB 34 about one stream in six.

**These are known-location rates.** Each stream is judged along its own,
known track. That is the right measure for recovering streams whose position is
already known — the first real-data target, the DES 2018 streams found by
visual inspection. A blind search for new streams tries every position and
orientation, and would need a stricter, look-elsewhere-corrected threshold;
that belongs to the discovery stage, after validation on real data.

**What is not settled and should not be assumed:**

- The learning rate and batch size were never varied.
- The threshold has no tuned value yet.
- Every number here is for one stream shape at one distance, so nothing above
  is guaranteed to hold once the stream parameters vary — which the DES 2018
  streams will require, since their distances, widths and lengths differ.

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
