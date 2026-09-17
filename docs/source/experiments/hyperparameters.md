# Hyperparameters

**Question.** The streams of interest are between surface brightness 33 and
34 mag arcsec⁻². Which training length, training surface brightness range,
network depth and width, learning rate and batch size give a model that
**detects most streams at SB 33** (missing a few is acceptable) **and declines
gradually through SB 34** rather than dropping from nearly all to nothing,
while keeping the background as clean as the current model's?

A first real-data target is to reproduce the DES 2018 stream search: its known
streams reach surface brightness 34 to 34.3, so detections at SB 34 and just
beyond matter, not only at 33. How detectable a real stream is will also
depend on where it lies (survey depth, extinction, footprint edges), which
these simulations do not vary.

**Status.** In progress: phase 1 done, phase 2 running.

## Starting point

The model from {doc}`loss_selection`: batch Dice loss, batch size 8,
background fraction 0.05, U-Net with depth 2 and base width 12, 1200 training
windows (40 epochs of 30), learning rate 2e-3, trained on surface brightness
31, 32, 33 and 34. Its background is clean, but it misses many faint streams.

Since the loss selection, the magnitude range was tightened to 16-24.5 in g
and r (it was 16-25). Every model on this page uses the new range.

**Detection criterion.** A stream counts as detected at a probability
threshold when at least 20 pixels are flagged within 1σ of its track and the
signal-to-noise of their density against stream-shaped background bands is
at least 2 (see "Is the stream detected?" in {doc}`index`). The main number
is the fraction of injected streams detected at each surface brightness,
at thresholds 0.5 and 0.1.

**Baseline.** Two models trained for the loss selection (seed 42, magnitude
range 16-25), scored with this criterion on 20 streams per surface brightness,
on a background they never saw:

| Surface brightness | batch Dice, batch 8, threshold 0.5 | threshold 0.1 | Dice, batch 2, threshold 0.5 | threshold 0.1 |
|---|---|---|---|---|
| 32 | 100% | 100% | 100% | 100% |
| 33 | 55% | 70% | 100% | 100% |
| 33.5 | 0% | 10% | 95% | 95% |
| 34 | 0% | 0% | 55% | 70% |
| 34.5 | 0% | 0% | 15% | 30% |

- **The faint signal is learnable.** Plain Dice detects every SB 33 stream and
  more than half at SB 34, with an SNR of about 28 at SB 33 and 5 at SB 34.
- **But plain Dice flags about 0.7% of the background everywhere**, which a
  survey-wide search cannot afford. Batch Dice flags none, but at SB 33 it
  flags on average only 28 of the ~218 pixels within 1σ of the track, and at
  SB 33.5 almost nothing.
- **The goal is plain Dice's sensitivity with batch Dice's clean background.**
  One hint: plain Dice itself loses faint streams when trained at batch 8
  instead of batch 2 (the same number of training windows, so 4 times fewer
  optimizer steps), which suggests the batch-8 models are undertrained.

## Plan

Loss, batch size (until phase 3), background fraction and magnitude range
stay fixed. Every model is scored on surface brightness 32, 33, 33.5, 34,
34.5 and 35, with 20 streams each, on the independent background, with the
per-threshold counts and track bands saved, so any threshold can be analysed
later without retraining.

| Phase | Varied | Values | Why |
|---|---|---|---|
| 1 | training length | 1200, 4800, 9600 windows (150, 600, 1200 optimizer steps) | the batch-8 models look undertrained |
| 1 | training surface brightness | 31-34 (current); 32, 33, 33.5, 34, 34.5 | more examples near the detection edge |
| 2 | network depth and base width | depth 2, 3, 4; width 12, 24 (from 4800 windows, training SB 32-34.5) | a faint stream is only visible by adding up pixels along its track (~70 px long), while a depth-2 network sees ~30-40 px at once |
| 3 | learning rate and batch size | 1e-3, 2e-3, 5e-3; batch 8, 16 | tune the optimization of the chosen setup |

Each phase starts from the best configuration of the previous one. Phases
are screened with 2 seeds per configuration; the best configurations are then
confirmed with 4 seeds, as the {doc}`shared protocol <index>` requires.

## Results

### Phase 1: training length and training surface brightness

Six configurations, 2 seeds each (40 streams per surface brightness), depth 2,
width 12, learning rate 2e-3, batch 8.

```{image} figures/hyperparameters/phase1_detection.png
:alt: Fraction of streams detected against surface brightness for the six phase-1 configurations, at thresholds 0.5 and 0.1
:width: 100%
```

```{image} figures/hyperparameters/phase1_pixel_metrics.png
:alt: Completeness, contamination and contrast at threshold 0.5 for the six phase-1 configurations
:width: 100%
```

At threshold 0.5:

| Training windows | Training SB | SB 33 | SB 33.5 | SB 34 | Background density in stream-free bands |
|---|---|---|---|---|---|
| 1200 | 31-34 (starting point) | 68% | 10% | 0% | 0.005% |
| 4800 | 31-34 | 80% | 23% | 7% | 0.01-0.02% |
| 9600 | 31-34 | 80% | 23% | 10% | 0.005% |
| 1200 | 32-34.5 | **100%** | **80%** | **38%** | 0.45-0.5% |
| 4800 | 32-34.5 | 88% | 17% | 0% | ≤ 0.005% |
| 9600 | 32-34.5 | 85% | 42% | 20% | 0.06-0.08% |

- **Detecting faint streams and keeping the background clean trade off
  against each other.** Only one configuration meets the target (1200 windows
  on the fainter range: every SB 33 stream, 80% at 33.5, 38% at 34, both seeds
  agreeing), and it flags about 0.5% of the background, close to plain Dice.
  Every configuration with a clean background stops at 80-88% at SB 33 and
  0-10% at SB 34, and lowering the threshold to 0.1 barely changes that: those
  models do not respond to the faintest streams at all.
- **Training longer helps a little on the current range**: SB 33 from 68% to
  80%, SB 33.5 from 10% to 23%, with a clean background.
- **Seeds disagree a lot at the faint end**: for 9600 windows on the fainter
  range, one seed detects 35% of the SB 34 streams and the other 5%.
  Differences below about 15 points are not established.

Training time: about 1.5, 3.5 and 6 minutes for 1200, 4800 and 9600 windows.

**Next.** Training length does not escape the trade-off, so phase 2 tests
whether the network's size does. A deeper network sees a longer piece of a
track at once, and could recognize a faint stream by its coherent line rather
than by local excesses of pixels, which background fluctuations also produce.
Phase 2 starts from 4800 windows on the fainter range (a clean background and
88% at SB 33).

**Fixed before any result was kept.** The first phase-1 run stopped when a
long training could not find a valid window for one training stream. The
investigation found two problems in the training-window sampler, both fixed
before phase 1 was restarted from scratch:

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

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/experiments/hyperparameters/run.py phase1      # then phase2, ...
python scripts/experiments/hyperparameters/figures.py phase1  # figures and numbers
```

`run.py` resumes where it stopped and re-scores saved models without
retraining. Models and results are written to
`data/experiments/hyperparameters/`.
