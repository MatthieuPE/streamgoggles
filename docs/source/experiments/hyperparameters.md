# Hyperparameters

**Question.** The streams of interest are between surface brightness 33 and
34 mag arcsec⁻². Which training length, training surface brightness range,
network depth and width, learning rate and batch size give a model that
**detects every stream at SB 33 and some at SB 34**, while keeping the
background as clean as the current model's?

**Status.** In progress: phase 1 is running.

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
| 2 | network depth and base width | depth 2, 3, 4; width 12, 24 | a faint stream is only visible by adding up pixels along its track (~70 px long), while a depth-2 network sees ~30-40 px at once |
| 3 | learning rate and batch size | 1e-3, 2e-3, 5e-3; batch 8, 16 | tune the optimization of the chosen setup |

Each phase starts from the best configuration of the previous one. Phases
are screened with 2 seeds per configuration; the best configurations are then
confirmed with 4 seeds, as the {doc}`shared protocol <index>` requires.

## Results

To come, phase by phase.

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
python scripts/experiments/hyperparameters/run.py phase1
```

`run.py` resumes where it stopped and re-scores saved models without
retraining. Models and results are written to
`data/experiments/hyperparameters/`.
