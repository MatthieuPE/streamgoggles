# Stage 2 planning: fixing amplitude recovery vs. rethinking the target

**Status:** discussion draft, no code changed. Branch `stage2-detection-planning`.

**Goal, restated precisely:** given a multi-channel map (one channel per
`(matched filter, trial distance)` pair), produce **per-pixel probability
that a stream is present**, for a specific filter at a specific distance.
That is the actual deliverable — not necessarily an exact star count.

**The concrete problem prompting this:** the current pipeline (Stage 1 —
a U-Net trained to regress `label_policy="stream_count"`, the true
stream-only star count per pixel) now localizes streams well but
**does not recover their amplitude**. Real, current numbers from
`train_model.ipynb`'s latest committed run (40 epochs × 30 steps, SB=32,
`distance_modulus=16.0`, single eval point, `nstars=13282`):

- Dice = **0.857** vs. a k·σ baseline's 0.118 — the model finds the right
  pixels, decisively better than a trivial threshold.
- Correlation = 0.73, weighted_recall = 0.996 — strong agreement with the
  true spatial pattern.
- **Predicted peak ≈ 2.5 vs. true peak ≈ 150** — amplitude under-recovered
  by roughly **60×** at the brightest pixel. (An earlier, richer run
  showed the same pattern at a worse ratio, ~1000×, before this round of
  tuning — consistent under-recovery, not a one-off.)

So: excellent *where*, poor *how much*. That gap is the actual thing to
resolve, and it bears directly on whether Stage 1 (count regression) is
even the right intermediate target for a system whose real goal is
detection, not photometry.

## Is this a fundamental limit, or a fixable modeling problem?

**Working answer: fixable modeling problem, not a fundamental one.**
Reasoning, checked against the real numbers above rather than assumed:

- **Signal vs. background, at the pixels that matter.** The mean
  stream-only count over a whole window is tiny (the stream only crosses
  a handful of pixels out of 400), but that's not the relevant
  comparison — matched-filter stream detection in real astronomy works
  precisely *because* the excess at the stream's own pixels is a
  meaningful fraction of the local background, not because it's below
  it. Concretely here: this run's *true* peak (150) against this
  project's typical per-channel background level (order ~750-1200 counts
  in earlier, richer-background runs) is a real, order-tens-of-percent
  excess at those pixels — detectable in principle, and evidently
  detectable in *location* by the network already. The "signal is
  smaller than the background so it truly can't be recovered" hypothesis
  doesn't hold up against these numbers as a *general* explanation
  (there will always be genuinely too-faint cases where it *does* hold —
  see the calibration idea below for how to find that boundary
  empirically instead of assuming it).
- **What actually looks responsible, mechanistically:**
  1. **Regression-to-the-mean under skewed-target MSE.** With most pixels
     near 0 and a few pixels very high, a model minimizing average
     squared error has a real incentive to hedge toward the low end
     almost everywhere — the "safe" prediction under uncertainty is
     closer to the conditional mean than to the true (rare) extreme. This
     is a well-known, generic failure mode of plain regression on
     heavy-tailed targets, not specific to this codebase.
  2. **`log1p` training compresses exactly the signal that matters most.**
     Training in `log1p(count)` space (necessary — raw-count MSE was
     unusably unstable, per `PLAN.md` §6.10's predecessor entries) also
     shrinks the *gradient cost* of being off by hundreds of counts at a
     bright pixel relative to being off by a handful at a faint one. The
     fix for stability plausibly reintroduced (or worsened) the
     amplitude bias.
  3. **Small, near-zero-initialized head.** `base_width=12`/`depth=2`
     (67K parameters) is tiny by segmentation standards, and the
     documented bias-init trick (`head_conv.weight.zero_()`) means the
     network starts predicting a spatially-*flat* value everywhere and
     has to *grow* the spatial contrast from scratch via gradient
     descent — with a training budget still measured in hundreds of
     steps, there may simply not have been enough updates for the
     relevant weights to grow large enough, even though the *direction*
     (where to push up vs. down) was clearly learned fast (Dice got good
     quickly in earlier runs).
  4. **`BatchNorm` with `batch_size=2` on a rare-event target.** Batch
     statistics computed from 2 samples, most of whose pixels are
     near-zero, calibrate the network's normalization for the "typical"
     regime — plausibly suppressing the scale of activations needed to
     ever output a large value, independent of how well the spatial
     *pattern* is learned.
  5. **`WeightedMSELoss`'s real, already-documented bug** (global, not
     per-sample, weight normalization — `datasets_and_models.md`) isn't
     in play in the *current* config (`loss_name="mse"`), but is worth
     remembering as a trap if anyone reaches for it again to try to fix
     this.

None of these is "the label is fundamentally unrecoverable" — they're
all addressable, individually testable levers. That said: **this is a
hypothesis, not yet a confirmed diagnosis** — see the recommended first
step below before committing real effort to any specific fix.

## Options

### A. Fix Stage 1's count regression directly

Keep the current architecture and label (`stream_count`), address the
amplitude bias mechanistically:

- **Distributional/count-aware loss** instead of plain MSE: a Poisson or
  Negative-Binomial negative-log-likelihood is the textbook "correct"
  loss for count data (naturally handles the fact that variance scales
  with the mean, and doesn't have MSE's flat quadratic penalty that
  encourages hedging toward the mean under skew). Real, moderate-effort
  addition to `models/losses.py`.
- **Per-sample weighted MSE**, fixing the batch-wide-normalization bug
  rather than routing around it — would let the "emphasize bright
  pixels" idea actually work as originally intended.
- **`GroupNorm`/`InstanceNorm` instead of `BatchNorm`** in `UNet`'s
  `ConvBlock` — standard fix for small-batch training, and a plausible
  contributor here specifically.
- **More capacity and/or more steps.** Directly testable, already
  partially validated (450 → this run's ~1200 samples-worth of steps
  measurably improved Dice) — worth knowing whether amplitude keeps
  improving on the same slope, or plateaus (which would point back at
  (1)-(4) rather than "just needs more training").
- **Post-hoc calibration**: fit a simple monotonic rescaling (even just a
  per-channel scalar, or an isotonic regression) from predicted → true
  peak values on a held-out eval set. Cheap, doesn't touch training at
  all, and directly answers "is the model's *ranking* of pixel brightness
  basically right, just compressed in scale?" — which is exactly the
  diagnostic recommended below.

**Pros:** keeps a physically interpretable output (a real star-count map,
independently useful for e.g. later richness estimation, not just
detection). Matches the original pivot's rationale (`PLAN.md` §6.3) for
why count was chosen over binary/surface-brightness in the first place.
**Cons:** count regression under extreme skew is a genuinely hard
optimization problem; every lever above is a real engineering
investment with an uncertain payoff, and detection (the actual goal) may
not need this precision at all.

### B. Retarget Stage 1 (or make it the only stage) to a detection label

The core insight: **`models/losses.py` already has `DiceLoss`, `FocalLoss`,
`TverskyLoss`, and `BCEWithLogitsLoss` — implemented, tested, and unused
since the 2026-09-09 pivot to count labels.** These are the standard tools
for exactly the "rare positive pixels among mostly-negative background"
problem this label already has, and they don't suffer count-regression's
regression-to-the-mean pathology, because the target is bounded and the
loss doesn't scale with the target's magnitude.

**Important trap to avoid:** the *old* pre-pivot binary/density labels
(`rasterize.py`) are **purely geometric** — "is this pixel within the
stream's true track," computed from `phi1`/`phi2` alone, identical
regardless of which matched filter or distance produced the channel. That
throws away exactly the property the count-label pivot was designed to
give (§6.3): a "bad" decoy filter should register near-zero signal even
at pixels a real stream crosses, because it didn't actually *select* real
stream stars there. Reverting to `rasterize.py` verbatim would silently
undo that.

**The fix that preserves it, with no new label-generation mechanism
needed:** derive the detection target directly from `stream_raw` — the
per-channel, filter-and-distance-specific raw star count `injector.py`
already computes as the intermediate step before it becomes the
`stream_count` label. Two variants:

- **Hard threshold:** `label = stream_raw > threshold` (e.g. `> 0.5`,
  i.e. "at least one real star selected here"). Simplest; pairs with
  `BCEWithLogitsLoss`/`DiceLoss`/`FocalLoss`.
- **Soft/saturating:** `label = 1 - exp(-stream_raw / tau)` for a tunable
  scale `tau` — bounded in `[0, 1)`, monotonic in count, so a
  denser/brighter true detection still gets a higher target than a
  marginal one, without the unbounded, heavy-tailed scale that's driving
  Stage 1's current difficulty. A genuine middle ground between "exact
  count" and "hard binary."

Either way: same input pipeline, same `UNet` (just `head="sigmoid"`
instead of `"softplus"`), a one-line change to which `models.losses` loss
gets used, and a small addition to `injector.py` (or a thin dataset-level
transform) to threshold/saturate `stream_raw` before it becomes the
label. This is the **lowest-effort, lowest-risk option to actually test**
— every piece except the threshold/saturation step already exists and is
tested.

**Pros:** targets the real deliverable directly (a probability map, no
extra interpretation step); sidesteps the amplitude problem entirely
rather than fighting it; reuses fully-implemented, previously-validated
loss machinery. **Cons:** loses the literal count as an output — if a
downstream use genuinely wants "how many stars," this doesn't give it
(though see the joint-head option below); a soft/saturating label
reintroduces one new small design choice (`tau`) to tune.

### C. Test whether Stage 1's *current* output is already good enough as Stage 2 input

The original plan was always two networks: Stage 1 cleans, Stage 2 (a
separate, not-yet-built network) takes Stage 1's output and predicts
per-pixel probability. **Worth checking before assuming Stage 1 needs
fixing at all:** if Stage 2 is itself a learned model, it can plausibly
learn to work with Stage 1's *current*, amplitude-compressed-but-
spatially-correct output — the same way any real ML pipeline's
intermediate representations don't need to be independently "correct" as
long as they carry the discriminating signal downstream, which Dice=0.857
suggests they do. This wouldn't need any Stage 1 change at all; it would
mean building Stage 2 next (still nothing implemented there:
`datasets/`, `models/`, `training/` for it don't exist) and seeing
directly whether the amplitude gap actually hurts detection performance,
instead of inferring that it must.

**Pros:** doesn't presuppose an answer; cheapest way to find out if this
is a real blocker at all. **Cons:** takes as long as actually building
Stage 2 to get an answer; if the amplitude gap *does* turn out to matter,
this is time spent before finding that out rather than before.

### D. Single network, two heads (count regression + detection probability)

Same shared encoder/decoder backbone, two separate output heads trained
jointly: one `softplus` head against `stream_count` (MSE or the
distributional loss from option A), one `sigmoid` head against the
option-B detection target (Dice/BCE/Focal), combined into one loss
(a weighted sum). Multi-task learning like this often improves *both*
heads relative to training either alone, since the shared backbone has to
learn features useful for both.

**Pros:** gets a genuinely useful probability map *and* keeps a count
estimate, without needing a second full network (Stage 2) at all — this
could plausibly be the entire pipeline, not just Stage 1. **Cons:** a real
architecture change to `UNet` (a second head, a combined loss), more
moving parts to tune (loss weighting between the two heads), and still
inherits whatever makes count regression hard for the count head
specifically (though that head no longer being the *only* signal reaching
the shared backbone might itself help, per the multi-task literature).

## Recommendation

1. **Cheap first step, before committing to any of the above:** a
   calibration check on the *current* trained model — scatter the
   predicted vs. true value at every valid pixel (or at least every
   pixel above some floor) across a few eval samples, on a log-log plot.
   If it's close to a straight line with slope < 1 through the origin
   (a *scale* problem), that's strong evidence for "needs more
   training/capacity" (option A's cheaper levers, or even just a
   post-hoc rescale) over "the model has learned something structurally
   wrong." If the relationship is noisy/non-monotonic instead, that
   points more toward the loss-shape issues (regression-to-the-mean,
   `log1p` compression) and favors option B or D. Costs nothing but a
   plotting pass over an existing checkpoint.
2. **In parallel, or next depending on (1)'s answer:** prototype **option
   B** (the threshold/saturate-`stream_raw` detection label) — it's the
   least invasive, reuses the most already-validated code, and most
   directly answers the question this document opened with ("just say
   where it's likely to have a stream"). If it works well, it may make
   the count-amplitude problem moot for the actual deliverable, and
   option D becomes the natural way to *also* keep a count estimate later
   without having solved option A's harder problem first.
3. Treat **option A**'s deeper fixes (distributional loss, GroupNorm,
   per-sample weighting) as real but lower-priority — worth doing if
   something downstream genuinely needs accurate counts, not required to
   get to a working detector.
4. **Option C** is really a variant of "just build Stage 2 and see" —
   reasonable, but only after (1)/(2) give some signal about whether it's
   worth the time; building a whole second network is the most expensive
   way to answer "does the amplitude gap matter."

## Open questions for discussion

- Does anything downstream of detection actually need the literal count
  (e.g. richness/mass estimation for a confirmed candidate), or is
  detection genuinely the whole Stage-1-relevant deliverable? This
  matters a lot for whether option A's effort is worth it at all versus
  B/D.
- If option B: hard threshold or soft/saturating label — and if soft,
  what `tau`? (Could be decided empirically once real predictions are in
  hand from a first B prototype.)
- Is there an appetite for the bigger step of just building Stage 2 now
  (option C) rather than iterating further on Stage 1 first?
