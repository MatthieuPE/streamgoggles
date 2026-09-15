# Notebooks

Two executable notebooks in `notebooks/` walk through the pipeline
interactively. Both are built the same way: plain, editable config
dictionaries at the top, then the pipeline step by step, then the same
result built again with the minimal high-level call. Both were executed
for real (against the project's `streamml` conda environment) as part of
building them — not just written and assumed to work.

## `create_data.ipynb`

Builds **one** training sample two ways: step by step (study region →
cached background per named matched filter → realized stream → placed in
the footprint → injected through the survey model → windowed → per-channel
select/combine/finalize/crop, including the label) and then the same thing
via the single `StreamInjector.inject_single_stream` call. Also shows the
shape-compatible background-only case (`inject_background_only`). The best
starting point for understanding exactly what one `Sample` contains and
how it was built — see {doc}`data_generation` for the module-level version
of the same story.

Runs at `nside=512` (the standard resolution for stream search — see
`PLAN.md` §6.12), with `label_policy="stream_detection"`: its §2.8
walkthrough shows both the raw per-channel stream-only count and the
hard-thresholded `{0, 1}` detection label built from it, side by side, so
the mechanism stays visible even though the label actually used is the
thresholded one.

Its final section (§3.2) goes one step further than the raw `Sample`: it
fits a `RobustNormalizer` and runs `StreamMapTransform` on
`sample_minimal`, then plots the raw combined map next to the *actual*
model input side by side — masked (see {doc}`datasets_and_models`'s note
on why raw vs. normalized invalid pixels are easy to conflate visually) so
it's obvious which pixels are real. This is the notebook to check "what
does the model actually see" against — separate from `train_model.ipynb`,
which cares about training dynamics, not input inspection.

## `train_model.ipynb`

Wires the rest of the pipeline together and runs one real training loop
end to end: `StreamMapDataset` (training + eval mode) →
`StreamMapTransform`/`RobustNormalizer` → `UNet` → `PlainTrainer` →
`evaluation/` (`evaluate_on_grid`, `build_baseline`,
`plot_recovery_vs_parameter`). Small (small images, a small model, no GPU)
but no longer trivially so — its current form is the `label_policy=
"stream_detection"` pivot (`PLAN.md` §6.12), which itself replaced an
earlier `label_policy="stream_count"` version that needed real tuning to
get anywhere (log1p-space training target, data-informed bias init, a
larger step budget — see the decision log for that history) and, even
tuned, persistently under-recovered peak amplitude while still localizing
streams well. Retargeting at a bounded detection label sidesteps that
problem entirely:

- **`nside=512`** (the standard resolution for stream search), with
  `pixel_scale_deg` never set by hand, and **`count_threshold=1.0`**
  (`StreamInjector`'s own default, PLAN.md §6.13) — calibrated
  empirically, not guessed: sweeping both the threshold and stream
  richness found a genuine trade-off (too high a threshold leaves faint
  streams with an entirely empty label; too low, and the "decoy" filter's
  own positive-pixel count becomes a substantial fraction of the real
  filter's at the bright end), and 1.0 is the lowest value that keeps
  every richness point in this project's working range non-empty. See
  `create_data.ipynb`'s "Calibrating count_threshold empirically" section
  (§4) for the full sweep and code. `richness = [31.0, 32.0, 33.0, 34.0]`
  (surface brightness, DISCRETE, an eval-grid point per value) — the
  bright end (SB 30) is dropped from this scan since it's both the
  easiest case for the network and the worst for decoy contamination, so
  it has the least to teach; whether the faintest point (SB 34) is
  recovered well is exactly what §7's per-richness Dice curve checks.
- **`head="sigmoid"` + `loss_name="dice"`**, not `"softplus"` + `"mse"` in
  `log1p` space — no longer needed, since a bounded `{0, 1}` target has no
  large dynamic range to compress. The **data-informed bias
  initialization** technique is kept, adapted rather than dropped: the
  head's bias starts at the **logit of the per-channel positive-pixel
  fraction** instead of the log1p-count mean, so the model begins
  predicting close to the empirical class prior everywhere rather than a
  default ~0.5.

It plots a training/validation loss curve, one prediction compared against
its true detection label *and* the masked residual between them (fixed to
`[-1, 1]`, the range a bounded target actually spans), and a completeness
(Dice) curve across the 4-point richness scan against the k·σ baseline —
1200 training samples (`epochs=40`, `steps_per_epoch=30`), 4x the budget
a single-richness run used. **Read honestly, not just as a success
number** — real results, from §7's actual output:

| surface_brightness | Dice | IoU | baseline Dice |
|---|---|---|---|
| 31 | 0.786 | 0.648 | 0.590 |
| 32 | 0.611 | 0.440 | 0.262 |
| 33 | 0.519 | 0.351 | 0.059 |
| 34 | 0.039 | 0.020 | 0.081 |

**A real, unplanned finding along the way:** this result is not
bit-for-bit reproducible across separate runs of the same notebook, same
code, same nominal seed — an earlier 1200-sample run (identical config)
gave SB 34 a hard `0.000`; this one gives `0.039`. Every RNG here is
explicitly seeded, but PyTorch's CPU convolution backward pass isn't
bit-deterministic across process runs by default (thread-scheduling-
dependent reduction order) — worth knowing before reading any one number
in this table too literally.

**§8 gives the deeper reason not to over-read a single per-richness
number, independent of the run-to-run variation above** — because every
eval-grid point is exactly *one* fixed realization (one window, one
orientation, one noise draw — see {doc}`datasets_and_models`'s "Eval grid:
one fixed realization per point" section), a Dice value at any one
richness is a measurement on a sample size of one, not an average over
what a stream at that richness typically looks like. §8 tests this
directly: running the *same trained model* (no retraining) against 8
independent SB 34 realizations finds the true behavior is **bimodal** — 5
of 8 confidently detect the stream (predicted probability 0.87-1.0 at the
true location, real non-zero Dice), 3 of 8 never activate at all
(probability stuck at 0.008-0.011, flat background level, Dice=0.0),
nothing in between. SB 33, checked the same way, detects confidently
(>=0.998) on *every* realization tried. So SB 34 sits right at a genuine
sensitivity edge for this model/budget — consistent with this run's own
single eval point (0.039) landing close to that boundary rather than
clearly in either regime. The training-budget question this run set out
to answer is still genuinely open: neither a single eval point nor a
comparison of two single eval points across runs can tell "budget doesn't
help" apart from "budget shifts the success fraction across realizations,
plus run-to-run training noise on top." Scaling this up further (an eval
grid with replicate realizations per point — not built yet, see
{doc}`datasets_and_models` — more steps/epochs, a real GPU device, and
eventually the `training/hyrax_runner.py` orchestration layer) is the
natural next step.
