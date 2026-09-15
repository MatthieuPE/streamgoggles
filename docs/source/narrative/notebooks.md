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
  `pixel_scale_deg` never set by hand. `richness = [30.0]` (surface
  brightness, a single fixed point rather than the earlier 30–33 scan):
  at this finer native pixel scale, per-pixel star counts drop a lot for
  the same total population, and `count_threshold=10`'s detection label
  needs enough stars in *some* pixel to ever produce a positive label at
  all — verified for real that 30 reliably does at this geometry and 31–33
  do not (see §1's config cell for the exact numbers).
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
(Dice) evaluation against the k·σ baseline. **Worth reading honestly, not
just as a success number**: at this richness/resolution the trivial
baseline scores comparably well (~0.94 Dice, vs. ~0.12 for the earlier
`stream_count` setup) — a real stream pixel is a large, sharp, easy-to-
threshold excess here, and the baseline is handed the "good" channel
directly, which the network has to identify on its own. The network's real
advantage over a fixed threshold — fainter richness, rejecting the "decoy"
channel without being told which is which — isn't demonstrated by this
single-point smoke-scale eval grid yet. Scaling this up further (a richer
eval grid, more steps/epochs, a real GPU device, and eventually the
`training/hyrax_runner.py` orchestration layer) is the natural next step.
