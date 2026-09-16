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
`[-1, 1]`, the range a bounded target actually spans), and a recovery
curve across the richness scan against the k·σ baseline — 1200 training
samples (`epochs=40`, `steps_per_epoch=30`), evaluated on a grid with
**5 independent realizations per richness** so each point is a mean and
spread rather than one arbitrary draw. Real results from §7:

| surface_brightness | nstars | Dice | IoU | baseline Dice |
|---|---|---|---|---|
| 31 | 33362 | 0.773 ± 0.012 | 0.630 | 0.634 |
| 32 | 13282 | 0.673 ± 0.051 | 0.509 | 0.301 |
| 33 |  5288 | 0.470 ± 0.068 | 0.309 | 0.132 |
| 34 |  2106 | 0.087 ± 0.088 | 0.047 | 0.038 |

Two things the replicates make visible that a single realization per point
could not:

- **Where the network actually earns its keep.** It beats the trivial
  baseline decisively at SB 32 (0.673 vs 0.301) and SB 33 (0.470 vs
  0.132). At SB 31 the two are *comparable* (0.773 vs 0.634 here; an
  earlier run of the same configuration had the network slightly behind at
  0.621, within the seed-to-seed spread) — read the bright end as "no
  reliable advantage either way". That is where a real stream is a large,
  sharp excess a plain threshold finds easily, and where the baseline is
  additionally handed the "good" channel directly, which the network has
  to identify for itself. The faint end is where learning clearly pays.
- **Which numbers are measurements and which are noise.** SB 34's standard
  deviation (0.088) is as large as its mean (0.087): that richness isn't
  "detected at 0.087", it's bimodal — confidently detected on some
  realizations, entirely missed on others. §8 shows the per-draw detail
  (peak predicted probability at the true location is either ~1.0 or
  ~0.01, nothing between), while SB 33 detects confidently on every
  realization tried. Earlier versions of this notebook reported a bare
  `0.000` at SB 34 from a single realization, which read as a definitive
  failure and wasn't.

Scaling up further (re-running the training-budget comparison now that
replicates can measure a *success fraction*, more steps/epochs, a real GPU
device, and eventually the `training/hyrax_runner.py` orchestration layer)
is the natural next step.

§9 and §10 move from windows to the HEALPix map that stream searches
actually produce (see {doc}`training_and_evaluation`, "Footprint-level
detection"):

- **§9** injects one stream into the full sky, tiles the area around it
  with overlapping windows, and stitches the model's output back to
  HEALPix, keeping the most-central window's value where tiles overlap.
  It plots three `skyproj` maps with the same framing: input
  (stream + background), detection label, and prediction.
- **§10** repeats this over 30 independent realizations for each surface
  brightness from 30 to 36. It plots the row-normalized confusion matrix
  per SB. It also plots the found fraction against SB, next to the
  fraction of background pixels flagged as stream, with the stream and
  on the same tiles without it. Half the true pixels are found down to
  SB ≈ 33.2, but ~0.9% of background is flagged even with no stream, which
  limits precision at the faint end (§10.1). About 75s of the notebook's
  ~190s total.
