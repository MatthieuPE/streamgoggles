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
- **`head="sigmoid"` + `loss_name="batch_dice"`, batch size 8,
  `background_fraction=0.05`**, chosen by the loss-selection experiment
  ({doc}`../experiments/loss_selection`). A bounded `{0, 1}` target needs no
  `log1p` compression. The **data-informed bias initialization** is kept,
  adapted: the head's bias starts at the **logit of the per-channel
  positive-pixel fraction**, so the model begins predicting close to the
  empirical class prior everywhere rather than a default ~0.5.
- **Magnitude range only in `clipping_cfg`** (16-25 in g and r). `cuts_cfg`
  is empty: it is for quality cuts that are not magnitude ranges (SNR,
  star/galaxy separation). The magnitude cuts it used to hold were
  redundant with the clipping and removed no star.

It plots a training/validation loss curve, one prediction compared against
its true detection label *and* the masked residual between them (fixed to
`[-1, 1]`, the range a bounded target actually spans), and a recovery
curve across the richness scan against the k·σ baseline — 1200 training
windows (`epochs=40`, `steps_per_epoch=30`), evaluated on a grid with
**5 independent realizations per richness** so each point is a mean and
spread rather than one arbitrary draw. Results from §7 (threshold 0.5):

| surface_brightness | nstars | Dice | IoU | baseline Dice |
|---|---|---|---|---|
| 31 | 33362 | 0.847 ± 0.007 | 0.735 | 0.634 |
| 32 | 13282 | 0.740 ± 0.072 | 0.591 | 0.301 |
| 33 |  5288 | 0.205 ± 0.129 | 0.118 | 0.132 |
| 34 |  2106 | 0.007 ± 0.015 | 0.003 | 0.038 |

- **The network clearly beats the trivial baseline at SB 31-32** (0.85 vs
  0.63, 0.74 vs 0.30).
- **SB 33 is the detection edge, and it is bimodal.** §8 re-runs the model on
  8 more realizations: about half get a confident response on the stream
  (peak probability 0.6-0.98), the others almost none (0.02-0.26). That is
  what §7's large standard deviation at SB 33 is made of. SB 34 gets no
  response at all (peak probability about 0.017), although it is inside the
  training range.

§9, §10 and §11 move from windows to the HEALPix map that stream searches
actually produce (see {doc}`training_and_evaluation`, "Footprint-level
detection"):

- **§9** injects one stream into the full sky, tiles the area around it
  with overlapping windows, and stitches the model's output back to
  HEALPix, keeping the most-central window's value where tiles overlap.
  It plots three `skyproj` maps with the same framing: input
  (stream + background), detection label, and prediction. The prediction
  follows the track with no scattered false alarms.
- **§10** repeats this over 30 independent realizations for each surface
  brightness from 30 to 36: the row-normalized confusion matrix per SB, and
  the found fraction next to the fraction of background flagged, with the
  stream and on the same tiles without it. At 0.5, half the true pixels are
  found down to SB ≈ 32.4; with the stream removed, not a single background
  pixel is flagged in any realization.
- **§11** turns the same realizations into completeness $C = S_s/S_t$,
  contamination $F = B_s/B_t$ and contrast $C/F$ against SB, one line per
  threshold (0.1 to 0.999), to show how far the threshold moves the maps.
  At 0.5: $C$ = 0.92 / 0.81 / 0.13 at SB 30 / 32 / 33, contamination about
  0.1%, contrast 1200 / 655 / 292.

The whole notebook runs in about 3.5 minutes on a laptop CPU.
