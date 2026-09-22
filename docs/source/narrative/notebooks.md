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
  (`StreamInjector`'s own default, PLAN.md §6.13): the lowest threshold that
  keeps a non-empty label for every surface brightness this project works
  with, up to 35. See `create_data.ipynb`'s "Calibrating count_threshold
  empirically" section (§4) for the sweep.
- **Two input channels**: the isochrone matched filter, and a fixed decoy
  colour-magnitude box (colour 1.2-1.5, magnitudes 18-24.5) redder than the
  isochrone locus. The decoy selects no stream star at any surface brightness,
  so it is a clean negative control: plenty of stars, none of them stream-like.
- **The configuration the experiments selected** ({doc}`../experiments/index`):
  `head="sigmoid"`, batch Dice loss, batch 8, `background_fraction=0.05`
  ({doc}`../experiments/loss_selection`); training on surface brightness 32,
  33, 33.5, 34 and 34.5 with 19200 windows (`epochs=40`,
  `steps_per_epoch=480`), U-Net depth 2 and base width 12
  ({doc}`../experiments/hyperparameters`). The head's bias starts at the logit
  of the per-channel positive-pixel fraction, so the model begins at the
  class prior rather than a default ~0.5.
- **Magnitude range only in `clipping_cfg`** (16-24.5 in g and r). `cuts_cfg`
  is empty: it is for quality cuts that are not magnitude ranges (SNR,
  star/galaxy separation).

§5-§8 train one model and look at it window by window: the loss curves, one
prediction against its label, a per-surface-brightness Dice curve against the
k·σ baseline (§7, 5 realizations per point), and §8's check of SB 33 and 34 on
8 more realizations each. SB 33 responds on every draw, with a Dice between
0.50 and 0.69; SB 34 gets no response in these windows.

§9-§11 move from windows to the HEALPix map that stream searches actually
produce (see {doc}`training_and_evaluation`, "Footprint-level detection"):

- **§9** injects one stream into the full sky, tiles the area around it with
  overlapping windows, and stitches the output back to HEALPix, keeping the
  most-central window's value where tiles overlap.
- **§10** repeats this over 30 independent realizations for each of SB 30, 32,
  33, 33.5, 34, 34.5 and 35: a row-normalized confusion matrix, and the found
  fraction next to the fraction of background flagged, with the stream and on
  the same tiles without it. At 0.5, half the true pixels are found down to
  SB ≈ 33.0, and with the stream removed not one background pixel is flagged.
- **§11** turns the same realizations into completeness $C$, contamination $F$
  and contrast $C/F$ against surface brightness, one line per threshold.

§12 counts **streams detected** rather than pixels, with the criterion the
experiments use (20 flagged pixels within 1σ of the track, SNR ≥ 2 against
stream-shaped background bands), and reads it two ways: at threshold 0.5, and
at the threshold where the model flags 1e-3 of stream-free sky. It does so on
the training background and on an independently seeded one the model never
saw. Fraction of injected streams detected, 30 per surface brightness, unseen
background:

| Surface brightness | threshold 0.5 | 1e-3 of stream-free sky flagged |
|---|---|---|
| 33 | 97% | 97% |
| 33.5 | 27% | 40% |
| 34 | 17% | 17% |

Two lessons the notebook draws from it. At 0.5 this model is cautious, so its
detections there understate it; the matched false-alarm rate is the fair
reading. And on its own training background it flags no stream-free pixel at
all, so a model's cleanliness must be judged on a sky it never saw. One model
on 30 streams per point is a demonstration: the hyperparameter experiment
measures this configuration on six trainings × 300 streams per point (95%, 49%
and 16% at SB 33, 33.5 and 34 at a matched 1e-3), and those are the numbers to
quote.

The whole notebook runs in about half an hour on a laptop CPU, most of it the
19200-window training and the two footprint evaluations.
