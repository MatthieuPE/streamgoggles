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
but no longer trivially so — it went through two real rounds of tuning
(not guesswork; see `PLAN.md`'s decision log entries for this notebook):

- **Richness bracket, 30–33 mag/arcsec² surface brightness**, not lower:
  below SB 30 at this stream geometry the star count explodes (SB 28 →
  ~1.1M stars) into physically implausible territory.
- **Training target is `log1p(count)`, not raw counts**, via two small
  notebook-local classes (`Log1pLabelDataset` wrapping the training/
  validation label, `ExpM1Wrapper` inverting the trained model's output
  back to real counts for everything downstream that wants physical
  units). Raw-count losses at this brightness were numerically unstable
  (millions, non-monotonic) and a small model couldn't learn from them in
  any practical number of steps — the same reason astronomical magnitudes
  are logarithmic. Combined with a **data-informed bias initialization**
  (the output layer starts at the target's own scale, not near zero) and a
  substantially larger step budget (~450 optimizer steps, up from an
  initial 60 that showed no amplitude learning at all — confirmed
  empirically, not assumed).

It plots a training/validation loss curve (in `log1p(count)` space), one
prediction compared against its true label *and* the masked residual
between them, and a completeness (Dice) curve against the network's free
richness parameter, with the k·σ baseline overlaid for comparison. Scaling
this up further (bigger images, a real GPU device, and eventually the
`training/hyrax_runner.py` orchestration layer) is the natural next step
once a scientifically meaningful run is wanted — this is still a real but
modest training run, not a tuned model.
