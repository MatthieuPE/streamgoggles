# streamgoggles

`streamgoggles` is a machine-learning pipeline for detecting Milky Way
stellar streams in wide-field photometric survey data. It builds on
[`streamobs`](https://github.com) (synthetic survey observation and stream
injection) and [`ugali`](https://github.com/DarkEnergySurvey/ugali)
(isochrones and stellar population modeling) to generate realistic,
labeled training data, and on [`hyrax`](https://github.com) for eventual
production training orchestration.

## What the pipeline does

Given a footprint on the sky, `streamgoggles`:

1. Builds a synthetic (or real) **background** star catalog restricted to
   that footprint, applies photometric cuts, and caches per-filter,
   per-distance count maps.
2. **Injects** a synthetic stellar stream — realized from an isochrone
   population, placed along a random great circle, run through the survey's
   noise/detection model — and combines it with the cached background.
3. Runs one or more **matched filters** (a "good" isochrone-consistent
   filter, plus deliberately "bad" decoy filters) over the combined field at
   several trial distances, producing one input channel per
   `(filter, distance)` pair.
4. Trains a **U-Net** to recover the stream-only star count per channel —
   a background-cleaning (image regression) task — and evaluates it against
   a trivial threshold baseline.

See {doc}`narrative/overview` for the full picture, {doc}`narrative/tutorial`
for a task-oriented walkthrough (fix parameters, size a dataset, train, then
detect streams across a real footprint), {doc}`narrative/ml_concepts` if
terms like "U-Net" or "Dice coefficient" need a primer first, or jump
straight to the {doc}`api` reference.

```{toctree}
:maxdepth: 2
:caption: Guide

narrative/overview
narrative/ml_concepts
narrative/data_generation
narrative/datasets_and_models
narrative/training_and_evaluation
narrative/notebooks
narrative/tutorial
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Building and viewing this site locally

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m sphinx -b html docs/source docs/_build/html
```

(the two environment variables are the same healpy/torch OpenMP workaround
`tests/conftest.py` needs — `sphinx.ext.autodoc` imports every module to
read its docstrings, which loads both libraries in one process.) The build
needs this project's real runtime environment (`torch`, `healpy`, `ugali`,
`streamobs`), not just the `docs` dependency group, for the same reason.

The result is a fully static, self-contained site — no external fonts,
scripts, or CDN calls — so it works with no internet connection. Two ways
to view it:

- Open `docs/_build/html/index.html` directly in a browser. Works offline;
  the one caveat is that some browsers block the search box's JS-driven
  index fetch on a bare `file://` page (search may not work, everything
  else does).
- Or serve it locally for full functionality including search:
  `cd docs/_build/html && python -m http.server 8000`, then open
  `http://localhost:8000` — still entirely offline (`localhost` only).

`docs/_build/` is gitignored; rebuilding after an edit just overwrites it.
