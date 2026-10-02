# Networks that answer with lines, on the real DES sky

The per-pixel network of {doc}`../real_des/index` acts as a local detector:
it finds a stream where a few degrees of it stand out, and needs about twice
the input S/N that the matched filter needs along a known track
({doc}`../real_des/labels_normalization`). This section follows a different
model, which builds the integration along a stream into the network: it
answers with **lines** through a window — the U-Net's features and its input
summed along every line (a Hough transform), a small network over the lines —
instead of pixels. How it works, how each network is plugged and what goes in
and out are in the guide, {doc}`../../narrative/line_model`.

It is trained and scored on the same real DES Y6 sky, folds and simulated
DES 2018 copies as the per-pixel network, but it is scored per window rather
than per pixel, run over the whole sky out of fold, and its detections are
lines that can be joined into tracks: hence its own section, script, results
and figures.

| page | what it holds | status |
|---|---|---|
| {doc}`window_level` | the line model on the copies, without knowing their track and along it, against the per-pixel network and the matched filter's own line search; S/N inputs | done, quick models (2026-10-01) |
| {doc}`line_sky` | the line model over the whole DES sky with the real streams in it: on-sky maps of the detected lines against the DES 2018 tracks; the bright dwarfs masked; the two line searches combined; a catalogue of tracks; why short streams were missed (the label), and the 2° label | done (2026-10-01) |
| {doc}`line_followup` | more models and longer training of the 2° line model; the catalogue's leads looked at one by one (maps, distance profiles, Hess differences) | done (2026-10-02) |

## Conclusions so far

1. **A network that answers with lines finds near, faint streams best** —
   once its inputs are matched-filter S/N maps. Without knowing the track it
   finds half of the near copies at an input S/N of 7.9 (the per-pixel
   network needs 12.1, and its test knows the track; the matched filter's own
   line search 12.5), and Wambelong, which no other method finds
   ({doc}`window_level`).
2. **What reaches the lines must be an S/N**: on per-window normalized maps
   the same model is worse than the matched filter everywhere
   ({doc}`window_level`).
3. **On the real sky, the two line searches together find eight to nine of
   the fourteen DES 2018 streams beyond chance**, without knowing any track,
   at the false-alarm rate of one search — the line network the near, wide
   ones (Jhelum, Indus), the matched filter's line search, once the bright
   dwarfs and clusters are masked, the short, bright ones (Tucana III,
   Willka Yaku) — against five for the per-pixel network. A line counts for
   a stream only if it runs along its track ({doc}`line_sky`).
4. **The line network missed short streams because its label taught it
   to**: the band label's 4° minimum made every window holding a shorter
   clear stretch a negative example; with a 2° minimum it finds them
   ({doc}`line_sky`).
5. **Two models per fold are enough on the copies, but more help on the
   sky**: four quick models, or one long one, give the line network seven or
   eight streams by itself and the combination nine — all but Wambelong,
   Turbio, Molonglo, Ravi and Aliqa Uma, two of which are not in the data
   ({doc}`line_followup`).
6. **Beyond DES 2018, three tracks hold**: Leiptr, NGC 1261's stream and
   Tucana III's extension east of its DES 2018 track (at its distance) —
   each along its proposed track, with a main sequence in its Hess
   difference. The other leads were the footprint's edge, depth changes, or
   lines crossing a known stream ({doc}`line_followup`).

## Where things live

- Script: `scripts/experiments/line_model/run.py` (the line configurations,
  training, the window-level scoring, the sky search, the track catalogue,
  the leads, the figures); it shares the skies, labels, normalizations and
  simulated copies of `scripts/experiments/real_des/patches.py`.
- Models and results: `data/experiments/line_model/` (`fold0/`, `fold1/`,
  `line_sky/`, `line_sky/leads/`; git-ignored).
- Figures: `docs/source/experiments/figures/line_model/`; the guide's in
  `docs/source/narrative/figures/line_model/`.

## Reproducing

From the repository root, in the `streamml` environment, with the skies of
{doc}`../real_des/index` built:

```bash
python scripts/experiments/line_model/run.py train --config "hough/band2 residual" --seed 42   # ~20 min a model; --train-sky fold1
python scripts/experiments/line_model/run.py evaluate --config "hough/band2 residual"          # ~10 min; --sets fainter / "length scan"
python scripts/experiments/line_model/run.py figures                                          # the copies' figures and the guide's
python scripts/experiments/line_model/run.py line-sky --config "hough/band2 residual x4"       # the whole sky, ~10 min
python scripts/experiments/line_model/run.py leads                                            # the leads, one by one
```

```{toctree}
:maxdepth: 1
:hidden:

window_level
line_sky
line_followup
```
