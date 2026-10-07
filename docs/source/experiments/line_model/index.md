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
| {doc}`long_and_fits` | four long line models per fold; the leads that held, fitted: a distance and a curved track each | done (2026-10-02, branch `line-model-long-fits`) |
| {doc}`des2018_known` | the DES 2018 streams where DES 2018 found them, each method side by side, along their tracks and without; each against its simulated copies at its strength; why the misses; training at the streams' strength, the label from S/N 5, both, longer training, more short streams, four quick S/N-5 models per fold, segment lines, the dwarfs' mask, and the selection's depth | done (2026-10-07, branches `des2018-recovery`, `des2018-s5-strength`, `des2018-s5-long`, `des2018-short-streams`, `des2018-s5-seeds`, `segment-lines`, `segment-lines-x4`, `des2018-input`) |

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
5. **The robust number is eight of the fourteen DES 2018 streams**, found
   by the two line searches together at the false-alarm rate of one: ATLAS,
   Chenab, Elqui, Indus, Jhelum, Phoenix, Tucana III, Willka Yaku — with two
   or four quick models per fold, or four long ones; one long model per fold
   found nine, partly by a good draw. Four long models are the most
   sensitive on the copies (half the near copies at input S/N 7.9). None
   finds Wambelong, Turbio, Molonglo, Ravi or Aliqa Uma; two of them are not
   in the data ({doc}`line_followup`, {doc}`long_and_fits`).
6. **Beyond DES 2018, three tracks hold**, each with a main sequence in its
   Hess difference ({doc}`line_followup`), and fitted from their stars
   ({doc}`long_and_fits`): Tucana III continuing 8° east of its DES 2018
   track at its own distance (m−M 16.7); NGC 1261's stream on its proposed
   track, at m−M 15.5; and a stream along Leiptr's direction at m−M 15.1,
   farther than Leiptr's catalogued 6-9 kpc. The other leads were the
   footprint's edge, depth changes, lines crossing a known stream — and
   Indus.
7. **Where DES 2018 found them, the line network finds eight of the fourteen
   streams along their tracks and six to seven without** (nine and eight with
   the matched filter's line sums) — and finds them as often as its simulated
   copies at the same strength predict: the simulations predict how the
   model does on the real sky. Of the twelve streams in our data, only Aliqa
   Uma (under the mask of Fornax's outskirts) and Wambelong (S/N 5.2) are
   found by no line search. The streams are 0.5-2 mag fainter in our data
   than their copies at Table 1's surface brightness, but training at their
   strength makes the network less sensitive, not more; teaching a line only
   where the stream reaches S/N 5 in the window (not 2) gives the best line
   network so far: 9 along the tracks, 7 without, with two quick models per
   fold; at the streams' strength it trades far streams for near ones, and
   trained long (19,200 windows) it buys nothing on the DES 2018 streams and
   loses the short streams again; more short streams in training make it
   worse at them. **Four quick S/N-5 models per fold are the best line
   network so far: 10 of the 14 DES 2018 streams along their tracks, 8
   without, 9 in the sky search with the matched filter's line sums**
   (Wambelong found, by a small margin). **Segment lines** (the lines of
   nine 5.5° sub-windows beside the window's) help the matched filter's
   line search (8 streams on the sky against 6), not the network: four
   segment models find what four window-line models find, and either search
   finds 10. The model work has reached a plateau; what is left is the input.
   **Masking the bright dwarfs to 2° at most** — their stars end at 1.5° —
   frees Aliqa Uma from Fornax's 4° mask at no cost: **11 of the 14 along
   their tracks**. A shallower selection would trade streams, not add them
   ({doc}`des2018_known`).

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
long_and_fits
des2018_known
```
