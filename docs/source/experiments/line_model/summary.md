# Summary: the DES 2018 streams found by networks trained on streamobs

The question of this project: **can streamobs simulations train a model that
finds stellar streams in matched-filter maps, in place of the by-eye
inspection** that found the fourteen DES 2018 streams (Shipp et al. 2018)?
The benchmark is those fourteen streams in the real DES Y6 data: twelve are in
our data, Ravi and Molonglo are not ({doc}`../real_des/des2018_reproduction`).
This page gathers what was tried, what worked and what did not, and the best
result so far; the details are in the pages it links.

## The best result

**A line network trained only on streamobs simulations finds, without being
told where to look, 8 of the 14 DES 2018 streams; with the matched filter's
own line search beside it, 9 — at one false line per hundred stream-free
windows. Along their known tracks it finds 11: every stream in our data but
Indus, which it finds without its track.** The per-pixel network of the first
training found 5 along their tracks.

The model and the search that give it:

- **inputs**: the DES Y6 matched-filter counts at the queried distance and
  ±0.5 mag, and a colour box off the isochrone, each turned into an S/N map —
  its excess over a local plane, over the plane's square root;
- **the line network** ({doc}`../../narrative/line_model`): a U-Net's feature
  maps and the inputs summed along every line through an 11° window (a Hough
  transform), a small network over the lines, one probability per line;
- **training**: streamobs streams (32-34.5 mag/arcsec², widths 0.1-1.5°,
  lengths 4-30°, m−M 15-19, ages 9-13.5 Gyr) injected into the stream-free
  real sky of one fold; **the label**: the lines of the stream's band where it
  runs at least 2° and stands at S/N 5 or more in the window; four quick
  models per fold (4,800 windows each), averaged (`hough/band2s5 residual
  x4`);
- **the search**: every 11° window of the DES footprint, half overlapping,
  scored by the models of the other fold; a line counts when it scores above
  the level the best line of 1% of stream-free windows reaches; the bright
  dwarfs masked to 2° at most, the globular clusters to 2° (`--mask tight`).

| | along the known track | without the track | in the sky search (beyond random tracks) |
|---|---|---|---|
| **line network** | **11** | **8** | **8** |
| matched filter's line sums | 9 | 6 | 6 |
| both, each at half its rate | 10 | 8 | **9** |
| per-pixel network, first training | 5 | — | — |

*Of the fourteen DES 2018 streams. Along the track: the stream's lines
against the same lines in 600 stream-free windows, found at 1 in 201; without
it: a line along the track above the 1% level; the sky search: the lines the
whole-sky search reports, matched to the track, more than random tracks of
its shape get but in 5% of places. {doc}`des2018_known`.*

| stream | S/N in our data | m−M | line network, along the track | line network, without it | sky search, both searches |
|---|---|---|---|---|---|
| ATLAS | 23.1 | 16.8 | ✓ | ✓ | ✓ |
| Elqui | 15.7 | 18.5 | ✓ | ✓ | ✓ |
| Phoenix | 12.9 | 16.4 | ✓ | ✓ | ✓ |
| Jhelum | 12.0 | 15.6 | ✓ | ✓ | ✓ |
| Chenab | 11.7 | 18.0 | ✓ | ✓ | ✓ |
| Tucana III | 11.1 | 17.0 | ✓ | ✓ | ✓ |
| Indus | 11.0 | 16.1 | — (*p* = 0.012) | ✓ | ✓ |
| Willka Yaku | 8.7 | 17.7 | ✓ | — | ✓ (matched filter) |
| Aliqa Uma | 7.9 | 17.3 | ✓ | — | — |
| Wambelong | 5.2 | 15.9 | ✓ | ✓ (small margin) | ✓ |
| Turbio | 5.1 | 16.1 | ✓ | — | — |
| Turranburra | 3.9 | 17.2 | ✓ | — | — |
| Molonglo | 1.1 | 16.8 | — | — | — |
| Ravi | −0.2 | 16.8 | — | — | — |

*S/N: the stream's core in our matched filter, from the density profile
across it ({doc}`../real_des/des2018_reproduction`).*

### What the search reports, distance by distance

```{image} ../figures/line_model/des2018_sky.gif
:alt: The matched filter over the DES footprint at each queried distance, with the lines the sky search finds, coloured by what they lie along, and the DES 2018 tracks
:width: 100%
```

*The matched filter over the DES footprint, one frame per queried distance
(DM = m−M, 15 to 19): its counts over their smooth local background,
smoothed with a 0.4° Gaussian and shown −8% to +8%. Dashed yellow: the DES
2018 tracks, bold and named at their own distance. Lines: what the sky search
reports at that distance with both searches at half their rate (solid, the
line network; dotted, the matched filter's line sums), coloured by what they
lie along. Top left, the DES 2018 streams: green, a line along the track at
this distance; black, at another distance only; red, at none. `run.py`,
`des2018_gif`; the frame at m−M 17 alone: `des2018_sky_17.png`.*

The search reports 556 lines over the nine distances — each stream, each
Magellanic periphery, each edge, cut by the half-overlapping windows into
several:

| the lines lie along | line network | matched-filter lines | all |
|---|---|---|---|
| a DES 2018 track | 92 | 70 | 162 (29%) |
| another stream galstreams traces (Tucana III's and Indus's extensions, NGC 1261's, C-7, M2) | 7 | 14 | 21 (4%) |
| the Magellanic Clouds' outskirts or a bright dwarf | 62 | 205 | 267 (48%) |
| nothing known: false alarm or candidate | 22 | 84 | 106 (19%) |

**Half the unexplained lines lie along one straight edge**: 50 of the 106
run at Dec −40.5 to −41 between RA 306° and 344°, parallel to the edge where
our maps' valid sky stops at Dec −40, a median 1.3° inside it — the search's
largest unidentified track, 23° long. The other 56 are spread over the
footprint; joined into tracks, 18 unidentified tracks hold two segments or
more. By
construction, 1% of stream-free windows hold a line at each search's level;
the leads that held up when examined one by one — Tucana III continuing
east, NGC 1261's stream, a stream along Leiptr's direction — are in
{doc}`line_followup` and {doc}`long_and_fits`.

## What worked, and what did not

| step | what it changed on the DES 2018 streams | verdict |
|---|---|---|
| per-pixel U-Net on the real sky ({doc}`../real_des/results`) | 5 of 14 along their tracks | the baseline |
| per-pixel levers: depth, cross-entropy, band label, fainter training, logits summed ({doc}`../real_des/labels_normalization`) | none moves its limit toward the matched filter's (S/N 9-15 against 5) | ✗ |
| a network that answers with lines (Hough), on per-window normalized inputs ({doc}`window_level`) | worse than the matched filter's own line search | ✗ |
| the same on matched-filter S/N inputs | half the near copies found blind at S/N 7.9 | ✓ |
| the whole sky searched out of fold, bright dwarfs and clusters masked, the two line searches combined ({doc}`line_sky`) | 8 of 14 on the sky | ✓ |
| the band label down to 2° (not 4°) | the short, bright streams found (Tucana III) | ✓ |
| longer training (19,200 windows; {doc}`line_followup`, {doc}`long_and_fits`) | the most sensitive on the copies, but loses the short streams; 8 on the sky | ✗ |
| the real streams against their copies ({doc}`des2018_known`) | found as often as their copies at the same strength predict: the simulations predict the real sky | ✓ (validation) |
| training at the real streams' strength (32.5-35.5) | less sensitive: 8 → 6 along the tracks | ✗ |
| **the band label from S/N 5** (lines only where they show) | 9 along the tracks, 7 without (two models per fold) | ✓ |
| the S/N-5 label at the streams' strength | near streams gained, far ones lost | ✗ |
| the S/N-5 label, longer training | nothing gained; the short streams lost again | ✗ |
| more short streams in training (log-uniform lengths) | worse at short streams | ✗ |
| **four quick S/N-5 models per fold** | 10 along the tracks, 8 without, 9 on the sky with the matched filter | ✓ |
| segment lines (sub-windows' lines too) | no better for the network; the matched filter's line search 6 → 8 on the sky | ✓ for the matched filter only |
| **the bright dwarfs masked to 2° at most** (their stars end at 1.5°) | Aliqa Uma along its track: 11 | ✓ |
| a shallower selection (g, r ≤ 23.5) | trades streams: Turranburra and Aliqa Uma up, Willka Yaku, Wambelong, Jhelum, Turbio down | ✗ |

Three lessons hold across the rounds:

- **A line the network cannot see is worse than no example**: fainter
  training, more short streams and longer training all made the network
  softer; labelling only what stands out (S/N 5) made it sharper.
- **The simulations are a sound bench**: what helped on the copies helped on
  the real streams, and the real streams are found as often as their copies
  at the same strength. The copies at Table 1's surface brightness are 0.5-2
  mag stronger than the real streams in our data, so the bench must be read
  at the real streams' strength.
- **One or two streams between two pairs of seeds is the noise**: every
  claim above rests on four models per fold or on both pairs agreeing.

## What limits it now

- **The input, for the streams missed without their track**: Aliqa Uma
  (S/N 7.9), Turbio (5.1) and Turranburra (3.9) lie below what a search at 1%
  false lines per window reaches (half found at S/N 8-12 on the copies);
  Ravi and Molonglo are not in our data. No single faint limit raises them
  all.
- **The network's own sensitivity**: it has reached a plateau — label,
  training range, lengths, budget, segment lines — at about the matched
  filter's along a known track (half the near copies at S/N 5.8, the matched
  filter's band test 5.2), and short of it without the track for short,
  narrow streams (Willka Yaku is found blind by the matched filter's lines
  only).
- **False alarms**: the Magellanic peripheries and one straight edge of the
  valid sky (Dec −40) make most of the lines that are not streams.

## What could come next

- **Two selections side by side** — a deep and a shallow matched filter as
  separate input channels — for the streams that a shallower selection
  raises (Turranburra, Aliqa Uma) without losing those it lowers.
- **The valid sky's edges**: half the unexplained lines run along one edge;
  a search that discounts lines parallel to an edge would remove them.
- **Discovery**: the unexplained tracks, with a significance that counts the
  whole search (the look-elsewhere effect), once the recovery of the known
  streams is settled.

## Reproducing

From the repository root, in the `streamml` environment, with the skies of
{doc}`../real_des/index` built:

```bash
python scripts/experiments/line_model/run.py train --config "hough/band2s5 residual" --seed 42   # 43, 44, 45; --train-sky fold1; ~35 min each, four at once
python scripts/experiments/line_model/run.py evaluate --config "hough/band2s5 residual x4"       # --train-sky fold1; --sets fainter / "length scan"
python scripts/experiments/line_model/run.py line-sky --config "hough/band2s5 residual x4" --mask tight
python scripts/experiments/line_model/run.py des2018 --config "hough/band2s5 residual x4" --mask tight
```

The GIF and the line counts: `des2018_gif()` and `des2018_sky_lines()` in
`run.py`.
