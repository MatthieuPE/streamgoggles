# The real DES Y6 background

Everything the models have seen so far was a simulated sky. This page records
how the real one was obtained and turned into a background that simulated
streams can be injected into: where the stars come from, what was cut and why,
and what was masked so that the only stream in a training window is the one
put there.

Every number and figure below is produced by
`scripts/real_data/background.py` (see "Reproducing" at the end), which
reads the same library functions training and evaluation will use, so the
background documented here is the background that gets used.

## Where the data comes from

The catalogue is **DES Y6 Gold**, the collaboration's value-added catalogue
built on the six years of DES imaging, served by the NOIRLab Astro Data Lab
as `des_dr2.y6_gold`. It does not appear in the Data Lab schema browser, but
it is queryable, and it is the right table for this rather than
`des_dr2.main`, which holds the same objects (691,483,608 of them) with less
information attached:

- **Foreground masking.** Gold carries `flags_foreground`, which flags the
  surroundings of bright stars, nearby galaxies and bright globular clusters.
  Each of those is a localized stellar overdensity, which is exactly what a
  stream detector reacts to; `main` would have required building that mask.
- **Photometry meant for stars.** Gold has PSF magnitudes; `main` has
  `mag_auto`, an aperture measurement designed for galaxies.
- **Star/galaxy separation.** Gold carries several classifiers, including
  `ext_xgb`, the one streamobs's DES Y6 model reproduces (below).

A single full-footprint query exceeds Data Lab's execution limit, so
`scripts/fetch_des_dr2.py` downloads one-degree declination strips from
Dec −70° to +5°. A strip the server refuses for its size is split in two and
retried; one that fails for any other reason (a 502, a dropped connection) is
retried after a pause, since splitting it would not help. Finished strips are
skipped, so an interrupted download resumes. A `manifest.json` beside the
files records the table, the exact SQL, the column meanings and the download
date.

**Result: 49,068,865 stars in 75 strips, 1.70 GB**, downloaded on
2026-09-24, into `~/Documents/data/DES_yr6/`.

## The selection applied at download

These cuts are applied by the query itself, so nothing outside them is ever
downloaded:

| cut | why |
|---|---|
| `flags_footprint = 1` | inside the survey footprint |
| `flags_foreground = 0` | not near a bright star, a nearby galaxy or a bright cluster |
| `flags_gold = 0` | Gold's own catalogue-quality flag |
| `0 <= ext_xgb <= 1` | a star, by the classifier streamobs reproduces |
| `psf_mag_err < 0.543` in g and r | signal-to-noise above 2, a floor only |
| `15.5 <= g, r <= 25` | dereddened PSF magnitudes, wider than the analysis clip |

**The star/galaxy cut is the one that must match the simulation.**
streamobs's DES Y6 survey model (`config/surveys/des_yr6.yaml`) derives both
its stellar completeness and its galaxy misclassification for exactly
`0 <= EXT_XGB <= 1`. Any other classifier would make the real catalogue and
the injected streams two different samples, and the difference is not small:
the classifiers disagree most at the faint end, where at g = 24-25.5
`ext_coadd` calls 12.9% of objects stars and `ext_mash` 3.5%. An early version
of the download used the union of `ext_coadd` and `ext_mash`, thinking it the
cautious choice; it missed **14.6% of all EXT_XGB stars**, and was discarded.

**Extinction is already applied**: the `psf_mag_aper_8_*_corrected` columns
are the raw magnitudes minus `a_fiducial_*` (to within 0.001 mag over the
range kept), and `a_fiducial_g / ebv_sfd98` is 3.186, the same SFD98
coefficient streamobs uses (2.140 in r).

Magnitudes are stored as float32, positions as float64. The two
classifiers that do not define the sample, `ext_coadd` and `ext_mash`, are
kept as diagnostics; they cost 1.9% of the file. Gold marks a missing
magnitude with `-9999000000` rather than a null, which the magnitude range
removes.

```{image} figures/real_background/density.png
:alt: Density of DES Y6 Gold stars over the footprint, as downloaded
:width: 100%
```

*Stars per pixel as downloaded, on a logarithmic scale shared by every map on
this page. The strong gradient is galactic latitude: the footprint's east and
west ends run toward the plane, an order of magnitude denser than its centre.
The round holes are `flags_foreground` at work — bright stars and the bright
globular clusters (NGC 1851, 1261, 288, 1904 and 7089 all sit in one).*

```{image} figures/real_background/cmd.png
:alt: Colour-magnitude diagram and magnitude distribution of the downloaded stars
:width: 100%
```

*Left: the colour-magnitude diagram, in stars per bin on a log scale. The
vertical edge near g − r = 0.3 is the halo's main-sequence turnoff, the
dense column at g − r ≈ 1.4 is the disc's M dwarfs, and the diagonal edges are
the 15.5-25 limits applied in both bands at once. Right: the magnitude
distributions. r turns over near 23.2, well before g, because a red star at
r ≈ 24 is fainter than 25 in g and fails the g limit.*

## Building the training background

### The analysis cuts

Two more cuts, applied to the downloaded stars:

| cut | value | why |
|---|---|---|
| signal-to-noise | > 5 in both g and r | streamobs anchors its DES Y6 depth at S/N 5 — the magnitude where the photometric scatter reaches 0.2171 mag — so real and simulated stars are cut at the same point |
| magnitude clip | 16 ≤ g, r ≤ 24.5 | what the trained models assume; 16 is also streamobs's saturation limit |

```{image} figures/real_background/signal_to_noise.png
:alt: Magnitude error against magnitude, with the signal-to-noise cut and the clip
:width: 100%
```

*Magnitude error against magnitude, with the S/N 5 threshold (dashed,
error 0.217) and the clip (dotted). The bulk of the stars only crosses the
threshold beyond 24.5, so inside the clip the S/N cut barely acts: it removes
33,058 stars, **0.08%** of those the clip keeps. S/N 10 would have removed
4.7%.*

Alone, the S/N cut keeps 99.14% of the download and the clip 81.81%; together
they keep **40,111,394 stars (81.7%)**.

### Known streams

A background containing a real stream would teach the model to call streams
background, so every stream the DES 2018 analysis identified is masked, plus
**Sagittarius**, which that analysis also excluded: no cold stream is being
looked for inside it.

- **Tracks** come from `galstreams`. Its `MWStreams()` class raises an
  `IndexError` against astropy 8, so the track files it ships are read
  directly; both kinds are already densified (measured tracks, and the Shipp et
  al. endpoint pairs interpolated to 200 points). All references for a stream
  are used, so the mask covers every published version of its track.
- **Widths** are those of Shipp et al. (2018), the same numbers the injected
  analogues use; a test keeps the two tables identical.
- **How far out**: three times the width for a cold stream, where the width is
  a Gaussian sigma. For a structure at least 2° wide the quoted width is
  already its extent, so it is masked to that width once: tripling
  Sagittarius's 6° would take 27% of the footprint by itself.

```{include} figures/real_background/streams_table.md
```

**Jhelum deserves a note.** It costs more than any other stream, and more than
Sagittarius, because `galstreams` holds a track for it from Ibata et al.
(2024) that is **95.6° long**, against about 28° in the earlier measurements —
it is the broad diagonal band crossing the whole footprint in the final mask
below. Masking it is the conservative choice and is what was done. Dropping it
is one argument (`exclude_tracks={"Jhelum.ibata2024"}`) and would raise the
usable sky from 68.4% of the footprint to 77.6%.

```{image} figures/real_background/streams.png
:alt: Tracks of the masked streams over the DES density
:width: 100%
```

*The tracks of every masked stream over the stars that passed the cuts. Five of
them — Jhelum, Ravi, Chenab, Tucana III and Indus — crowd together between
RA 320° and 345°, Dec −50° and −62°, which is why their labels are offset.*

```{image} figures/real_background/streams_masked.png
:alt: DES density after removing the stream mask
:width: 100%
```

*The same stars with the stream mask applied: 13,208,188 removed.*

### Globular clusters and dwarf galaxies

The same argument applies to catalogued clusters and dwarfs: real stellar
overdensities, already known, and not what the model is for. They are masked
in training so they are never learned as background, and must be masked in
evaluation too, so that finding one counts neither as a detection nor as a
false alarm.

The catalogues are those of the Local Volume Database, kept in `data/others/`
and read by `streamgoggles.objects_overlap`. An object is masked if its centre
lies on a footprint pixel (and, for a cluster, more than 10° from the galactic
plane), out to **five half-light radii**, with a 0.05° floor for the many
objects whose radius is not published.

Two things about what that selects:

- **Only three globular clusters** — AM 1, Eridanus and Whiting 1 — because
  the bright ones are already gone: `flags_foreground` cut out a hole around
  each of NGC 1851, 1261, 288, 1904 and 7089, far wider than five half-light
  radii, so their centres are no longer on a footprint pixel. On a footprint
  without Gold's foreground flag they would have to be selected by proximity
  instead.
- **The "dwarf" catalogue includes other Local Volume galaxies**, not only
  Milky Way satellites: the clump at RA 0-15°, Dec −18° to −40° is the
  Sculptor group (NGC 55, NGC 247, NGC 300 and their companions), and IC 1613
  sits near the equator. They are masked all the same.

```{image} figures/real_background/objects.png
:alt: Globular clusters and dwarf galaxies inside the DES footprint
:width: 100%
```

*The 49 dwarf galaxies (circles) and 3 globular clusters (squares) inside the
footprint, labelled where a label fits.*

The masks are small: 12.2 deg² in all, of which 6.5 deg² is not already inside
a stream mask. The full list:

```{include} figures/real_background/objects_table.md
```

### The final mask and the background

```{image} figures/real_background/final_mask.png
:alt: The final mask: background, known streams, clusters and dwarfs
:width: 100%
```

*Each pixel of the footprint in exactly one category, streams taking
precedence where they overlap an object, so the areas in the legend add up to
the footprint.*

| | area | share of footprint | stars |
|---|---|---|---|
| footprint | 5,026 deg² | 100% | 40,111,394 after the cuts |
| known streams | 1,582 deg² | 31.5% | 13,208,188 removed |
| clusters and dwarfs, outside the streams | 6.5 deg² | 0.1% | 131,884 removed (some also in a stream) |
| **background** | **3,438 deg²** | **68.4%** | **26,834,962** |

```{image} figures/real_background/background.png
:alt: Density of the final background
:width: 100%
```

*The background streams will be injected into, on the same density scale as
the download.*

Three products come out of this, all outside version control:

- `~/Documents/data/DES_yr6/des_yr6_background.parquet`: the 26,834,962
  stars, as `ra`, `dec`, `des_yr6_g_obs`, `des_yr6_r_obs` — the four columns
  the injection pipeline reads.
- `des_yr6_background.json` beside it: everything above, machine-readable —
  the inherited SQL, the cuts, the mask parameters, each stream's width and
  cost, and the totals.
- `data/others/mask_des_yr6_background_nside512.fits.gz`: the usable sky,
  available as `get_footprint("des_yr6_background")`, so training and
  evaluation read the same mask by name.

## What comes next

The mask exists but is not yet used by the injector or the evaluator; that is
the next step, together with training on this background instead of the
simulated one.

**Dust, as a possible extra input.** The density gradient toward the galactic
plane is real structure the model will see in every window near the edges of
the footprint. One way to let it learn to be less confident there is to give
it a window of the E(B−V) map as an additional channel. streamobs already
carries that map for DES Y6 (`ebv_map` on the loaded survey), and it is
**nside 512 in ring order** — the pipeline's own resolution — so it can be cut
out with the same window projection as the matched-filter maps and inherits
the same valid-pixel mask, with no regridding. A survey whose map differs in
resolution would need `healpy.ud_grade` first; it should then be masked with
the same footprint before normalization, so the channel never sees pixels the
data does not have.

## Reproducing

From the repository root, in the `streamml` environment:

```bash
python scripts/fetch_des_dr2.py --out ~/Documents/data/DES_yr6    # the download, restartable
python scripts/real_data/background.py --write                    # mask, figures, background
```

The first writes the strips and `manifest.json`; the second writes the mask to
`data/others/`, every figure and both tables on this page to
`docs/source/narrative/figures/real_background/`, and, with `--write`, the
background and its sidecar. The notebook `notebooks/real_des_data.ipynb` walks
through the same steps interactively.
