"""The line model on the real DES sky.

A network that answers with lines -- the U-Net's features and its input
summed along every line through a window, a small network over the lines
(`streamgoggles.models.hough`; the guide's narrative/line_model) -- trained
and scored on the real DES Y6 sky like the per-pixel network of
`scripts/experiments/real_des/patches.py`, whose skies, labels,
normalizations and simulated copies it shares: trained on one fold's
training sky, scored on the other's, and run out of fold over the whole sky.

Steps (from the repository root, streamml environment):

  python scripts/experiments/line_model/run.py train --config "hough/band2 residual" --seed 42 [--train-sky fold1]
  python scripts/experiments/line_model/run.py evaluate --config "hough/band2 residual" [--train-sky fold1] [--sets fainter | "length scan"]
  python scripts/experiments/line_model/run.py figures     # the copies' figures, and the guide's
  python scripts/experiments/line_model/run.py line-sky --config "hough/band2 residual x4"   # the whole sky
  python scripts/experiments/line_model/run.py leads       # the catalogue's leads, one by one
  python scripts/experiments/line_model/run.py des2018     # the DES 2018 streams where they are

Docs: docs/source/experiments/line_model.
"""

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "line_model"
DOC_FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "line_model"


def _load_patches():
    spec = importlib.util.spec_from_file_location(
        "real_des_patches", REPO / "scripts" / "experiments" / "real_des" / "patches.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# What this experiment shares with the per-pixel one (patches.py): the skies,
# the labels and normalizations, the simulated copies and where they are
# placed, the matched filter's smooth background, and its per-pixel models
# for the comparisons.
_patches = _load_patches()
real_des = _patches.real_des
build_sky = _patches.build_sky
normalizer = _patches.normalizer
channels = _patches.channels
evaluation_streams = _patches.evaluation_streams
smooth_backgrounds = _patches.smooth_backgrounds
half_recovery_snr = _patches.half_recovery_snr
PIXEL_CONFIGS = _patches.CONFIGS
pixel_model_stem = _patches.model_stem
pixel_result_dir = _patches.result_dir
SKIES = _patches.SKIES
EVALUATED_ON = _patches.EVALUATED_ON
SEEDS = _patches.SEEDS
WINDOWS = _patches.WINDOWS
IMAGE_PIX = _patches.IMAGE_PIX
N_PLACEMENTS = _patches.N_PLACEMENTS
N_NULL_BANDS = _patches.N_NULL_BANDS
EVAL_SEED = _patches.EVAL_SEED
MAX_LENGTH = _patches.MAX_LENGTH
SNR_EDGES = _patches.SNR_EDGES
MF_BACKGROUND_NSIDE = _patches.MF_BACKGROUND_NSIDE

CONFIGS = {
    # the answer is a line: U-Net features and the input summed along every
    # line through the window (Hough), then a small network over the lines
    # (`streamgoggles.models.hough`); trained on the band label turned into
    # its line, with cross-entropy, and more stream-free windows (30%) since
    # a window may now answer "no line"
    # a combination: the four per-pixel count-label models, their output
    # (logit minus its window median) summed along every line -- the same
    # line search as the line model, on top of the per-pixel network
    "count/window x4 lines": {
        "label": None,
        "normalizer": "window",
        "parts": ["count/window"],
        "seeds": [42, 43, 44, 45],
        "lines": True,
    },
    "hough/band": {
        "label": "band",
        "normalizer": "window",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
    },
    # the same line model on inputs that are matched-filter S/N maps: each
    # channel's excess over a local plane (1.8-degree Gaussian), over its
    # square root (`ResidualNormalizer`) -- so the input's own line sums, which
    # reach the lines untouched, are the matched filter's line search
    "hough/band residual": {
        "label": "band",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
    },
    # the line model on S/N inputs, with the band label down to 2 degrees: the
    # 4-degree minimum made every window holding a shorter clear stretch a
    # negative example, and the model learned to answer "no line" to short
    # streams however bright (the length scan)
    "hough/band2 residual": {
        "label": "band2",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
    },
    # what more models and longer training buy the 2-degree line model: four
    # quick seeds, one quick seed (the fair reference for one long model), and
    # one model trained four times longer (19,200 windows)
    "hough/band2 residual x4": {
        "label": None,
        "normalizer": "residual",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "parts": ["hough/band2 residual"],
        "seeds": [42, 43, 44, 45],
    },
    "hough/band2 residual s42": {
        "label": None,
        "normalizer": "residual",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "parts": ["hough/band2 residual"],
        "seeds": [42],
    },
    "hough/band2 residual long": {
        "label": "band2",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
        "windows": 19200,
        "seeds": [42],
    },
    # the 2-degree line model trained on streams as strong as the DES 2018
    # streams are in our data -- surface brightness 32.5-35.5 instead of
    # 32-34.5: the copies at Table 1's surface brightness are 0.5-2 mag
    # stronger than the real streams (des2018_known.md)
    "hough/band2 residual des": {
        "label": "band2",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
        "training_set": "population des2018",
    },
    # the 2-degree line model taught lines only where they show: the band
    # label from S/N 5 in the window instead of 2 -- training on fainter
    # streams made the model less sensitive (des2018_known.md), as if lines it
    # could not see taught it to answer softly everywhere
    "hough/band2s5 residual": {
        "label": "band2s5",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
    },
    # both: lines only where they show, on streams as strong as the DES 2018
    # streams are in our data (32.5-35.5) -- fainter training failed under the
    # S/N-2 label, perhaps only because of the lines it could not see
    "hough/band2s5 residual des": {
        "label": "band2s5",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
        "training_set": "population des2018",
    },
    # the S/N-5 label at the long tier (19,200 windows), two models per fold:
    # what longer training buys the best quick line model -- and whether it
    # keeps the short streams one long S/N-2 model per fold lost
    "hough/band2s5 residual long": {
        "label": "band2s5",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
        "windows": 19200,
        "seeds": [42, 43],
    },
    # the S/N-5 line model on more short streams: lengths log-uniform over
    # 4-30 degrees -- trained long, the line model lost the short streams
    # whatever its label, as if the uniform range's long streams tuned it
    "hough/band2s5 residual short": {
        "label": "band2s5",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "training": {"background_fraction": 0.3},
        "training_set": "population short",
    },
    # the quick S/N-5 models' seed spread: two more per fold (44, 45), alone
    # and averaged with the first two -- how much of their lead is a good draw
    "hough/band2s5 residual s44": {
        "label": None,
        "normalizer": "residual",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "parts": ["hough/band2s5 residual"],
        "seeds": [44, 45],
    },
    "hough/band2s5 residual x4": {
        "label": None,
        "normalizer": "residual",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "parts": ["hough/band2s5 residual"],
        "seeds": [42, 43, 44, 45],
    },
    # the S/N-5 line model with segment lines: the window's lines and those
    # of nine half-overlapping 48-pixel sub-windows (5.5 degrees), so that a
    # short stream fills a line of its size (`SegmentLines`)
    "hough/band2s5 residual seg": {
        "label": "band2s5",
        "normalizer": "residual",
        "loss": "bce",
        "hough": {
            "features": 8,
            "n_theta": 90,
            "rho_step": 2.0,
            "min_pixels": 20,
            "sub_size": 48,
            "sub_stride": 24,
        },
        "training": {"background_fraction": 0.3},
    },
    # the segment-line model's seeds: two more per fold (44, 45), alone and
    # averaged with the first two
    "hough/band2s5 residual seg s44": {
        "label": None,
        "normalizer": "residual",
        "hough": {
            "features": 8,
            "n_theta": 90,
            "rho_step": 2.0,
            "min_pixels": 20,
            "sub_size": 48,
            "sub_stride": 24,
        },
        "parts": ["hough/band2s5 residual seg"],
        "seeds": [44, 45],
    },
    "hough/band2s5 residual seg x4": {
        "label": None,
        "normalizer": "residual",
        "hough": {
            "features": 8,
            "n_theta": 90,
            "rho_step": 2.0,
            "min_pixels": 20,
            "sub_size": 48,
            "sub_stride": 24,
        },
        "parts": ["hough/band2s5 residual seg"],
        "seeds": [42, 43, 44, 45],
    },
    # four of them per fold: is the long model's sky a good draw?
    "hough/band2 residual long x4": {
        "label": None,
        "normalizer": "residual",
        "hough": {"features": 8, "n_theta": 90, "rho_step": 2.0, "min_pixels": 20},
        "parts": ["hough/band2 residual long"],
        "seeds": [42, 43, 44, 45],
    },
}


def result_dir(train_sky):
    """Where the line models trained on a sky, and their scores, are kept."""
    return OUT / train_sky


def model_stem(config, seed, train_sky="fold0"):
    return result_dir(train_sky) / "models" / f"{config.replace('/', '_')}_seed{seed}"


def hough_parts(config):
    """(model factory, view wrapper) for a Hough configuration: the model from
    `build_model(kind="hough")` with the configuration's options (backbone
    as the per-pixel models'), and the view turning the label into lines."""
    from streamgoggles.models import build_model
    from streamgoggles.models.hough import HoughTargetTransform, line_grid

    sp = real_des().stream_parameters_module()
    options = CONFIGS[config]["hough"]
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    grid = line_grid(
        image_pix,
        image_pix,
        options["n_theta"],
        options["rho_step"],
        options["min_pixels"],
        options.get("sub_size"),
        options.get("sub_stride"),
    )

    def factory(in_channels):
        return build_model(
            in_channels,
            image_pix,
            kind="hough",
            depth=sp.MODEL["depth"],
            base_width=sp.MODEL["base_width"],
            **options,
        )

    def wrapper(view):
        return HoughTargetTransform(view, grid)

    return factory, wrapper


def load_line_models(config, train_sky, seeds=SEEDS):
    """The line models of a configuration trained on one sky, in eval mode:
    each of its ``parts`` (the configurations whose trained models it
    gathers; by default itself) at each of its seeds."""
    import torch

    from streamgoggles.datasets.transforms import QueryDistanceTransform

    factory, _ = hough_parts(config)
    models = []
    for part in CONFIGS[config].get("parts", [config]):
        for seed in CONFIGS[config].get("seeds", seeds):
            model = factory(QueryDistanceTransform.n_channels)
            model.load_state_dict(
                torch.load(model_stem(part, seed, train_sky).with_suffix(".pt"))
            )
            models.append(model.eval())
    return models


def train(config, seed, train_sky="fold0"):
    """One line model of a configuration, on one fold's training sky."""
    import torch

    rd = real_des()
    sp = rd.stream_parameters_module()
    stem = model_stem(config, seed, train_sky)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.with_suffix(".pt").exists():
        print(f"{stem.name}: already trained", flush=True)
        return
    background, injector, _ = build_sky(
        train_sky,
        CONFIGS[config]["label"],
        CONFIGS[config].get("image_pix", IMAGE_PIX),
    )
    start = time.time()
    windows = CONFIGS[config].get("windows", WINDOWS)
    factory, wrapper = hough_parts(config)
    model, _, result = sp.train(
        seed,
        windows,
        background,
        injector,
        training_set=CONFIGS[config].get("training_set", rd.TRAINING_SET),
        normalizer=normalizer(CONFIGS[config]["normalizer"], channels()),
        loss_name=CONFIGS[config].get("loss"),
        training_options=CONFIGS[config].get("training"),
        model_factory=factory,
        view_wrapper=wrapper,
    )
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": config,
                "seed": seed,
                "windows": windows,
                "train_s": time.time() - start,
                "train_losses": result["train_losses"],
                "val_losses": result["val_losses"],
            }
        )
    )
    print(f"{stem.name}: trained in {time.time() - start:.0f}s", flush=True)


HOUGH_NULL_WINDOWS = 600  # stream-free windows per queried distance
HOUGH_FALSE_ALARM = 0.01  # blind test: share of stream-free windows with a line found
HOUGH_MIN_LENGTH_DEG = 2.0  # a window scores a copy if it holds this much of it
# A stream-free window must lie on calibration sky: at least this share of
# its valid pixels. The per-pixel false-alarm rates are measured on
# calibration pixels alone; a window reaching into the training-only sky
# meets the leftover structure the calibration mask removes -- Sagittarius's
# wing past its 6-degree mask, a real line the line model rightly finds.
HOUGH_NULL_CALIBRATION = 0.95
# The lines of a per-pixel model scored as lines (configurations with "lines")
HOUGH_GRID = {"n_theta": 90, "rho_step": 2.0, "min_pixels": 20}


def _run_length_pix(mask, grid):
    """The length, in pixels, of a band's longest straight run in a window:
    the most of the mask a line gathers (`line_counts`), over the width of the
    strip a line gathers, ``rho_step`` pixels. (Before 2026-10-02 the scripts
    compared `line_counts` itself with 4 degrees: the same windows as 2
    degrees of run now.)"""
    from streamgoggles.models.hough import line_counts

    return float(line_counts(mask, grid).max()) / grid.rho_step


def window_level_sky(eval_sky, image_pix=IMAGE_PIX, label="count"):
    """The evaluation sky as the window-level scoring sees it.

    A namespace: background, injector, pix, nside, valid (the valid sky),
    window_deg, tiles (the half-overlapping windows a search would use) with
    their tile_pixels and projections, and calibration -- the calibration
    sky exactly as `evaluate` has it (valid, calibration mask, and seen by
    the window most central to it), so copies land on the same places and
    the evaluations compare copy by copy.
    """
    import types

    import healpy as hp
    import numpy as np

    from streamgoggles.matched_filter import (
        WindowProjection,
        stitch_windows_to_healpix,
        window_to_healpix_indices,
    )
    from streamgoggles.windows import tile_footprint

    rd = real_des()
    background, injector, pix = build_sky(eval_sky, label, image_pix)
    nside = pix.nside
    valid = background.valid_mask_full
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        valid, nside, tile_size_deg=window_deg, stride_deg=window_deg / 2
    )
    projections = [WindowProjection.for_window(t, pix, valid) for t in tiles]
    calibration = valid.copy()
    if "fold" in SKIES[eval_sky]:
        calibration &= hp.read_map(rd.CALIBRATION_MASK).astype(bool)
    seen = stitch_windows_to_healpix(
        [np.where(p.valid, 1.0, np.nan) for p in projections], tiles, pix, nside
    )[0]
    calibration &= valid & np.isfinite(seen)
    return types.SimpleNamespace(
        background=background,
        injector=injector,
        pix=pix,
        nside=nside,
        valid=valid,
        window_deg=window_deg,
        tiles=tiles,
        tile_pixels=[window_to_healpix_indices(t, pix, nside)[0] for t in tiles],
        projections=projections,
        calibration=calibration,
    )


def copy_places(index, stream, sky):
    """The places of the evaluation's copy number ``index`` (its position in
    `evaluation_streams`), drawn as `evaluate` draws them: a random
    calibration pixel and position angle, kept if at least 90% of the copy's
    band is on calibration sky. Yields (placement, ra, dec, rotation, band)
    for at most N_PLACEMENTS places, the band cut to calibration sky."""
    import healpy as hp
    import numpy as np

    from streamgoggles.evaluation.footprint import track_band

    rd = real_des()
    rng = np.random.default_rng([EVAL_SEED, index])
    candidates = np.flatnonzero(sky.calibration)
    placed = 0
    tries = 0
    while placed < N_PLACEMENTS and tries < 50 * N_PLACEMENTS:
        tries += 1
        ra, dec = hp.pix2ang(sky.nside, int(rng.choice(candidates)), lonlat=True)
        rotation = float(rng.uniform(0, 360))
        band = track_band(
            [rd.great_circle(float(ra), float(dec), rotation, stream["length"])],
            stream["width"],
            sky.nside,
        )
        if band.sum() == 0 or (band & sky.calibration).sum() / band.sum() < 0.9:
            continue
        yield placed, float(ra), float(dec), rotation, band & sky.calibration
        placed += 1


def scoring_models(config, train_sky, seeds=SEEDS):
    """(models, grid): a configuration's models trained on one sky, in eval
    mode, and the lines their answers are scored on -- a line model's own;
    for a per-pixel ensemble (configurations with "lines"), HOUGH_GRID's."""
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads

    # before any model is built: a sparse tensor's coalesce starts OpenMP
    # workers, which crash with this environment's duplicate libomp
    configure_torch_threads(num_workers=0)
    if "hough" in CONFIGS[config]:  # a line model: one answer per line
        models = load_line_models(config, train_sky, seeds)
        return models, models[0].hough.grid
    from streamgoggles.datasets.transforms import QueryDistanceTransform
    from streamgoggles.models import build_model
    from streamgoggles.models.hough import HoughLines

    sp = real_des().stream_parameters_module()
    models = []
    for part in CONFIGS[config].get("parts", [config]):
        for seed in CONFIGS[config].get("seeds", seeds):
            model = build_model(
                QueryDistanceTransform.n_channels,
                **{**sp.MODEL, **PIXEL_CONFIGS[part].get("model", {})},
            )
            model.load_state_dict(
                torch.load(pixel_model_stem(part, seed, train_sky).with_suffix(".pt"))
            )
            models.append(model.eval())
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    return models, HoughLines(image_pix, image_pix, **HOUGH_GRID)


def line_scorer(config, models, grid):
    """``score(maps_full, valid, smooth, windows, query)``: the scores of
    every line through some windows of a sky at one queried distance --
    {"network": ..., "matched filter": ...}, (n_windows, n_theta, n_rho)
    each, NaN on lines too short to count.

    The network's score is the line model's probability (the mean over
    ``models``); for a per-pixel ensemble, the logit of its mean output, minus
    its median over the window, summed along the line over the square root of
    the line's length. The matched filter's is the counts at the queried
    distance minus the smooth local background, over the background's square
    root, summed the same way (`HoughLines`). ``maps_full``: the sky's
    channels, full-sky; ``valid``: its valid sky; ``smooth``: its smooth
    background at the queried distance (`smooth_backgrounds`); ``windows``:
    `WindowProjection`s.
    """
    import numpy as np
    import torch

    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )

    sp = real_des().stream_parameters_module()
    chans = channels()
    norm = normalizer(CONFIGS[config]["normalizer"], chans)
    transforms = {
        q: QueryDistanceTransform(
            StreamMapTransform(normalizer=norm, augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=q,
        )
        for q in sp.QUERY_GRID
    }
    good_at = {
        q: next(
            i
            for i, c in enumerate(chans)
            if c["filter"] == "good" and c["distance_modulus"] == q
        )
        for q in sp.QUERY_GRID
    }

    def score(maps_full, valid, smooth, windows, query):
        inputs, matched, valids = [], [], []
        for projection in windows:
            valids.append(projection.valid)
            valid_at = valid[projection.pixnums]
            stack = np.stack(
                [
                    projection.image(np.where(valid_at, m[projection.pixnums], 0.0))
                    for m in maps_full
                ]
            ).astype(np.float32)
            sample = {
                "map_stack": stack,
                "label_stack": np.zeros_like(stack),
                "valid_mask": projection.valid,
                "params": {},
                "metadata": {"channels": chans},
            }
            inputs.append(transforms[query](sample)["map_stack"])
            background = projection.image(
                np.where(valid_at, smooth[projection.pixnums], 0.0)
            )
            residual = (stack[good_at[query]] - background) / np.sqrt(
                np.maximum(background, 0.5)
            )
            matched.append(grid(np.where(projection.valid, residual, 0.0)))
        network = []
        with torch.no_grad():
            for start in range(0, len(inputs), 32):
                x = torch.as_tensor(np.stack(inputs[start : start + 32]))
                if "hough" in CONFIGS[config]:
                    network.append(
                        np.mean([torch.sigmoid(m(x))[:, 0].numpy() for m in models], 0)
                    )
                    continue
                answer = np.mean([m(x)[:, 0].numpy() for m in models], 0)
                answer = np.log(np.clip(answer, 1e-6, 1 - 1e-6))
                answer -= np.log1p(-np.exp(answer))  # the logit
                for image, ok in zip(answer, valids[start : start + 32], strict=True):
                    residual = np.where(ok, image - np.median(image[ok]), 0.0)
                    network.append(grid(residual)[None])
        network = np.concatenate(network).astype(np.float32)
        matched = np.stack(matched).astype(np.float32)
        network[:, ~grid.valid] = np.nan
        matched[:, ~grid.valid] = np.nan
        return {"network": network, "matched filter": matched}

    return score


def null_windows(sky):
    """The stream-free windows of an evaluation sky (`window_level_sky`):
    HOUGH_NULL_WINDOWS windows centred on random calibration pixels, at least
    half valid, lying on calibration sky (HOUGH_NULL_CALIBRATION of their
    valid pixels), unrotated like the tiles -- the same at every call."""
    import healpy as hp
    import numpy as np

    from streamgoggles.matched_filter import WindowProjection
    from streamgoggles.windows import Window

    rng = np.random.default_rng([EVAL_SEED, 7])
    candidates = np.flatnonzero(sky.calibration)
    windows = []
    while len(windows) < HOUGH_NULL_WINDOWS:
        ra, dec = hp.pix2ang(sky.nside, int(rng.choice(candidates)), lonlat=True)
        window = Window(
            center_ra=float(ra),
            center_dec=float(dec),
            width_deg=sky.window_deg,
            height_deg=sky.window_deg,
        )
        projection = WindowProjection.for_window(window, sky.pix, sky.valid)
        if projection.valid.mean() < 0.5:
            continue
        on_calibration = projection.image(
            sky.calibration[projection.pixnums].astype(float)
        )
        share = (on_calibration > 0.99)[projection.valid].mean()
        if share >= HOUGH_NULL_CALIBRATION:
            windows.append(projection)
    return windows


def evaluate_hough(
    config, seeds=SEEDS, train_sky="fold0", sets=("DES 2018", "distance scan")
):
    """Window-level scoring of a line model -- or of a per-pixel ensemble
    with a line search on top (configurations with "lines") -- and of the
    matched filter's own line sums (a classical line search), on the same
    copies as `evaluate`.

    Each window answers with one score per line through it: the line model's
    probability (the mean over its seeds); for a per-pixel ensemble, the logit
    of its mean output, minus its median over the window, summed along the
    line over the square root of the line's length; for the matched filter,
    the counts at the queried distance minus the smooth local background, over
    the background's square root, summed the same way (`HoughLines`). The
    lines "along the track" of a copy, in a window, are those `hough_target`
    makes of the copy's band there; a window scores a copy if it holds at
    least HOUGH_MIN_LENGTH_DEG of it.

    Two tests, against HOUGH_NULL_WINDOWS stream-free windows centred on
    random points of the calibration sky and lying on it (at least
    HOUGH_NULL_CALIBRATION of their valid pixels), at the same queried
    distance:

    - blind: found if, in a window, a line along the track scores above the
      level that the best line of a stream-free window exceeds in
      HOUGH_FALSE_ALARM of them -- a search that does not know the track,
      with that rate of false lines per window;
    - known track: in the window holding the longest stretch of the copy, the
      best line along the track against the same lines in the null windows;
      found when at most a fraction 1 / (N_NULL_BANDS + 1) of them score as
      high (the level of `evaluate`'s band tests).
    """
    import numpy as np
    import pandas as pd

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.models.hough import hough_target

    configure_torch_threads(num_workers=0)
    rd = real_des()
    sp = rd.stream_parameters_module()
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    sky = window_level_sky(EVALUATED_ON[train_sky], image_pix)
    background, injector, pix = sky.background, sky.injector, sky.pix
    valid, tile_pixels, projections = sky.valid, sky.tile_pixels, sky.projections
    models, grid = scoring_models(config, train_sky, seeds)
    score = line_scorer(config, models, grid)
    min_length_pix = HOUGH_MIN_LENGTH_DEG / pix.pixel_scale_deg
    streams = evaluation_streams(sets)
    queries = sorted(
        {
            min(sp.QUERY_GRID, key=lambda q: abs(q - p["distance_modulus"]))
            for *_, p in streams
        }
    )
    chans = channels()
    good_at = {
        q: next(
            i
            for i, c in enumerate(chans)
            if c["filter"] == "good" and c["distance_modulus"] == q
        )
        for q in queries
    }
    smooth_background = smooth_backgrounds(background, queries)
    background_maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
    windows = null_windows(sky)
    null, thresholds, null_rows, null_best = {}, {}, [], {}
    start = time.time()
    for q in queries:
        null[q] = score(background_maps, valid, smooth_background[q], windows, q)
        for scorer, values in null[q].items():
            best = np.nanmax(values.reshape(len(values), -1), axis=1)
            null_best[f"{scorer}|{q:.1f}"] = best
            thresholds[(scorer, q)] = float(np.quantile(best, 1 - HOUGH_FALSE_ALARM))
            null_rows.append(
                {
                    "config": config,
                    "scorer": scorer,
                    "query": q,
                    "threshold": thresholds[(scorer, q)],
                    "median_best_line": float(np.median(best)),
                }
            )
    print(f"{config}: null windows in {time.time() - start:.0f}s", flush=True)

    known_level = 1.0 / (N_NULL_BANDS + 1)
    rows = []
    for index, (kind, name, stream) in enumerate(streams):
        start = time.time()
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - stream["distance_modulus"]))
        good = good_at[query]
        n_placed = 0
        for placed, ra, dec, rotation, band in copy_places(index, stream, sky):
            n_placed += 1
            params = {
                "morphology": "uniform",
                "isochrone_model": sp.ISOCHRONE_FAMILY,
                "orientation": rotation,
                **stream,
            }
            injected = injector.inject_streams_full_sky(
                [params],
                np.random.default_rng([EVAL_SEED, index, placed]),
                centers=[(ra, dec)],
            )
            stream_stars = injected["stream_raw_full"][good][band].sum()
            background_stars = background.raw_map_full_dict["good"][query][band].sum()
            # the windows holding enough of the copy, and its lines in each
            scored, regions, lengths = [], [], []
            for i, px in enumerate(tile_pixels):
                if not band[px].any():
                    continue
                projection = projections[i]
                inside = projection.image(band[projection.pixnums].astype(float)) > 0.5
                length = _run_length_pix(inside, grid)
                if length >= min_length_pix:
                    scored.append(projection)
                    regions.append(hough_target(inside, grid) > 0)
                    lengths.append(length)
            row = {
                "config": config,
                "set": kind,
                "stream": name,
                "distance_modulus": stream["distance_modulus"],
                "query": query,
                "placement": placed,
                "input_snr": stream_stars / np.sqrt(max(background_stars, 1.0)),
                "n_windows": len(scored),
            }
            if scored:
                answers = score(
                    injected["map_full"], valid, smooth_background[query], scored, query
                )
                longest = int(np.argmax(lengths))
                for scorer, values in answers.items():
                    along = [
                        np.nanmax(v[r]) for v, r in zip(values, regions, strict=True)
                    ]
                    null_along = np.nanmax(
                        null[query][scorer][:, regions[longest]], axis=1
                    )
                    p_value = (1 + np.sum(null_along >= along[longest])) / (
                        1 + len(null_along)
                    )
                    row.update(
                        {
                            f"{scorer} score": float(max(along)),
                            f"{scorer} blind": bool(
                                max(along) >= thresholds[(scorer, query)]
                            ),
                            f"{scorer} known p": float(p_value),
                            f"{scorer} known": bool(p_value <= known_level + 1e-12),
                        }
                    )
            else:
                for scorer in ("network", "matched filter"):
                    row.update({f"{scorer} blind": False, f"{scorer} known": False})
            rows.append(row)
        print(
            f"{config} {name}: {n_placed} placements, {time.time() - start:.0f}s",
            flush=True,
        )

    out = result_dir(train_sky)
    out.mkdir(parents=True, exist_ok=True)
    name = config.replace("/", "_")
    suffix = (
        ""
        if tuple(sets) == ("DES 2018", "distance scan")
        else "__" + "_".join(x.replace(" ", "-") for x in sets)
    )
    pd.DataFrame(rows).to_csv(out / f"hough_{name}{suffix}.csv", index=False)
    if not suffix:
        pd.DataFrame(null_rows).to_csv(out / f"hough_null_{name}.csv", index=False)
        # each stream-free window's best line, per scorer and distance, paired
        # window by window: what a combination's false-alarm rate is read from
        np.savez(out / f"hough_null_best_{name}.npz", **null_best)


HOUGH_FIGURES = {
    # without the track: what a search can claim (the per-pixel test, which
    # counts flagged pixels near the true track, is shown for reference)
    "blind": [
        ("pixels", "count/window x4", "detected", "per-pixel network, current test"),
        ("lines", "hough/band", "network blind", "line network"),
        ("lines", "hough/band residual", "network blind", "line network, S/N inputs"),
        ("lines", "hough/band", "matched filter blind", "matched-filter lines"),
    ],
    # along the known track: the most the data allow
    "known": [
        ("lines", "hough/band", "network known", "line network"),
        ("lines", "hough/band residual", "network known", "line network, S/N inputs"),
        ("lines", "hough/band", "matched filter known", "matched-filter lines"),
        ("pixels", "count/window x4", "matched_filter_detected", "matched-filter band"),
    ],
    # the per-pixel network with a line search on top, blind and known track
    "combination": [
        ("pixels", "count/window x4", "detected", "per-pixel network, current test"),
        ("lines", "count/window x4 lines", "network blind", "per-pixel + lines, blind"),
        (
            "lines",
            "count/window x4 lines",
            "network known",
            "per-pixel + lines, known track",
        ),
        ("lines", "hough/band", "matched filter blind", "matched-filter lines, blind"),
    ],
}
# one colour per method, the same in every figure; dashed: no network
HOUGH_COLOURS = {
    "per-pixel network, current test": "#4d4d4d",
    "line network": "#1f6fb4",
    "line network, S/N inputs": "#2e8b57",
    "matched-filter lines": "#e07b39",
    "matched-filter lines, blind": "#e07b39",
    "matched-filter band": "#9e9e9e",
    "per-pixel + lines, blind": "#8e44ad",
    "per-pixel + lines, known track": "#c39bd3",
}


# Stream-free windows, centred on calibration sky but reaching out of it,
# where the best lines are strongest at m-M 17: two of the line model's (the
# edge of the Sagittarius mask) and the matched filter's (near a masked dwarf).
HOUGH_NULL_EXAMPLES = [(32.87, -11.34), (29.79, -12.94), (35.33, -39.94)]


def hough_null_figure(train_sky="fold0", config="hough/band", query=17.0):
    """hough_null_windows_{sky}.png: for each example window, the isochrone
    and decoy inputs, the matched-filter residual (smoothed), the valid sky,
    and the line model's answer over lines; the best line drawn in orange."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy import ndimage

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.matched_filter import WindowProjection
    from streamgoggles.windows import Window

    configure_torch_threads(num_workers=0)
    sp = real_des().stream_parameters_module()
    background, _, pix = build_sky(EVALUATED_ON[train_sky], "count")
    valid = background.valid_mask_full
    chans = channels()
    models = load_line_models(config, train_sky)
    grid = models[0].hough.grid
    view = QueryDistanceTransform(
        StreamMapTransform(normalizer=normalizer("window", chans), augment=False),
        query_grid=sp.QUERY_GRID,
        step=sp.STEP,
        query=query,
    )
    smooth_background = smooth_backgrounds(background, [query])[query]
    good = next(
        i
        for i, c in enumerate(chans)
        if c["filter"] == "good" and c["distance_modulus"] == query
    )
    maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
    size = pix.image_size_pix[0]
    window_deg = size * pix.pixel_scale_deg
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2
    fig, axes = plt.subplots(
        len(HOUGH_NULL_EXAMPLES), 5, figsize=(17, 3.6 * len(HOUGH_NULL_EXAMPLES))
    )
    for row, (ra, dec) in zip(axes, HOUGH_NULL_EXAMPLES, strict=True):
        window = Window(
            center_ra=ra, center_dec=dec, width_deg=window_deg, height_deg=window_deg
        )
        projection = WindowProjection.for_window(window, pix, valid)
        valid_at = valid[projection.pixnums]
        stack = np.stack(
            [
                projection.image(np.where(valid_at, m[projection.pixnums], 0.0))
                for m in maps
            ]
        ).astype(np.float32)
        x = view(
            {
                "map_stack": stack,
                "label_stack": np.zeros_like(stack),
                "valid_mask": projection.valid,
                "params": {},
                "metadata": {"channels": chans},
            }
        )["map_stack"]
        with torch.no_grad():
            answer = np.mean(
                [
                    torch.sigmoid(m(torch.as_tensor(x)[None]))[0, 0].numpy()
                    for m in models
                ],
                0,
            )
        answer[~grid.valid] = np.nan
        t, r = np.unravel_index(np.nanargmax(answer), answer.shape)
        theta, rho = grid.thetas[t], grid.rhos[r]
        best_line = np.abs(xx * np.cos(theta) + yy * np.sin(theta) - rho) < 1.0
        smooth = projection.image(
            np.where(valid_at, smooth_background[projection.pixnums], 0.0)
        )
        residual = np.where(
            projection.valid,
            (stack[good] - smooth) / np.sqrt(np.maximum(smooth, 0.5)),
            0.0,
        )
        panels = [
            (f"isochrone, m−M {query:g}", ndimage.gaussian_filter(x[1], 2)),
            ("decoy", ndimage.gaussian_filter(x[3], 2)),
            ("matched-filter residual", ndimage.gaussian_filter(residual, 2)),
            ("valid sky", projection.valid.astype(float)),
        ]
        for ax, (title, image) in zip(row[:4], panels, strict=True):
            ax.imshow(image, origin="lower", cmap="gray_r")
            ax.contour(best_line, levels=[0.5], colors="#e07b39", linewidths=0.8)
            ax.set_title(f"{title} — ({ra:.1f}, {dec:.1f})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        image = row[4].imshow(answer, origin="lower", aspect="auto", cmap="viridis")
        row[4].set_title(
            f"line network over (θ, ρ): best {np.nanmax(answer):.2f}", fontsize=8
        )
        row[4].set_xlabel("ρ bin", fontsize=8)
        row[4].set_ylabel("θ bin", fontsize=8)
        fig.colorbar(image, ax=row[4], fraction=0.046)
    fig.tight_layout()
    fig.savefig(
        DOC_FIGURES / f"hough_null_windows_{train_sky}.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


LINE_MODEL_FIGURES = REPO / "docs" / "source" / "narrative" / "figures" / "line_model"
# The worked examples of the guide page, each the evaluation's copy at its
# first place: the widest stream (found by the S/N-input line model at 7 of 8
# places), the faintest (6 of 8; the per-pixel network 1 of 8), and the
# shortest (0 of 8, where every other method finds it).
LINE_MODEL_EXAMPLES = [("Jhelum", 0), ("Wambelong", 0), ("Tucana III", 0)]


def line_model_figures(train_sky="fold0"):
    """The figures of the guide page `narrative/line_model.md`: the Hough
    transform on toy windows, the networks' plumbing, and worked examples on
    the real sky."""
    LINE_MODEL_FIGURES.mkdir(parents=True, exist_ok=True)
    line_model_toy()
    line_model_architecture()
    for name, placement in LINE_MODEL_EXAMPLES:
        line_model_example(name, placement, train_sky)


def line_model_toy():
    """hough_toy.png: four windows and their line sums -- a line, a point, a
    short segment, and a faint line invisible pixel by pixel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from streamgoggles.models.hough import HoughLines

    size = 96
    grid = HoughLines(size, size)
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2

    def band(theta_deg, rho, along=None):
        theta = np.deg2rad(theta_deg)
        on = np.abs(xx * np.cos(theta) + yy * np.sin(theta) - rho) < 1.0
        if along is not None:  # a segment: (centre, half-length) along the line
            position = -xx * np.sin(theta) + yy * np.cos(theta)
            on &= np.abs(position - along[0]) < along[1]
        return on.astype(float)

    point = (np.hypot(xx - 20, yy + 10) < 2).astype(float)
    level = 20.0
    rng = np.random.default_rng(3)
    counts = rng.poisson(level + 2.0 * band(60, 10)).astype(float)
    cases = [
        ("a line", band(30, 15), (30, 15)),
        ("a point (or a compact blob)", point, None),
        ("a short segment", band(120, -20, along=(10, 12)), (120, -20)),
        (
            "a faint line in noise, as an S/N map",
            (counts - level) / np.sqrt(level),
            (60, 10),
        ),
    ]
    extent = [grid.rhos[0], grid.rhos[-1], 0, 180]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    for column, (title, image, truth) in enumerate(cases):
        top, bottom = axes[0, column], axes[1, column]
        top.imshow(image, origin="lower", cmap="gray_r")
        top.set_title(title, fontsize=9)
        top.set_xticks([])
        top.set_yticks([])
        sums = np.where(grid.valid, grid(image), np.nan)
        bottom.imshow(
            sums, origin="lower", aspect="auto", extent=extent, cmap="viridis"
        )
        bottom.set_xlabel("ρ (pixels from the centre)", fontsize=8)
        if column == 0:
            bottom.set_ylabel("θ (degrees)", fontsize=8)
        if truth is not None:
            bottom.plot(
                truth[1], truth[0], "o", mfc="none", mec="#e07b39", ms=14, mew=1.5
            )
        if column == 3:
            t = round(truth[0] / 2) % grid.shape[0]
            r = int(np.argmin(np.abs(grid.rhos - truth[1])))
            noise = np.nanstd(sums)
            bottom.set_title(
                f"line sums: the line at {sums[t, r] / noise:.1f}σ\n"
                f"(one pixel: {2.0 / np.sqrt(level):.2f}σ)",
                fontsize=9,
            )
        elif column == 1:
            bottom.set_title(
                "line sums: every line through it,\na sinusoid", fontsize=9
            )
        elif column == 2:
            bottom.set_title("line sums: a peak spread in θ", fontsize=9)
        else:
            bottom.set_title("line sums: one peak, at its (θ, ρ)", fontsize=9)
        bottom.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(LINE_MODEL_FIGURES / "hough_toy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def line_model_architecture():
    """architecture.png: how each scorer is plugged, from window to answer,
    with the shapes in between."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    colours = {
        "data": ("#eeeeee", "the window's maps"),
        "net": ("#cfe0f3", "trained"),
        "fixed": ("#fbe3cf", "fixed operation"),
        "out": ("#d5ecd9", "the answer"),
    }
    rows = [
        (
            'per-pixel U-Net\nbuild_model(kind="unet")',
            [
                ("input\n7 × 96 × 96", "data"),
                ("U-Net\n(depth 2)", "net"),
                ("probability\nper pixel\n1 × 96 × 96", "out"),
            ],
        ),
        (
            'line model\nbuild_model(kind="hough")',
            [
                ("input\n7 × 96 × 96", "data"),
                ("U-Net backbone\n8 features, + the\ninput: 15 × 96 × 96", "net"),
                ("Hough: sums along\nevery line, / √n\n15 × 90 × 69", "fixed"),
                ("2 convolutions\nover (θ, ρ)", "net"),
                ("probability\nper line\n1 × 90 × 69", "out"),
            ],
        ),
        (
            "per-pixel U-Net\n+ line search\n(config with lines)",
            [
                ("input\n7 × 96 × 96", "data"),
                ("U-Net (as above)\nlogit − its median\n1 × 96 × 96", "net"),
                ("Hough: sums along\nevery line, / √n", "fixed"),
                ("score\nper line\n90 × 69", "out"),
            ],
        ),
        (
            "matched-filter\nline search\n(no network)",
            [
                ("counts at the\nqueried distance\n96 × 96", "data"),
                ("(counts − background)\n/ √background", "fixed"),
                ("Hough: sums along\nevery line, / √n", "fixed"),
                ("S/N\nper line\n90 × 69", "out"),
            ],
        ),
    ]
    width, height, step = 1.75, 0.95, 2.15
    fig, ax = plt.subplots(figsize=(15, 8.2))
    for row, (label, boxes) in enumerate(rows):
        y = -1.55 * row
        ax.text(-0.35, y, label, ha="right", va="center", fontsize=9, weight="bold")
        for i, (text, kind) in enumerate(boxes):
            x = i * step
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - height / 2),
                    width,
                    height,
                    boxstyle="round,pad=0.03,rounding_size=0.08",
                    fc=colours[kind][0],
                    ec="#7a7a7a",
                    lw=0.8,
                )
            )
            ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=8)
            if i:
                ax.annotate(
                    "",
                    xy=(x - 0.02, y),
                    xytext=(x - step + width + 0.02, y),
                    arrowprops={"arrowstyle": "->", "color": "#4d4d4d", "lw": 1.0},
                )
    for i, (colour, legend) in enumerate(colours.values()):
        ax.add_patch(
            FancyBboxPatch(
                (i * 2.6, 1.05),
                0.35,
                0.3,
                boxstyle="round,pad=0.02",
                fc=colour,
                ec="#7a7a7a",
                lw=0.8,
            )
        )
        ax.text(i * 2.6 + 0.45, 1.2, legend, va="center", fontsize=8)
    ax.text(
        0,
        -1.55 * len(rows) + 0.35,
        "input channels: 0-2 the isochrone filter at the queried distance − 0.5, at it, "
        "+ 0.5; 3 the decoy box; 4-6 constant maps of those three distances",
        fontsize=8,
        color="#4d4d4d",
    )
    ax.set_xlim(-3.2, 4 * step + width + 0.2)
    ax.set_ylim(-1.55 * len(rows) + 0.1, 1.6)
    ax.axis("off")
    fig.savefig(LINE_MODEL_FIGURES / "architecture.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def line_model_example(
    name, placement=0, train_sky="fold0", config="hough/band residual"
):
    """example_<name>.png: one copy of the evaluation -- a DES 2018 stream at
    its Table 1 parameters, at its ``placement``-th place on the evaluation
    sky -- in the search tile holding the longest stretch of it (the window
    the known-track test reads), followed through both networks: its counts,
    the S/N input, the per-pixel network's answer, the line network's best
    line back on the sky; the input's own line sums (the matched-filter line
    search), the lines along the track, the line network's answer over
    lines, and how it is read."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy import ndimage

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.models import build_model
    from streamgoggles.models.hough import hough_target, line_counts

    configure_torch_threads(num_workers=0)
    sp = real_des().stream_parameters_module()
    sky = window_level_sky(EVALUATED_ON[train_sky])
    pix = sky.pix
    streams = evaluation_streams(("DES 2018", "distance scan"))
    index = next(
        i for i, (kind, n, _) in enumerate(streams) if kind == "DES 2018" and n == name
    )
    stream = streams[index][2]
    query = min(sp.QUERY_GRID, key=lambda q: abs(q - stream["distance_modulus"]))
    chans = channels()
    good = next(
        i
        for i, c in enumerate(chans)
        if c["filter"] == "good" and c["distance_modulus"] == query
    )
    places = copy_places(index, stream, sky)
    for placed, ra, dec, rotation, band in places:
        if placed == placement:
            break
    injected = sky.injector.inject_streams_full_sky(
        [
            {
                "morphology": "uniform",
                "isochrone_model": sp.ISOCHRONE_FAMILY,
                "orientation": rotation,
                **stream,
            }
        ],
        np.random.default_rng([EVAL_SEED, index, placed]),
        centers=[(ra, dec)],
    )

    line_models = load_line_models(config, train_sky)
    grid = line_models[0].hough.grid
    pixel_models = []
    for s in PIXEL_CONFIGS["count/window x4"]["seeds"]:
        model = build_model(QueryDistanceTransform.n_channels, **sp.MODEL)
        model.load_state_dict(
            torch.load(
                pixel_model_stem("count/window", s, train_sky).with_suffix(".pt")
            )
        )
        pixel_models.append(model.eval())

    # the tile holding the longest straight stretch of the copy
    best, best_length = None, 0.0
    for i, px in enumerate(sky.tile_pixels):
        if band[px].any():
            projection = sky.projections[i]
            inside = projection.image(band[projection.pixnums].astype(float)) > 0.5
            length = line_counts(inside, grid).max()
            if length > best_length:
                best, best_length, track = projection, length, inside
    valid_at = sky.valid[best.pixnums]
    stack = np.stack(
        [
            best.image(np.where(valid_at, m[best.pixnums], 0.0))
            for m in injected["map_full"]
        ]
    ).astype(np.float32)
    sample = {
        "map_stack": stack,
        "label_stack": np.zeros_like(stack),
        "valid_mask": best.valid,
        "params": {},
        "metadata": {"channels": chans},
    }

    def view(normalization):
        return QueryDistanceTransform(
            StreamMapTransform(
                normalizer=normalizer(normalization, chans), augment=False
            ),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=query,
        )(dict(sample))["map_stack"]

    snr, standardized = view("residual"), view("window")
    with torch.no_grad():
        x = torch.as_tensor(snr)[None]
        lines = np.mean([torch.sigmoid(m(x))[0, 0].numpy() for m in line_models], 0)
        x = torch.as_tensor(standardized)[None]
        pixels = np.mean([m(x)[0, 0].numpy() for m in pixel_models], 0)
    lines[~grid.valid] = np.nan
    target = hough_target(track, grid)
    along = np.nanmax(np.where(target > 0, lines, np.nan))
    matched = np.where(grid.valid, grid(snr[1]), np.nan)
    t, r = np.unravel_index(np.nanargmax(lines), lines.shape)
    size = pix.image_size_pix[0]
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2
    theta, rho = grid.thetas[t], grid.rhos[r]
    best_line = np.abs(xx * np.cos(theta) + yy * np.sin(theta) - rho) < 1.0
    thresholds = pd.read_csv(
        result_dir(train_sky) / f"hough_null_{config.replace('/', '_')}.csv"
    )
    level = float(
        thresholds[
            (thresholds.scorer == "network") & np.isclose(thresholds["query"], query)
        ].threshold.iloc[0]
    )
    valid = best.valid
    extent = [grid.rhos[0], grid.rhos[-1], 0, 180]

    def on_sky(ax, image, title, cmap="gray_r", vmin=None, vmax=None):
        ax.imshow(
            np.where(valid, image, np.nan),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.contour(
            track, levels=[0.5], colors="#e07b39", linewidths=0.7, linestyles="--"
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    def over_lines(ax, values, title):
        ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.contour(
            target, levels=[0.5], colors="#e07b39", linewidths=0.8, extent=extent
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ρ (pixels)", fontsize=8)
        ax.set_ylabel("θ (degrees)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    on_sky(
        axes[0, 0],
        ndimage.gaussian_filter(stack[good], 1.5),
        f"counts at m−M {query:g} (smoothed for the eye)",
    )
    snr_image = ndimage.gaussian_filter(snr[1], 1.5)
    on_sky(axes[0, 1], snr_image, "S/N input", cmap="RdBu_r", vmin=-1, vmax=1)
    on_sky(
        axes[0, 2],
        pixels,
        f"per-pixel network (4 models): max {pixels[valid].max():.2f}",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    on_sky(axes[0, 3], snr_image, "", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 3].contour(best_line, levels=[0.5], colors="#1f6fb4", linewidths=1.2)
    axes[0, 3].set_title(
        f"the line network's best line (blue): p = {np.nanmax(lines):.2f}", fontsize=9
    )
    over_lines(axes[1, 0], matched, "the S/N input's own line sums")
    over_lines(axes[1, 1], target, "the lines along the copy's track")
    over_lines(axes[1, 2], lines, "the line network: probability per line")
    axes[1, 2].plot(rho, np.rad2deg(theta), "x", color="white", ms=8, mew=1.5)
    axes[1, 3].axis("off")
    verdict = "found" if along >= level else "missed"
    axes[1, 3].text(
        0.0,
        0.97,
        f"{name}, DES 2018 parameters: m−M {stream['distance_modulus']:g},\n"
        f"width {stream['width']:g}°, length {stream['length']:g}°, "
        f"{stream['richness']:g} mag/arcsec²;\n"
        f"its place {placement + 1} on fold 1; queried at m−M {query:g}\n\n"
        f"best line along the track: p = {along:.2f}\n"
        f"1% level of stream-free windows: {level:.2f}\n"
        f"→ {verdict} without the track, in this window\n\n"
        "orange, dashed: the copy's band\n"
        "orange contour over (θ, ρ): its lines\n"
        "white cross: the line network's best line",
        va="top",
        fontsize=9,
        transform=axes[1, 3].transAxes,
    )
    fig.tight_layout()
    stem = name.lower().replace(" ", "_")
    fig.savefig(
        LINE_MODEL_FIGURES / f"example_{stem}.png", dpi=100, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"{name}: along the track p = {along:.2f} (level {level:.2f}, {verdict}); "
        f"best line p = {np.nanmax(lines):.2f}, on the track: {bool(target[t, r])}; "
        f"per-pixel max {pixels[valid].max():.2f}",
        flush=True,
    )


LINE_SKY = OUT / "line_sky"
LINE_SKY_CONFIG = "hough/band residual"
# A detected line runs along a DES 2018 track if at least MIN_ALONG_DEG of it
# lies within max(TRACK_TOLERANCE_DEG, two widths) of the track.
TRACK_TOLERANCE_DEG = 1.0
MIN_ALONG_DEG = 3.0
# ... and runs along it: its direction within TRACK_ALIGN_DEG of the track's
# where they meet -- a line crossing a track at an angle is not following it
TRACK_ALIGN_DEG = 15.0


def _tangents(points):
    """Unit tangents of a polyline of unit vectors (central differences)."""
    import numpy as np

    tangents = np.gradient(points, axis=0)
    tangents -= (tangents * points).sum(1, keepdims=True) * points
    return tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)


def _aligned_length(
    points, step, reference, tolerance_deg, tangents=None, reference_tangents=None
):
    """Degrees of a polyline (unit vectors, ``step`` degrees apart) that lie
    within ``tolerance_deg`` of a reference polyline AND run along it: their
    directions within TRACK_ALIGN_DEG of the reference's nearest point's."""
    import numpy as np

    if tangents is None:
        tangents = _tangents(points)
    if reference_tangents is None:
        reference_tangents = _tangents(reference)
    cosines = points @ reference.T
    nearest = cosines.argmax(1)
    close = cosines[np.arange(len(points)), nearest] >= np.cos(
        np.radians(tolerance_deg)
    )
    aligned = np.abs((tangents * reference_tangents[nearest]).sum(1)) >= np.cos(
        np.radians(TRACK_ALIGN_DEG)
    )
    return float((close & aligned).sum() * step)


# Objects marked on the maps, with the radius within which a line counts as
# theirs: the Magellanic Clouds' outskirts and the two bright dwarf
# spheroidals, whose stars past their masks every line through them crosses
LINE_SKY_OBJECTS = {
    "LMC": (80.89, -69.76, 20.0),
    "SMC": (13.19, -72.83, 12.0),  # the calibration's disc (run.py)
    "Fornax dSph": (40.0, -34.45, 4.0),
    "Sculptor dSph": (15.04, -33.71, 4.0),
}
# The panels of the maps: distance ranges, as the queried distances in each
DISTANCE_RANGES = {
    "m−M 15-16": (15.0, 15.5, 16.0),
    "m−M 16.5-17.5": (16.5, 17.0, 17.5),
    "m−M 18-19": (18.0, 18.5, 19.0),
}


def _unit(ra, dec):
    import numpy as np

    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1
    )


def _arc(ra1, dec1, ra2, dec2, n):
    """n points along the great circle between two positions, (ra, dec) deg."""
    import numpy as np

    a, b = _unit(ra1, dec1), _unit(ra2, dec2)
    angle = np.arccos(np.clip(a @ b, -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    if angle < 1e-9:
        points = np.repeat(a[None], n, 0)
    else:
        points = (np.sin((1 - t) * angle) * a + np.sin(t * angle) * b) / np.sin(angle)
    return np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360, np.degrees(
        np.arcsin(np.clip(points[:, 2], -1, 1))
    )


def _suffix(mask_objects):
    """The search mask's tag in file names: "" for the object mask
    (``mask_objects`` True), "__unmasked" (False), "__tight" (``"tight"``,
    the bright dwarfs masked to LINE_SKY_DWARF_MAX_DEG at most)."""
    if mask_objects == "tight":
        return "__tight"
    return "" if mask_objects else "__unmasked"


def _figure_tag(config):
    """ "" for LINE_SKY_CONFIG's figures, "_<label>[-<variant>]" for another
    model's ("hough/band2 residual" -> "_band2", "hough/band2 residual long
    x4" -> "_band2-long-x4"), so each set stays side by side."""
    if config == LINE_SKY_CONFIG:
        return ""
    label, *variant = config.split("/")[1].split()
    return "_" + "-".join([label, *(v for v in variant if v != "residual")])


def inference_sky(config, mask_objects=True):
    """The DES inference sky as the sky search sees it, a namespace:
    background, pix, nside; valid (the valid sky, less the bright objects'
    outskirts with ``mask_objects``: `object_mask` with dwarfs brighter than
    LINE_SKY_DWARF_MV); tiles (the search's windows, half overlapping, over
    the inference footprint); queries; maps (the channels, full-sky); smooth
    ({query: the smooth local background}); grid; score ({train sky: the
    `line_scorer` of the configuration's models trained there}); levels and
    half ({(train sky, scorer, query): the level the best line of 1% of the
    models' stream-free windows reaches, and of 0.5%}); and combination (the
    two searches' false-alarm rate together, each at half its rate)."""
    import types

    import numpy as np
    import pandas as pd

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.objects_overlap import get_footprint
    from streamgoggles.windows import tile_footprint

    configure_torch_threads(num_workers=0)
    rd = real_des()
    sp = rd.stream_parameters_module()
    background, _, pix = build_sky(
        "inference", "count", CONFIGS[config].get("image_pix", IMAGE_PIX)
    )
    nside = pix.nside
    valid = background.valid_mask_full
    if mask_objects:
        valid = valid & ~rd.object_mask(
            nside,
            max_dwarf_mv=LINE_SKY_DWARF_MV,
            max_radius_deg=LINE_SKY_DWARF_MAX_DEG if mask_objects == "tight" else None,
        )
    usable = get_footprint("des_yr6_inference", nside=nside)[0]
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        usable & valid, nside, tile_size_deg=window_deg, stride_deg=window_deg / 2
    )
    name = config.replace("/", "_")
    score, levels = {}, {}
    for train_sky in ("fold0", "fold1"):
        models, grid = scoring_models(config, train_sky)
        score[train_sky] = line_scorer(config, models, grid)
        null = pd.read_csv(result_dir(train_sky) / f"hough_null_{name}.csv")
        for row in null.itertuples():
            levels[(train_sky, row.scorer, round(row.query, 1))] = row.threshold
    # the combination: each search at half the false-alarm rate, and what the
    # two give together on the same stream-free windows
    half, combination = {}, []
    for train_sky in ("fold0", "fold1"):
        best = np.load(result_dir(train_sky) / f"hough_null_best_{name}.npz")
        for q in sorted({float(k.split("|")[1]) for k in best.files}):
            together = np.zeros(len(best[f"network|{q:.1f}"]), bool)
            for scorer in ("network", "matched filter"):
                values = best[f"{scorer}|{q:.1f}"]
                half[(train_sky, scorer, round(q, 1))] = float(
                    np.quantile(values, 1 - HOUGH_FALSE_ALARM / 2)
                )
                together |= values >= half[(train_sky, scorer, round(q, 1))]
            combination.append(
                {"models": train_sky, "query": q, "false_alarm_rate": together.mean()}
            )
    queries = list(sp.QUERY_GRID)
    return types.SimpleNamespace(
        background=background,
        pix=pix,
        nside=nside,
        valid=valid,
        tiles=tiles,
        queries=queries,
        maps=[
            background.raw_map_full_dict[c["filter"]][c["distance_modulus"]]
            for c in channels()
        ],
        smooth=smooth_backgrounds(background, queries, valid),
        grid=grid,
        score=score,
        levels=levels,
        half=half,
        combination=combination,
    )


def line_sky(config=LINE_SKY_CONFIG, mask_objects=True):
    """The line model over the whole DES inference sky -- the known streams
    in it -- and the matched filter's line search alongside.

    The search's tiles (11-degree windows, half overlapping) cover the
    inference sky. In each tile and at each queried distance, every line is
    scored by the models trained on the fold the tile's centre is not in, so
    no model judges sky it trained on, and by the matched filter: this sky's
    counts minus their smooth local background, over its square root, summed
    along the line. A line is detected when it is a peak over (theta, rho)
    (the highest within 5 x 5 cells) and scores above the level that 1% of
    stream-free windows reach on the calibration sky of the tile's fold
    (`evaluate_hough` of the scoring models). Each detection is the stretch
    of its line over valid sky in the tile, a segment on the sky.

    With ``mask_objects`` (the default), the compact objects whose outskirts
    the training mask leaves -- dwarfs brighter than LINE_SKY_DWARF_MV to 12
    half-light radii, globular clusters to 2 degrees (`object_mask`) -- are
    masked from the search sky too: every line through one is bright.
    Each detection also records ``level_half``, its scorer's level at half
    the false-alarm rate (0.5% of stream-free windows): the combination of
    the two searches keeps either's lines above it, for about 1% together.
    Writes line_sky/detections_<config>.csv (``__unmasked`` without the
    object mask) and the combination's levels and false-alarm rates.
    """
    import dataclasses

    import numpy as np
    import pandas as pd
    from scipy import ndimage

    from streamgoggles.matched_filter import WindowProjection, _tangent_plane_radec
    from streamgoggles.objects_overlap import spatial_fold

    rd = real_des()
    sky = inference_sky(config, mask_objects)
    pix, valid, grid, tiles = sky.pix, sky.valid, sky.grid, sky.tiles
    name = config.replace("/", "_")
    size = pix.image_size_pix[0]
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2
    print(f"{len(tiles)} tiles", flush=True)
    rows = []
    start = time.time()
    for number, tile in enumerate(tiles):
        fold = int(
            spatial_fold(np.array([tile.center_ra]), rd.STRIPE_DEG, rd.N_FOLDS)[0]
        )
        scoring = f"fold{1 - fold}"  # the models that never trained on this fold
        projection = WindowProjection.for_window(tile, pix, valid)
        ra_grid, dec_grid = _tangent_plane_radec(
            dataclasses.replace(
                pix,
                center_ra=tile.center_ra,
                center_dec=tile.center_dec,
                rotation_deg=tile.rotation_deg,
            )
        )
        for q in sky.queries:
            answers = sky.score[scoring](
                sky.maps, valid, sky.smooth[q], [projection], q
            )
            for scorer, values in answers.items():
                values = np.where(grid.valid, values[0], -np.inf)
                level = sky.levels[(scoring, scorer, round(q, 1))]
                peaks = (values == ndimage.maximum_filter(values, size=5)) & (
                    values >= level
                )
                for t, r in zip(*np.nonzero(peaks), strict=True):
                    theta, rho = grid.thetas[t], grid.rhos[r]
                    line = grid.line_mask(t, r)  # within its part of the grid
                    on = line & projection.valid
                    if on.sum() < 2:
                        continue
                    along = (-xx * np.sin(theta) + yy * np.cos(theta))[on]
                    first, last = np.argmin(along), np.argmax(along)
                    rows.append(
                        {
                            "scorer": scorer,
                            "tile": number,
                            "tile_ra": tile.center_ra,
                            "tile_dec": tile.center_dec,
                            "fold": fold,
                            "models": scoring,
                            "query": q,
                            "theta_deg": float(np.rad2deg(theta)),
                            "rho_pix": float(rho),
                            "part": next(
                                k
                                for k, c in enumerate(grid.parts)
                                if c.start <= r < c.stop
                            ),
                            "score": float(values[t, r]),
                            "level": float(level),
                            "level_half": sky.half[(scoring, scorer, round(q, 1))],
                            "ra1": float(ra_grid[on][first]),
                            "dec1": float(dec_grid[on][first]),
                            "ra2": float(ra_grid[on][last]),
                            "dec2": float(dec_grid[on][last]),
                            "length_deg": float(
                                (along[last] - along[first]) * pix.pixel_scale_deg
                            ),
                            "valid_share": float(on.sum() / line.sum()),
                        }
                    )
        if (number + 1) % 25 == 0:
            print(
                f"  {number + 1}/{len(tiles)} tiles, {time.time() - start:.0f}s",
                flush=True,
            )
    LINE_SKY.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(
        LINE_SKY / f"detections_{name}{_suffix(mask_objects)}.csv", index=False
    )
    combination = pd.DataFrame(sky.combination)
    combination.to_csv(LINE_SKY / f"combination_{name}.csv", index=False)
    print(
        "combination, stream-free windows with a line from either search: "
        f"{combination.false_alarm_rate.min():.2%}-{combination.false_alarm_rate.max():.2%}",
        flush=True,
    )
    print(
        table.groupby(["scorer", "query"]).size().unstack("scorer").to_string(),
        flush=True,
    )


# The search masks the bright dwarfs' outskirts and the globular clusters
# (`object_mask`): every line through them is bright. Not the ultra-faint
# dwarfs, which make no bursts and some of which lie on streams.
LINE_SKY_DWARF_MV = -8.0
# ... to at most this radius with the "tight" mask: Fornax's and Sculptor's
# stars stand out in the matched filter to 1.5 degrees (+5-33% and +12-18% at
# 1-1.5 degrees, nothing beyond), where 12 half-light radii reach 4.0 and 2.2
# -- Fornax's 4 degrees hid 59% of Aliqa Uma's band
LINE_SKY_DWARF_MAX_DEG = 2.0
# The scorers of the sky maps: each search at its own 1% level, and their
# combination -- either's lines above its level at half that rate, about 1%
# of stream-free windows together (line_sky/combination_<config>.csv)
LINE_SKY_SCORERS = ("network", "matched filter", "combined")


def _scorers(table):
    """The scorers a detections table holds (the combination needs the
    half-rate levels, which the unmasked run did not record)."""
    return LINE_SKY_SCORERS if "level_half" in table else LINE_SKY_SCORERS[:2]


def _chosen(table, scorer):
    """Which detections belong to a scorer of the sky maps (a boolean Series)."""
    if scorer == "combined":
        return table.score >= table.level_half
    return table.scorer == scorer


def _level(table, scorer):
    return table.level_half if scorer == "combined" else table.level


def line_sky_matches(config=LINE_SKY_CONFIG, mask_objects=True):
    """Which DES 2018 streams have a detected line along their track at
    their distance (the queried distance nearest it): a segment with at
    least MIN_ALONG_DEG of it within max(TRACK_TOLERANCE_DEG, two widths) of
    the track as the paper draws it, and running along it (directions within
    TRACK_ALIGN_DEG). With the per-pixel network's result
    (the first training, run.py detect) alongside. Writes
    line_sky/matches_<config>.csv, and marks each detection with the DES 2018
    stream it runs along, if any, in the detections file."""
    import numpy as np
    import pandas as pd

    from streamgoggles.objects_overlap import des2018_arc

    rd = real_des()
    sp = rd.stream_parameters_module()
    name = config.replace("/", "_") + _suffix(mask_objects)
    table = pd.read_csv(LINE_SKY / f"detections_{name}.csv")
    scorers = _scorers(table)
    segments = [
        _unit(*_arc(r.ra1, r.dec1, r.ra2, r.dec2, max(int(r.length_deg / 0.1), 2)))
        for r in table.itertuples()
    ]
    step = np.array(
        [r.length_deg / max(int(r.length_deg / 0.1) - 1, 1) for r in table.itertuples()]
    )
    segment_tangents = [_tangents(seg) for seg in segments]
    pixel_test = pd.read_csv(rd.DETECTIONS)
    table["along"] = ""
    rows = []
    for stream, (width, _, distance, _) in sp.DES_STREAMS.items():
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
        track = _unit(*des2018_arc(stream, n=600))
        track_tangents = _tangents(track)
        tolerance = max(TRACK_TOLERANCE_DEG, 2 * width)
        along = np.array(
            [
                _aligned_length(seg, s, track, tolerance, tangents, track_tangents)
                for seg, s, tangents in zip(
                    segments, step, segment_tangents, strict=True
                )
            ]
        )
        runs_along = along >= MIN_ALONG_DEG
        table.loc[runs_along, "along"] = np.where(
            table.loc[runs_along, "along"] == "",
            stream,
            table.loc[runs_along, "along"] + "; " + stream,
        )
        row = {"stream": stream, "distance_modulus": distance, "query": query}
        for scorer in scorers:
            hits = (
                runs_along & _chosen(table, scorer) & np.isclose(table["query"], query)
            )
            row[f"{scorer} found"] = bool(hits.any())
            row[f"{scorer} lines"] = int(hits.sum())
            row[f"{scorer} best"] = (
                float((table.score / _level(table, scorer))[hits].max())
                if hits.any()
                else np.nan
            )
        pixel = pixel_test[
            (pixel_test.stream == stream) & np.isclose(pixel_test["query"], query)
        ]
        row["per-pixel found"] = bool(pixel.detected.any()) if len(pixel) else np.nan
        rows.append(row)
    # the other lines: around a Magellanic Cloud or a bright dwarf, or elsewhere
    middle = _unit((table.ra1 + table.ra2) / 2, (table.dec1 + table.dec2) / 2)
    ends = (_unit(table.ra1, table.dec1), _unit(table.ra2, table.dec2))
    table["near"] = ""
    for obj, (ra, dec, radius) in LINE_SKY_OBJECTS.items():
        centre = _unit(ra, dec)
        closest = np.stack([middle @ centre, ends[0] @ centre, ends[1] @ centre]).max(0)
        close = np.degrees(np.arccos(np.clip(closest, -1, 1))) <= radius
        table.loc[close & (table.near == ""), "near"] = obj
    table["kind"] = np.where(
        table.along != "",
        "along a DES 2018 track",
        np.where(table.near != "", "around " + table.near, "elsewhere"),
    )
    matches = pd.DataFrame(rows)
    matches.to_csv(LINE_SKY / f"matches_{name}.csv", index=False)
    table.to_csv(LINE_SKY / f"detections_{name}.csv", index=False)
    print(
        pd.DataFrame(
            {
                scorer: table[_chosen(table, scorer)].kind.value_counts()
                for scorer in scorers
            }
        )
        .fillna(0)
        .astype(int),
        flush=True,
    )
    print(matches.to_string(index=False), flush=True)
    for scorer in scorers:
        mine = table[_chosen(table, scorer)]
        print(
            f"{scorer}: {len(mine)} lines, {(mine.along != '').sum()} along a DES 2018 "
            f"track (any distance); streams found at their distance: "
            f"{int(matches[f'{scorer} found'].sum())} of {len(matches)}",
            flush=True,
        )


def line_sky_figures(config=LINE_SKY_CONFIG, mask_objects=True):
    """line_sky_<scorer>.png: the detected lines over the DES sky, one panel
    per range of distance, coloured by the queried distance, over the
    inference footprint, with the DES 2018 tracks of the streams in that
    range (grey, named)."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from streamgoggles.objects_overlap import des2018_arc, get_footprint

    sp = real_des().stream_parameters_module()
    suffix = _suffix(mask_objects)
    name = config.replace("/", "_") + suffix
    table = pd.read_csv(LINE_SKY / f"detections_{name}.csv")
    footprint = get_footprint("des_yr6_inference", nside=64)[0]
    fra, fdec = hp.pix2ang(64, np.flatnonzero(footprint), lonlat=True)

    def wrap(ra):
        return np.where(np.asarray(ra) > 180, np.asarray(ra) - 360, ra)

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(15.0, 19.0)
    titles = {
        "network": "line network (S/N inputs), out of fold",
        "matched filter": "matched-filter line search"
        + (", dwarfs and clusters masked" if mask_objects else ""),
        "combined": "both searches, each at half its false-alarm rate "
        "(solid: line network; dashed: matched filter)",
    }
    for scorer in _scorers(table):
        title = titles[scorer]
        mine = table[_chosen(table, scorer)]
        fig, axes = plt.subplots(
            len(DISTANCE_RANGES),
            1,
            figsize=(12, 4.6 * len(DISTANCE_RANGES)),
            layout="constrained",
        )
        for ax, (label, queries) in zip(axes, DISTANCE_RANGES.items(), strict=True):
            ax.scatter(wrap(fra), fdec, s=6, marker="s", color="#efefef", lw=0)
            for stream, (_, _, distance, _) in sp.DES_STREAMS.items():
                nearest = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
                if not any(np.isclose(nearest, q) for q in queries):
                    continue
                ra, dec = des2018_arc(stream, n=200)
                ax.plot(
                    wrap(ra),
                    dec,
                    color="#9e9e9e",
                    lw=5,
                    alpha=0.6,
                    solid_capstyle="round",
                )
                ax.text(
                    wrap(ra[len(ra) // 2]) + 1.5,
                    dec[len(dec) // 2] + 1.0,
                    stream,
                    fontsize=8,
                    color="#4d4d4d",
                )
            for obj, (ra, dec, _) in LINE_SKY_OBJECTS.items():
                shown = max(dec, -68.5)  # the Clouds' centres lie past the edge
                ax.plot(wrap(ra), shown, "+", color="#c0392b", ms=9, mew=1.5)
                ax.text(wrap(ra) - 1.5, shown + 1.0, obj, fontsize=8, color="#c0392b")
            lines = mine[np.isin(np.round(mine["query"], 1), queries)]
            for r in lines.itertuples():
                ra, dec = _arc(r.ra1, r.dec1, r.ra2, r.dec2, 20)
                style = "--" if r.scorer == "matched filter" else "-"
                ax.plot(wrap(ra), dec, style, color=cmap(norm(r.query)), lw=1.2)
            ax.set_title(f"{label}: {len(lines)} lines", fontsize=10)
            ax.set_xlim(112, -65)
            ax.set_ylim(-70, 8)
            ax.set_aspect(1 / np.cos(np.radians(35)))
            ax.set_xlabel("RA (deg)", fontsize=9)
            ax.set_ylabel("Dec (deg)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            fraction=0.02,
            pad=0.01,
            label="queried distance modulus",
        )
        level = "half its level" if scorer == "combined" else "the 1% level"
        fig.suptitle(
            f"{title}: lines above {level} of stream-free windows", fontsize=11
        )
        stem = scorer.replace(" ", "_")
        fig.savefig(
            DOC_FIGURES / f"line_sky_{stem}{suffix}{_figure_tag(config)}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)


LINE_SKY_CHANCE_TRACKS = 200
LINE_SKY_CHANCE_SEED = 2029


def line_sky_chance(
    config=LINE_SKY_CONFIG, mask_objects=True, n_random=LINE_SKY_CHANCE_TRACKS
):
    """How often each stream would be "found" by chance: its track, as a
    great circle of its length, at n_random random places and position
    angles on the inference footprint (at least 90% of it on the footprint),
    matched to the same detections at the same queried distance, by the same
    rule as `line_sky_matches`. A long, wide track is crossed by a stray line
    often, so what tells a stream from chance is how many lines run along
    it: "<scorer> chance" is the share of random tracks with at least one
    line along them, "<scorer> p" the share with at least as many as the
    stream itself (from n_random + 1, the stream counted). Adds both to
    line_sky/matches_<config>.csv."""
    import healpy as hp
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    from streamgoggles.objects_overlap import get_footprint

    rd = real_des()
    sp = rd.stream_parameters_module()
    name = config.replace("/", "_") + _suffix(mask_objects)
    table = pd.read_csv(LINE_SKY / f"detections_{name}.csv")
    matches = pd.read_csv(LINE_SKY / f"matches_{name}.csv")
    footprint = get_footprint("des_yr6_inference", nside=64)[0]
    candidates = np.flatnonzero(footprint)
    rng = np.random.default_rng(LINE_SKY_CHANCE_SEED)
    scorers = _scorers(table)
    for scorer in scorers:
        rates, p_values = [], []
        # in the order of `line_sky_matches`'s rows: DES_STREAMS's
        for width, length, distance, _ in sp.DES_STREAMS.values():
            query = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
            mine = table[_chosen(table, scorer) & np.isclose(table["query"], query)]
            if mine.empty:
                rates.append(0.0)
                p_values.append(np.nan)
                continue
            points, owner, steps = [], [], []
            for k, r in enumerate(mine.itertuples()):
                n = max(int(r.length_deg / 0.1), 2)
                points.append(_unit(*_arc(r.ra1, r.dec1, r.ra2, r.dec2, n)))
                owner.append(np.full(n, k))
                steps.append(r.length_deg / max(n - 1, 1))
            points, owner, steps = (
                np.concatenate(points),
                np.concatenate(owner),
                np.array(steps),
            )
            tree = cKDTree(points)
            # each segment's great circle, to keep only those along the track
            poles = []
            for r in mine.itertuples():
                a, b = _unit(r.ra1, r.dec1), _unit(r.ra2, r.dec2)
                pole = np.cross(a, b)
                poles.append(pole / max(np.linalg.norm(pole), 1e-12))
            poles = np.array(poles)
            chord = 2 * np.sin(np.radians(max(TRACK_TOLERANCE_DEG, 2 * width)) / 2)
            counts = []
            placed = 0
            while placed < n_random:
                ra, dec = hp.pix2ang(64, int(rng.choice(candidates)), lonlat=True)
                track = rd.great_circle(
                    float(ra), float(dec), float(rng.uniform(0, 360)), length, n=150
                )
                inside = footprint[hp.ang2pix(64, track[0], track[1], lonlat=True)]
                if inside.mean() < 0.9:
                    continue
                placed += 1
                track_points = _unit(*track)
                near = tree.query_ball_point(track_points, chord)
                hits = np.unique(np.concatenate([np.asarray(h, int) for h in near]))
                along = np.bincount(owner[hits], minlength=len(steps)) * steps
                track_pole = np.cross(track_points[0], track_points[-1])
                track_pole /= max(np.linalg.norm(track_pole), 1e-12)
                aligned = np.abs(poles @ track_pole) >= np.cos(
                    np.radians(TRACK_ALIGN_DEG)
                )
                counts.append(int(((along >= MIN_ALONG_DEG) & aligned).sum()))
            counts = np.array(counts)
            observed = int(matches.loc[len(rates), f"{scorer} lines"])
            rates.append(float((counts >= 1).mean()))
            p_values.append(
                (1 + (counts >= observed).sum()) / (1 + n_random)
                if observed
                else np.nan
            )
        matches[f"{scorer} chance"] = rates
        matches[f"{scorer} p"] = p_values
    matches.to_csv(LINE_SKY / f"matches_{name}.csv", index=False)
    columns = ["stream"]
    for scorer in scorers:
        columns += [f"{scorer} lines", f"{scorer} p"]
    print(matches[[*columns, "per-pixel found"]].to_string(index=False), flush=True)
    for scorer in scorers:
        print(
            f"{scorer}: found {int(matches[f'{scorer} found'].sum())}, expected by chance "
            f"{matches[f'{scorer} chance'].sum():.1f}",
            flush=True,
        )


LINE_SKY_SIGNIFICANCE = 0.05  # a found stream counts if chance does as well this rarely


def line_sky_summary(config=LINE_SKY_CONFIG, mask_objects=True):
    """line_sky_streams.png: the fourteen DES 2018 streams, nearest first,
    and which method finds them at their distance: the per-pixel network
    (the first training's test), the line network, the matched filter's line
    search, and either line search. For the line searches, filled: lines
    along the track, more than random tracks of its shape get but in a
    fraction LINE_SKY_SIGNIFICANCE of places (`line_sky_chance`); tinted:
    lines along it, but no more than chance gives; open: none."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    suffix = _suffix(mask_objects)
    matches = pd.read_csv(LINE_SKY / f"matches_{config.replace('/', '_')}{suffix}.csv")
    matches = matches.sort_values("distance_modulus").reset_index(drop=True)
    scorers = [s for s in LINE_SKY_SCORERS if f"{s} found" in matches]
    state = {}
    for scorer in scorers:
        significant = matches[f"{scorer} found"] & (
            matches[f"{scorer} p"] <= LINE_SKY_SIGNIFICANCE
        )
        state[scorer] = np.where(
            significant,
            "significant",
            np.where(matches[f"{scorer} found"], "chance", "none"),
        )
    state["per-pixel"] = np.where(
        matches["per-pixel found"].astype(bool), "significant", "none"
    )
    # either search at its own 1% level: about 2% false lines together
    state["either"] = np.where(
        (state["network"] == "significant")
        | (state["matched filter"] == "significant"),
        "significant",
        np.where(
            (state["network"] == "chance") | (state["matched filter"] == "chance"),
            "chance",
            "none",
        ),
    )
    columns = {
        "per-pixel": ("per-pixel network\n(first training)", "#4d4d4d"),
        "network": ("line network", "#2e8b57"),
        "matched filter": (
            "matched-filter\nline search"
            + ("\n(objects masked)" if mask_objects else ""),
            "#e07b39",
        ),
        "combined": ("both, at half\nthe rate each\n(~1% together)", "#1f6fb4"),
        "either": ("either, at its\nown 1%\n(~2% together)", "#8e44ad"),
    }
    columns = {k: v for k, v in columns.items() if k in state}
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    for x, (key, (label, colour)) in enumerate(columns.items()):
        faces = {"significant": colour, "chance": colour, "none": "white"}
        alphas = {"significant": 1.0, "chance": 0.3, "none": 1.0}
        for y, value in enumerate(state[key]):
            ax.scatter(
                x,
                y,
                s=140,
                marker="o",
                facecolors=faces[value],
                edgecolors=colour,
                linewidths=1.5,
                alpha=alphas[value],
            )
        ax.text(
            x,
            -1.1,
            f"{int((state[key] == 'significant').sum())} of {len(matches)}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label for label, _ in columns.values()], fontsize=9)
    ax.set_yticks(matches.index)
    ax.set_yticklabels(
        [
            f"{s}  (m−M {d:g})"
            for s, d in zip(matches.stream, matches.distance_modulus, strict=True)
        ],
        fontsize=9,
    )
    ax.set_ylim(len(matches) - 0.4, -1.6)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title(
        "DES 2018 streams found along their track, at their distance\n"
        "filled: found (lines: beyond chance, p ≤ 0.05); tinted: as chance would; open: not",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(
        DOC_FIGURES / f"line_sky_streams{suffix}{_figure_tag(config)}.png",
        dpi=110,
        bbox_inches="tight",
    )
    plt.close(fig)


# Segments of one track: great circles within TRACK_LINK_POLE_DEG of each
# other, coming within TRACK_LINK_GAP_DEG on the sky, at queried distances at
# most TRACK_LINK_QUERY apart.
TRACK_LINK_POLE_DEG = 5.0
TRACK_LINK_GAP_DEG = 1.0
TRACK_LINK_QUERY = 0.5
GALSTREAMS_TOLERANCE_DEG = 1.5  # a track is a known stream's within this, over 3 deg
# a track runs along the footprint's edge if this share of it lies within
# TRACK_EDGE_DEG of sky outside the footprint
TRACK_EDGE_DEG = 1.5
TRACK_EDGE_SHARE = 0.6


def _known_tracks():
    """{name: (ra, dec)} of every stream galstreams traces near the DES
    footprint (its measured tracks, read from its files; `stream_tracks`)."""
    import galstreams
    import numpy as np
    from astropy.table import Table

    tracks = {}
    for path in sorted(
        (Path(galstreams.__file__).parent / "tracks").glob("track.st.*.ecsv")
    ):
        if path.name.endswith(".summary.ecsv"):
            continue
        table = Table.read(path)
        ra = np.asarray(table["ra"].value, float)
        dec = np.asarray(table["dec"].value, float)
        wrapped = np.where(ra > 180, ra - 360, ra)
        if ((dec < 10) & (wrapped > -70) & (wrapped < 115)).any():
            tracks[path.name[len("track.st.") : -len(".ecsv")]] = (ra, dec)
    return tracks


def line_sky_tracks(config=LINE_SKY_CONFIG, mask_objects=True, scorer="combined"):
    """The detected segments joined into tracks: a catalogue of candidates.

    Segments of ``scorer``'s detections (by default the combination, both
    searches at half their false-alarm rate) are joined when their great
    circles agree within TRACK_LINK_POLE_DEG, they come within
    TRACK_LINK_GAP_DEG of each other, and their queried distances differ by
    at most TRACK_LINK_QUERY -- the same structure seen by overlapping tiles
    and neighbouring distances. Each track: the great circle through its
    segments (the plane of least scatter), its extent along it, its mean
    distance (weighted by how far each segment stands above its level) and
    range, its number of segments, tiles and sources, and an identification
    -- a DES 2018 stream at its distance, another stream of galstreams, the
    periphery of a Magellanic Cloud, or none. Writes
    line_sky/tracks_<config>.csv.
    """
    import numpy as np
    import pandas as pd

    from streamgoggles.objects_overlap import des2018_arc

    sp = real_des().stream_parameters_module()
    name = config.replace("/", "_") + _suffix(mask_objects)
    table = pd.read_csv(LINE_SKY / f"detections_{name}.csv")
    table = table[_chosen(table, scorer)].reset_index(drop=True)
    table["ratio"] = table.score / _level(table, scorer)
    points = [
        _unit(*_arc(r.ra1, r.dec1, r.ra2, r.dec2, max(int(r.length_deg / 0.25), 2)))
        for r in table.itertuples()
    ]
    ends = [(p[0], p[-1]) for p in points]
    poles = np.array([np.cross(a, b) / np.linalg.norm(np.cross(a, b)) for a, b in ends])
    middles = np.array([p[len(p) // 2] for p in points])
    parent = list(range(len(table)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gap = np.cos(np.radians(TRACK_LINK_GAP_DEG))
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            if abs(table["query"][i] - table["query"][j]) > TRACK_LINK_QUERY + 1e-9:
                continue
            if abs(poles[i] @ poles[j]) < np.cos(np.radians(TRACK_LINK_POLE_DEG)):
                continue
            reach = (table.length_deg[i] + table.length_deg[j]) / 2 + TRACK_LINK_GAP_DEG
            if middles[i] @ middles[j] < np.cos(np.radians(reach)):
                continue
            if (points[i] @ points[j].T).max() >= gap:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(table)):
        groups.setdefault(root(i), []).append(i)

    known = _known_tracks()
    known_points = {k: _unit(*v) for k, v in known.items()}
    # the sky outside the footprint, for tracks that hug its edge
    import healpy as hp
    from scipy.spatial import cKDTree

    from streamgoggles.objects_overlap import get_footprint

    outside = np.flatnonzero(~get_footprint("des_yr6_inference", nside=64)[0])
    edge_tree = cKDTree(_unit(*hp.pix2ang(64, outside, lonlat=True)))
    edge_chord = 2 * np.sin(np.radians(TRACK_EDGE_DEG) / 2)
    des = {
        stream: (_unit(*des2018_arc(stream, n=600)), width, distance)
        for stream, (width, _, distance, _) in sp.DES_STREAMS.items()
    }

    reference_tangents = {k: _tangents(v) for k, v in known_points.items()}
    reference_tangents.update({k: _tangents(v[0]) for k, v in des.items()})

    def overlap(track_points, reference_name, reference, tolerance_deg):
        """Degrees of the track within tolerance of a reference track and
        running along it."""
        return _aligned_length(
            track_points,
            0.25,
            reference,
            tolerance_deg,
            reference_tangents=reference_tangents[reference_name],
        )

    rows = []
    for members in groups.values():
        cloud = np.concatenate([points[i] for i in members])
        pole = np.linalg.eigh(cloud.T @ cloud)[1][:, 0]
        u = cloud[0] - (cloud[0] @ pole) * pole
        u /= np.linalg.norm(u)
        v = np.cross(pole, u)
        phi = np.sort(np.degrees(np.arctan2(cloud @ v, cloud @ u)) % 360)
        gaps = np.diff(np.concatenate([phi, [phi[0] + 360]]))
        widest = int(np.argmax(gaps))
        start = phi[(widest + 1) % len(phi)]
        length = 360 - gaps[widest]
        along = np.radians(start + np.linspace(0, length, max(int(length / 0.25), 2)))
        arc = np.cos(along)[:, None] * u + np.sin(along)[:, None] * v
        ra = np.degrees(np.arctan2(arc[:, 1], arc[:, 0])) % 360
        dec = np.degrees(np.arcsin(np.clip(arc[:, 2], -1, 1)))
        mine = table.loc[members]
        weights = mine.ratio.to_numpy()
        distance = float(np.average(mine["query"], weights=weights))
        identification, how = "", ""
        best = 0.0
        for stream, (reference, width, stream_distance) in des.items():
            degrees = overlap(
                arc, stream, reference, max(TRACK_TOLERANCE_DEG, 2 * width)
            )
            if degrees >= MIN_ALONG_DEG and degrees > best:
                best = degrees
                at = abs(distance - stream_distance) <= 1.0
                identification = stream
                how = "DES 2018" if at else "DES 2018 track, another distance"
        edge_share = float(
            np.mean([len(h) > 0 for h in edge_tree.query_ball_point(arc, edge_chord)])
        )
        # the Magellanic Clouds' outskirts before other streams: they fill
        # their region with lines, which any stream crossing it would claim
        if not identification:
            middle = arc[len(arc) // 2]
            for obj in ("LMC", "SMC"):
                ra0, dec0, radius = LINE_SKY_OBJECTS[obj]
                if (
                    np.degrees(np.arccos(np.clip(middle @ _unit(ra0, dec0), -1, 1)))
                    <= radius
                ):
                    identification, how = f"{obj} periphery", "Magellanic"
                    break
        known_best, known_name = 0.0, ""
        for reference_name, reference in known_points.items():
            degrees = overlap(arc, reference_name, reference, GALSTREAMS_TOLERANCE_DEG)
            if degrees >= MIN_ALONG_DEG and degrees > known_best:
                known_best, known_name = degrees, reference_name
        # a track hugging the footprint's edge: the background model's edge
        # effect before any stream that happens to run there
        if not identification and edge_share >= TRACK_EDGE_SHARE:
            identification, how = "footprint edge", "edge"
        if not identification and known_name:
            identification, how = known_name, "galstreams"
        rows.append(
            {
                "length_deg": float(length),
                "ra_start": float(ra[0]),
                "dec_start": float(dec[0]),
                "ra_end": float(ra[-1]),
                "dec_end": float(dec[-1]),
                "ra_mid": float(ra[len(ra) // 2]),
                "dec_mid": float(dec[len(dec) // 2]),
                "distance_modulus": distance,
                "query_min": float(mine["query"].min()),
                "query_max": float(mine["query"].max()),
                "segments": len(members),
                "tiles": int(mine.tile.nunique()),
                "network_segments": int((mine.scorer == "network").sum()),
                "matched_filter_segments": int((mine.scorer == "matched filter").sum()),
                "best_ratio": float(weights.max()),
                "summed_ratio": float(weights.sum()),
                # the share of its lines' pixels on valid sky: low along the
                # edges of the footprint and of masks
                "valid_share": float(mine.valid_share.mean()),
                "edge_share": edge_share,
                "identification": identification or "unidentified",
                "kind": how or "unidentified",
                "galstreams": known_name,
            }
        )
    tracks = (
        pd.DataFrame(rows)
        .sort_values(["segments", "best_ratio"], ascending=False)
        .reset_index(drop=True)
    )
    tracks.index.name = "track"
    tracks.to_csv(LINE_SKY / f"tracks_{name}.csv")
    print(
        f"{len(table)} segments -> {len(tracks)} tracks; "
        f"{(tracks.segments >= 2).sum()} seen in at least two segments",
        flush=True,
    )
    print(tracks.groupby("kind").size().to_string(), flush=True)
    print(
        tracks[tracks.segments >= 2][
            [
                "identification",
                "kind",
                "length_deg",
                "ra_mid",
                "dec_mid",
                "distance_modulus",
                "segments",
                "tiles",
                "network_segments",
                "matched_filter_segments",
                "best_ratio",
            ]
        ]
        .round(2)
        .to_string(),
        flush=True,
    )


def line_sky_track_figure(config=LINE_SKY_CONFIG, mask_objects=True, min_segments=2):
    """line_sky_tracks.png: the tracks seen in at least ``min_segments``
    segments, over the DES sky, coloured by distance, the unidentified ones
    numbered; the DES 2018 tracks in grey underneath."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from streamgoggles.objects_overlap import des2018_arc, get_footprint

    sp = real_des().stream_parameters_module()
    suffix = _suffix(mask_objects)
    name = config.replace("/", "_") + suffix
    tracks = pd.read_csv(LINE_SKY / f"tracks_{name}.csv", index_col="track")
    shown = tracks[tracks.segments >= min_segments]
    footprint = get_footprint("des_yr6_inference", nside=64)[0]
    fra, fdec = hp.pix2ang(64, np.flatnonzero(footprint), lonlat=True)

    def wrap(ra):
        return np.where(np.asarray(ra) > 180, np.asarray(ra) - 360, ra)

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(15.0, 19.0)
    fig, ax = plt.subplots(figsize=(13, 7.2), layout="constrained")
    ax.scatter(wrap(fra), fdec, s=6, marker="s", color="#efefef", lw=0)
    for stream in sp.DES_STREAMS:
        ra, dec = des2018_arc(stream, n=200)
        ax.plot(wrap(ra), dec, color="#9e9e9e", lw=6, alpha=0.5, solid_capstyle="round")
    for obj, (ra, dec, _) in LINE_SKY_OBJECTS.items():
        at = max(dec, -68.5)
        ax.plot(wrap(ra), at, "+", color="#c0392b", ms=9, mew=1.5)
        ax.text(wrap(ra) - 1.5, at + 1.0, obj, fontsize=8, color="#c0392b")
    for number, r in shown.iterrows():
        ra, dec = _arc(r.ra_start, r.dec_start, r.ra_end, r.dec_end, 40)
        width = 1.2 + 0.4 * min(r.segments, 8)
        ax.plot(wrap(ra), dec, color=cmap(norm(r.distance_modulus)), lw=width)
        label = (
            r.identification.split(".")[0]
            if r.kind in ("DES 2018", "galstreams")
            else f"#{number}"
        )
        quiet = r.kind in ("Magellanic", "edge") or (
            r.kind == "galstreams" and r.segments < 4 and r.network_segments == 0
        )
        if not quiet:
            ax.text(
                wrap(r.ra_mid) - 1.0,
                r.dec_mid + 1.2,
                label,
                fontsize=7.5,
                color="#1a1a1a" if r.kind == "unidentified" else "#4d4d4d",
                weight="bold" if r.kind == "unidentified" else "normal",
            )
    ax.set_xlim(112, -65)
    ax.set_ylim(-70, 8)
    ax.set_aspect(1 / np.cos(np.radians(35)))
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.025,
        pad=0.01,
        label="distance modulus (mean of the track's segments)",
    )
    ax.set_title(
        f"tracks seen in at least {min_segments} segments (both searches at half "
        "their false-alarm rate); thicker: more segments; grey: DES 2018",
        fontsize=10,
    )
    fig.savefig(
        DOC_FIGURES / f"line_sky_tracks{suffix}{_figure_tag(config)}.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


# The leads of the track catalogues, looked at one by one: the ends of the
# detected track (the 2-degree model's catalogue, or the 4-degree model's
# where only it has the lead), its distance, the galstreams track it was
# matched to, and which search found it. ATLAS, a DES 2018 stream both
# searches find, is the control: what a real stream looks like here.
LEADS = {
    "ATLAS (control)": {
        "ends": ((9.84, -20.01), (33.75, -34.50)),
        "distance": 16.77,
        "compare": "AAU-ATLAS.li2021",
        "found_by": "both searches",
    },
    "Jhelum's eastern extension": {
        "ends": ((74.98, -23.57), (79.34, -34.35)),
        "distance": 15.2,
        "compare": "Jhelum.ibata2024",
        "found_by": "the line network",
    },
    "New-4 (matched filter's track)": {
        "ends": ((89.75, -45.39), (93.10, -34.80)),
        "distance": 16.26,
        "compare": "New-4.ibata2024",
        "found_by": "the matched filter",
    },
    "eastern track A": {
        "ends": ((95.45, -45.39), (86.85, -34.39)),
        "distance": 16.25,
        "compare": "New-4.ibata2024",
        "found_by": "the line network",
    },
    "eastern track B": {
        "ends": ((95.59, -50.10), (91.33, -40.03)),
        "distance": 16.5,
        "compare": None,
        "found_by": "the line network",
    },
    "Leiptr": {
        "ends": ((82.92, -18.15), (88.83, -28.90)),
        "distance": 15.25,
        "compare": "Leiptr.ibata2021",
        "found_by": "the line network",
    },
    "Cetus-Palca?": {
        "ends": ((27.57, -40.00), (37.75, -50.73)),
        "distance": 15.51,
        "compare": "Cetus-Palca.thomas2021",
        "found_by": "the line network",
    },
    # from the four-model catalogue: Tucana III's extensions (Ibata et al.
    # 2024) beyond the DES 2018 track, either side, and NGC 1261's stream
    "Tucana III, west of its DES 2018 track": {
        "ends": ((339.11, -62.65), (351.0, -61.64)),
        "distance": 16.29,
        "compare": "TucanaIII.ibata2024",
        "found_by": "the line network",
    },
    "Tucana III, east of its DES 2018 track": {
        "ends": ((6.0, -58.76), (19.47, -54.68)),
        "distance": 16.73,
        "compare": "TucanaIII.ibata2024",
        "found_by": "the line network",
    },
    "NGC 1261's stream": {
        "ends": ((48.96, -52.73), (29.30, -58.87)),
        "distance": 15.23,
        "compare": "NGC1261.ibata2024",
        "found_by": "the line network",
    },
}
LEAD_WIDTH_DEG = 0.4  # half-width of the band the profile and the diagram use
LEAD_OFF_DEG = (1.5, 3.0)  # the flanking bands, either side of the track
LEADS_DIR = LINE_SKY / "leads"


def _lead_frame(ends):
    """The lead's great circle: (pole, u, length_deg), u at its first end,
    so a position's along-track angle is atan2(p.(pole x u), p.u)."""
    import numpy as np

    a, b = _unit(*ends[0]), _unit(*ends[1])
    pole = np.cross(a, b)
    pole /= np.linalg.norm(pole)
    return pole, a, float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def _lead_coordinates(ra, dec, frame):
    """(offset from the track, position along it), degrees, for positions."""
    import numpy as np

    pole, u, _ = frame
    p = _unit(ra, dec)
    offset = np.degrees(np.arcsin(np.clip(p @ pole, -1, 1)))
    along = np.degrees(np.arctan2(p @ np.cross(pole, u), p @ u))
    return offset, along


def lead_inspection(names=None):
    """For each of LEADS: a zoomed map of the matched filter's excess (S/N)
    at its distance, the matched-filter band significance along it at every
    queried distance (the band against 200 null bands on the calibration
    sky), and the Hess difference of its stars against flanking bands, with
    the matched filter's polygon at its distance. Writes
    line_sky/leads/<lead>.png and line_sky/leads/leads.csv."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy import ndimage

    from streamgoggles.evaluation.footprint import (
        band_mean_statistics,
        null_band_placements,
        track_band,
    )
    from streamgoggles.matched_filter import build_matched_filters
    from streamgoggles.objects_overlap import des2018_arc, get_footprint

    rd = real_des()
    sp = rd.stream_parameters_module()
    background, _, pix = build_sky("inference", "count")
    nside = pix.nside
    valid = background.valid_mask_full & ~rd.object_mask(
        nside, max_dwarf_mv=LINE_SKY_DWARF_MV
    )
    valid &= get_footprint("des_yr6_inference", nside=nside)[0]
    calibration = valid & hp.read_map(rd.CALIBRATION_MASK).astype(bool)
    queries = list(sp.QUERY_GRID)
    smooth = smooth_backgrounds(background, queries, valid)
    excess = {
        q: np.where(valid, background.raw_map_full_dict["good"][q] - smooth[q], 0.0)
        for q in queries
    }
    snr_map = {
        q: np.where(
            valid,
            (background.raw_map_full_dict["good"][q] - smooth[q])
            / np.sqrt(np.maximum(smooth[q], 0.5)),
            hp.UNSEEN,
        )
        for q in queries
    }
    good = build_matched_filters(rd.filters_config(), namespace=rd.NAMESPACE)["good"]
    known = _known_tracks()
    stars_path = rd.INFERENCE_CATALOGUE
    g_col, r_col = f"{rd.NAMESPACE}_g_obs", f"{rd.NAMESPACE}_r_obs"
    LEADS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, lead in LEADS.items():
        if names is not None and name not in names:
            continue
        frame = _lead_frame(lead["ends"])
        length = frame[2]
        arc_ra, arc_dec = _arc(*lead["ends"][0], *lead["ends"][1], 200)
        band = track_band([(arc_ra, arc_dec)], LEAD_WIDTH_DEG, nside) & valid
        rng = np.random.default_rng([EVAL_SEED, 31])
        placements = null_band_placements(band, calibration, rng, N_NULL_BANDS)
        at = np.flatnonzero(band)
        profile = []
        for q in queries:
            stats = band_mean_statistics(excess[q], at, excess[q], placements)
            profile.append((q, stats["snr"], stats["p_value"]))
        profile = np.array(profile)
        query = min(queries, key=lambda q: abs(q - lead["distance"]))

        # the stars around the lead
        lo_ra, hi_ra = np.min(arc_ra) - 6, np.max(arc_ra) + 6
        lo_dec, hi_dec = np.min(arc_dec) - 5, np.max(arc_dec) + 5
        table = pq.read_table(
            stars_path,
            columns=["ra", "dec", g_col, r_col],
            filters=[
                ("ra", ">=", lo_ra),
                ("ra", "<=", hi_ra),
                ("dec", ">=", lo_dec),
                ("dec", "<=", hi_dec),
            ],
        ).to_pandas()
        on_sky = valid[
            hp.ang2pix(nside, table.ra.values, table.dec.values, lonlat=True)
        ]
        stars = table[on_sky]
        offset, along = _lead_coordinates(stars.ra.values, stars.dec.values, frame)
        inside = (along >= 0) & (along <= length)
        on = inside & (np.abs(offset) <= LEAD_WIDTH_DEG)
        off = (
            inside
            & (np.abs(offset) >= LEAD_OFF_DEG[0])
            & (np.abs(offset) <= LEAD_OFF_DEG[1])
        )
        # valid areas of the two regions, from the HEALPix pixels' centres
        pixels = hp.query_disc(
            nside, _unit(*np.mean(lead["ends"], axis=0)), np.radians(length / 2 + 4)
        )
        pixels = pixels[valid[pixels]]
        p_ra, p_dec = hp.pix2ang(nside, pixels, lonlat=True)
        p_offset, p_along = _lead_coordinates(p_ra, p_dec, frame)
        p_inside = (p_along >= 0) & (p_along <= length)
        area_on = (p_inside & (np.abs(p_offset) <= LEAD_WIDTH_DEG)).sum()
        area_off = (
            p_inside
            & (np.abs(p_offset) >= LEAD_OFF_DEG[0])
            & (np.abs(p_offset) <= LEAD_OFF_DEG[1])
        ).sum()
        scale = area_on / max(area_off, 1)
        g = stars[g_col].values
        colour = g - stars[r_col].values
        bins = (np.arange(-0.3, 1.25, 0.05), np.arange(16.5, 24.6, 0.2))
        h_on = np.histogram2d(colour[on], g[on], bins=bins)[0]
        h_off = np.histogram2d(colour[off], g[off], bins=bins)[0]
        hess = ndimage.gaussian_filter(h_on - scale * h_off, 1.0)
        polygon = good._polygon(["g", "r"], query)
        from matplotlib.path import Path as MplPath

        in_filter = MplPath(polygon).contains_points(np.c_[colour, g])
        n_on = int((on & in_filter).sum())
        n_off = int((off & in_filter).sum())
        cmd_excess = n_on - scale * n_off
        cmd_snr = cmd_excess / np.sqrt(max(n_on + scale**2 * n_off, 1))

        # the figure: map, profile, Hess difference
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        middle = np.mean(np.array(lead["ends"]), axis=0)
        projector = hp.projector.GnomonicProj(
            rot=(float(middle[0]), float(middle[1]), 0.0), xsize=240, reso=6.0
        )
        image = projector.projmap(
            snr_map[query], lambda x, y, z: hp.vec2pix(nside, x, y, z)
        )
        seen = np.isfinite(image) & (image > hp.UNSEEN / 2)
        values = np.where(seen, image, 0.0)
        weight = ndimage.gaussian_filter(seen.astype(float), 3.0)
        shown = ndimage.gaussian_filter(values, 3.0) / np.maximum(weight, 1e-3)
        extent = projector.get_extent()
        ax = axes[0]
        ax.imshow(
            np.where(seen, shown, np.nan),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-0.5,
            vmax=0.5,
        )

        def draw(ra, dec, ax=ax, projector=projector, **style):
            x, y = projector.ang2xy(np.asarray(ra), np.asarray(dec), lonlat=True)
            ax.plot(x, y, **style)

        pole = frame[0]
        for side in (-1, 1):  # two rails beside the track, not over it
            shift = np.radians(side * 1.0)
            rail = np.cos(shift) * _unit(arc_ra, arc_dec) + np.sin(shift) * pole
            draw(
                np.degrees(np.arctan2(rail[:, 1], rail[:, 0])) % 360,
                np.degrees(np.arcsin(np.clip(rail[:, 2], -1, 1))),
                color="#1a1a1a",
                lw=0.8,
                ls="--",
            )
        angle, overlap = np.nan, 0.0
        if lead["compare"] and lead["compare"] in known:
            draw(*known[lead["compare"]], color="#2e8b57", lw=1.2, ls=":")
            # how it meets the literature track: the angle where they are
            # closest, and how much of the lead lies within 1.5 deg of it
            ours = _unit(arc_ra, arc_dec)
            theirs = _unit(*known[lead["compare"]])
            cosines = ours @ theirs.T
            i, j = np.unravel_index(cosines.argmax(), cosines.shape)
            ours_t = np.cross(frame[0], ours[i])
            ours_t /= np.linalg.norm(ours_t)
            angle = float(
                np.degrees(np.arccos(np.clip(abs(_tangents(theirs)[j] @ ours_t), 0, 1)))
            )
            overlap = float(
                (cosines.max(1) >= np.cos(np.radians(GALSTREAMS_TOLERANCE_DEG))).sum()
                * length
                / (len(ours) - 1)
            )
        for stream in sp.DES_STREAMS:
            draw(*des2018_arc(stream, n=200), color="#7a7a7a", lw=3, alpha=0.4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        meeting = (
            f" (at {angle:.0f}°, {overlap:.0f}° within 1.5°)"
            if lead["compare"] and np.isfinite(angle)
            else ""
        )
        ax.set_title(
            f"matched-filter excess (S/N, smoothed 0.3°) at m−M {query:g}\n"
            f"dashed: the track ±1°; green dotted: {lead['compare'] or '—'}{meeting}",
            fontsize=8.5,
        )
        ax = axes[1]
        ax.plot(profile[:, 0], profile[:, 1], "o-", color="#1f6fb4", lw=2)
        ax.axhline(0, color="#b0b0b0", lw=0.8)
        ax.axvline(lead["distance"], color="#e07b39", ls="--", lw=1)
        for q, snr, p_value in profile:
            if p_value <= 1.0 / (N_NULL_BANDS + 1) + 1e-12:
                ax.plot(q, snr, "o", ms=11, mfc="none", mec="#c0392b", mew=1.5)
        ax.set_xlabel("queried distance modulus")
        ax.set_ylabel("band S/N against 200 null bands")
        ax.set_title(
            f"along the track ({length:.0f}°, ±{LEAD_WIDTH_DEG}°); orange: the "
            "detection's distance;\nred circles: beyond every null band",
            fontsize=8.5,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax = axes[2]
        limit = np.nanpercentile(np.abs(hess), 99) or 1.0
        ax.imshow(
            hess.T,
            origin="lower",
            aspect="auto",
            extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="#1a1a1a", lw=0.8)
        ax.set_ylim(bins[1][-1], bins[1][0])
        ax.set_xlabel("g − r")
        ax.set_ylabel("g")
        ax.set_title(
            f"stars on the track minus flanking bands ({LEAD_OFF_DEG[0]:g}-"
            f"{LEAD_OFF_DEG[1]:g}°), by area;\nin the filter at m−M {query:g}: "
            f"excess {cmd_excess:.0f} stars, {cmd_snr:.1f}σ",
            fontsize=8.5,
        )
        fig.suptitle(f"{name} — found by {lead['found_by']}", fontsize=11)
        fig.tight_layout()
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        fig.savefig(LEADS_DIR / f"{slug}.png", dpi=100, bbox_inches="tight")
        fig.savefig(DOC_FIGURES / f"lead_{slug}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        best = int(np.nanargmax(profile[:, 1]))
        rows.append(
            {
                "lead": name,
                "found_by": lead["found_by"],
                "length_deg": length,
                "detection_distance": lead["distance"],
                "profile_peak_distance": float(profile[best, 0]),
                "profile_peak_snr": float(profile[best, 1]),
                "snr_at_detection": float(profile[queries.index(query), 1]),
                "p_at_detection": float(profile[queries.index(query), 2]),
                "cmd_excess": float(cmd_excess),
                "cmd_snr": float(cmd_snr),
                "compare": lead["compare"] or "",
                "angle_to_compare_deg": angle,
                "overlap_with_compare_deg": overlap,
            }
        )
        print(f"{name}: done", flush=True)
    table = pd.DataFrame(rows)
    if names is not None and (LEADS_DIR / "leads.csv").exists():
        # keep the rows of the leads not looked at again
        kept = pd.read_csv(LEADS_DIR / "leads.csv")
        table = pd.concat([kept[~kept.lead.isin(table.lead)], table], ignore_index=True)
    table.to_csv(LEADS_DIR / "leads.csv", index=False)
    print(table.round(2).to_string(index=False), flush=True)


# The leads that held, fitted: a distance and a curved track each, with
# ATLAS as the control. Literature distance moduli: galstreams' where it has
# one (ATLAS, Li et al. 2021; Leiptr, Ibata et al. 2021); NGC 1261's stream
# at its cluster's (16.3 kpc, Harris 2010); Tucana III's extensions at
# Tucana III's (Shipp et al. 2018).
FIT_LEADS = {
    "ATLAS (control)": 16.65,
    "Leiptr": 14.25,
    "NGC 1261's stream": 16.06,
    "Tucana III, east of its DES 2018 track": 17.0,
    "Tucana III, west of its DES 2018 track": 17.0,
}
FIT_DISTANCES = (13.5, 18.5, 0.1)  # the trial distances: from, to, step
FIT_BIN_DEG = 1.5  # bins along the track for the track fit
FIT_HALF_WIDTH_DEG = 4.0  # how far across the track the fit looks
FIT_MARGIN_DEG = 3.0  # how far beyond the lead's ends
FIT_STEP_DEG = 0.2  # bins across the track


def _scan_distance(
    phi1, phi2, colour, g, polygons, track, width, length, pixels, offsets
):
    """The band's excess against flanking bands, at each trial distance:
    (S/N, excess) arrays. ``track(phi1)`` is the track's across offset;
    ``pixels`` the valid pixels' (phi1, phi2), for the areas."""
    import numpy as np
    from matplotlib.path import Path as MplPath

    inside = (phi1 >= 0) & (phi1 <= length)
    across = np.abs(phi2 - track(phi1))
    on = inside & (across <= width)
    off = inside & (across >= offsets[0]) & (across <= offsets[1])
    p_inside = (pixels[0] >= 0) & (pixels[0] <= length)
    p_across = np.abs(pixels[1] - track(pixels[0]))
    area_on = (p_inside & (p_across <= width)).sum()
    area_off = (p_inside & (p_across >= offsets[0]) & (p_across <= offsets[1])).sum()
    scale = area_on / max(area_off, 1)
    snr, excess = [], []
    points = np.c_[colour, g]
    for polygon in polygons:
        selected = MplPath(polygon).contains_points(points)
        n_on, n_off = (on & selected).sum(), (off & selected).sum()
        excess.append(n_on - scale * n_off)
        snr.append(excess[-1] / np.sqrt(max(n_on + scale**2 * n_off, 1)))
    return np.array(snr), np.array(excess)


def lead_fits(names=None):
    """For each of FIT_LEADS: its stars, selected by the matched filter at each
    trial distance from 13.5 to 18.5 (below the search's grid, which starts at
    15); the distance where the excess along its track stands out most
    against flanking bands; there, a curved track -- the across-track
    position of a Gaussian fitted to the excess in 1.5-degree bins along the
    track, then a quadratic through them -- and a width; and the distance
    again, along the curved track. Writes line_sky/leads/fits.csv and
    fit_<lead>.png."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy import ndimage
    from scipy.optimize import curve_fit

    from streamgoggles.matched_filter import build_matched_filters
    from streamgoggles.objects_overlap import get_footprint

    rd = real_des()
    background, _, pix = build_sky("inference", "count")
    nside = pix.nside
    valid = background.valid_mask_full & ~rd.object_mask(
        nside, max_dwarf_mv=LINE_SKY_DWARF_MV
    )
    valid &= get_footprint("des_yr6_inference", nside=nside)[0]
    good = build_matched_filters(rd.filters_config(), namespace=rd.NAMESPACE)["good"]
    distances = np.round(np.arange(*FIT_DISTANCES), 2)
    polygons = [good._polygon(["g", "r"], float(d)) for d in distances]
    known = _known_tracks()
    g_col, r_col = f"{rd.NAMESPACE}_g_obs", f"{rd.NAMESPACE}_r_obs"
    LEADS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, literature in FIT_LEADS.items():
        if names is not None and name not in names:
            continue
        lead = LEADS[name]
        frame = _lead_frame(lead["ends"])
        pole, u, length = frame
        v = np.cross(pole, u)
        reach = FIT_MARGIN_DEG + FIT_HALF_WIDTH_DEG
        arc_ra, arc_dec = _arc(*lead["ends"][0], *lead["ends"][1], 200)
        widen = 1 / np.cos(np.radians(min(np.max(np.abs(arc_dec)) + reach, 80)))
        table = pq.read_table(
            rd.INFERENCE_CATALOGUE,
            columns=["ra", "dec", g_col, r_col],
            filters=[
                ("ra", ">=", float(np.min(arc_ra) - reach * widen)),
                ("ra", "<=", float(np.max(arc_ra) + reach * widen)),
                ("dec", ">=", float(np.min(arc_dec) - reach)),
                ("dec", "<=", float(np.max(arc_dec) + reach)),
            ],
        ).to_pandas()
        on_sky = valid[
            hp.ang2pix(nside, table.ra.values, table.dec.values, lonlat=True)
        ]
        stars = table[on_sky]
        phi2, phi1 = _lead_coordinates(stars.ra.values, stars.dec.values, frame)
        near = (
            (np.abs(phi2) <= FIT_HALF_WIDTH_DEG)
            & (phi1 >= -FIT_MARGIN_DEG)
            & (phi1 <= length + FIT_MARGIN_DEG)
        )
        phi1, phi2 = phi1[near], phi2[near]
        g = stars[g_col].values[near]
        colour = g - stars[r_col].values[near]
        disc = hp.query_disc(
            nside,
            _unit(*np.mean(lead["ends"], axis=0)),
            np.radians(length / 2 + reach + 1),
        )
        disc = disc[valid[disc]]
        p2, p1 = _lead_coordinates(*hp.pix2ang(nside, disc, lonlat=True), frame)
        keep = (
            (np.abs(p2) <= FIT_HALF_WIDTH_DEG)
            & (p1 >= -FIT_MARGIN_DEG)
            & (p1 <= length + FIT_MARGIN_DEG)
        )
        pixels = (p1[keep], p2[keep])

        # 1. the distance, along the straight lead
        straight = lambda x: np.zeros_like(x)
        snr_straight, _ = _scan_distance(
            phi1,
            phi2,
            colour,
            g,
            polygons,
            straight,
            LEAD_WIDTH_DEG,
            length,
            pixels,
            LEAD_OFF_DEG,
        )
        best = int(np.argmax(snr_straight))

        # 2. the track: a Gaussian across the track in each bin along it
        from matplotlib.path import Path as MplPath

        selected = MplPath(polygons[best]).contains_points(np.c_[colour, g])
        edges1 = np.arange(-FIT_MARGIN_DEG, length + FIT_MARGIN_DEG + 1e-9, FIT_BIN_DEG)
        edges2 = np.arange(-FIT_HALF_WIDTH_DEG, FIT_HALF_WIDTH_DEG + 1e-9, FIT_STEP_DEG)
        centres2 = 0.5 * (edges2[1:] + edges2[:-1])
        counts = np.histogram2d(phi1[selected], phi2[selected], bins=(edges1, edges2))[
            0
        ]
        area = np.histogram2d(pixels[0], pixels[1], bins=(edges1, edges2))[0]
        density = np.where(area > 0, counts / np.maximum(area, 1), np.nan)

        def model(x, amplitude, centre, sigma, level, slope):
            return (
                amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
                + level
                + slope * x
            )

        fitted = []
        for k in range(len(edges1) - 1):
            profile, ok = density[k], np.isfinite(density[k])
            if ok.sum() < 15:
                continue
            smooth = ndimage.gaussian_filter1d(
                np.where(ok, profile, np.nanmedian(profile)), 1.5
            )
            window = np.abs(centres2) <= 1.5
            start = centres2[window][np.argmax(smooth[window])]
            level = np.nanmedian(profile)
            try:
                values, cov = curve_fit(
                    model,
                    centres2[ok],
                    profile[ok],
                    p0=[max(np.nanmax(smooth) - level, 1e-3), start, 0.3, level, 0.0],
                    sigma=np.sqrt(np.maximum(counts[k][ok], 1))
                    / np.maximum(area[k][ok], 1),
                    bounds=(
                        [0, -2.5, 0.05, -np.inf, -np.inf],
                        [np.inf, 2.5, 1.5, np.inf, np.inf],
                    ),
                    maxfev=5000,
                )
            except (RuntimeError, ValueError):
                continue
            errors = np.sqrt(np.diag(cov))
            if values[0] / max(errors[0], 1e-12) >= 2 and errors[1] < 1.0:
                fitted.append(
                    (0.5 * (edges1[k] + edges1[k + 1]), values[1], errors[1], values[2])
                )
        fitted = np.array(fitted)
        if len(fitted) >= 3:
            coefficients = np.polyfit(fitted[:, 0], fitted[:, 1], 2, w=1 / fitted[:, 2])
            width = float(np.median(fitted[:, 3]))
        else:  # too few bins to bend the track: keep it straight
            coefficients, width = np.zeros(3), LEAD_WIDTH_DEG
        curved = np.poly1d(coefficients)

        # 3. the distance again, along the curved track
        band = float(np.clip(1.5 * width, 0.25, 1.0))
        snr_curved, excess_curved = _scan_distance(
            phi1,
            phi2,
            colour,
            g,
            polygons,
            curved,
            band,
            length,
            pixels,
            (max(LEAD_OFF_DEG[0], 3 * band), max(LEAD_OFF_DEG[1], 3 * band + 1.5)),
        )
        # the curved track is kept only if it gathers more of the excess than
        # the straight lead: with few bins fitted, a quadratic can swing off
        # the stream (and then the distance scan along it means nothing)
        accepted = snr_curved.max() > snr_straight.max()
        if not accepted:
            curved = straight
            snr_curved, excess_curved = _scan_distance(
                phi1,
                phi2,
                colour,
                g,
                polygons,
                straight,
                LEAD_WIDTH_DEG,
                length,
                pixels,
                LEAD_OFF_DEG,
            )
            width = LEAD_WIDTH_DEG
        best_curved = int(np.argmax(snr_curved))
        within = distances[snr_curved >= snr_curved[best_curved] - 1]

        # the curved track on the sky, and against the literature track
        along = np.linspace(0, length, 120)
        offset = np.radians(curved(along))
        position = np.radians(along)
        points = (
            np.cos(offset)[:, None]
            * (np.cos(position)[:, None] * u + np.sin(position)[:, None] * v)
            + np.sin(offset)[:, None] * pole
        )
        track_ra = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
        track_dec = np.degrees(np.arcsin(np.clip(points[:, 2], -1, 1)))
        separation = np.nan
        if lead["compare"] in known:
            theirs = _unit(*known[lead["compare"]])
            closest = np.degrees(np.arccos(np.clip((points @ theirs.T).max(1), -1, 1)))
            separation = float(np.median(closest))

        # the figure: the stars across and along the track, the distance scans
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        ax = axes[0]
        # each bin along the track minus its own median: the stream, not the
        # density's gradient along the track
        shown = ndimage.gaussian_filter(
            np.nan_to_num(density - np.nanmedian(density, axis=1, keepdims=True)), 1.0
        )
        ax.imshow(
            shown.T,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            extent=[edges1[0], edges1[-1], edges2[0], edges2[-1]],
        )
        ax.plot(
            along,
            curved(along),
            color="#e07b39",
            lw=1.5,
            label="the fitted track" if accepted else "the lead (curve rejected)",
        )
        ax.axhline(0, color="#1f6fb4", lw=1, ls="--", label="the lead (straight)")
        if len(fitted):
            ax.errorbar(
                fitted[:, 0],
                fitted[:, 1],
                yerr=fitted[:, 2],
                fmt="o",
                color="#e07b39",
                ms=4,
            )
        ax.set_xlabel("along the lead (deg)")
        ax.set_ylabel("across (deg)")
        ax.set_title(
            f"stars in the filter at m−M {distances[best]:.1f}, density minus each "
            "column's median (smoothed); "
            f"width {width:.2f}°",
            fontsize=9,
        )
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax = axes[1]
        ax.plot(
            distances,
            snr_straight,
            color="#1f6fb4",
            lw=1.5,
            label="along the straight lead",
        )
        ax.plot(
            distances,
            snr_curved,
            color="#e07b39",
            lw=2,
            label="along the fitted track" if accepted else "(the lead again)",
        )
        ax.axvline(distances[best_curved], color="#e07b39", ls=":", lw=1)
        ax.axvline(
            literature, color="#2e8b57", ls="--", lw=1.2, label="literature distance"
        )
        ax.axvspan(
            15.0, 19.0, color="#f2f2f2", zorder=0, label="the search's distances"
        )
        ax.set_xlabel("trial distance modulus")
        ax.set_ylabel("excess S/N against flanking bands")
        ax.set_title(
            f"best m−M {distances[best_curved]:.1f} (S/N within 1: "
            f"{within.min():.1f}-{within.max():.1f}); literature {literature:.2f}",
            fontsize=9,
        )
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{name} — fitted", fontsize=11)
        fig.tight_layout()
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        fig.savefig(DOC_FIGURES / f"fit_{slug}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        rows.append(
            {
                "lead": name,
                "length_deg": length,
                "distance_straight": float(distances[best]),
                "snr_straight": float(snr_straight[best]),
                "distance": float(distances[best_curved]),
                "distance_low": float(within.min()),
                "distance_high": float(within.max()),
                "snr": float(snr_curved[best_curved]),
                "excess_stars": float(excess_curved[best_curved]),
                "literature_distance": literature,
                "width_deg": width,
                "bins_fitted": len(fitted),
                "curved_track_kept": bool(accepted),
                "curvature_deg": float(
                    curved(length / 2) - 0.5 * (curved(0) + curved(length))
                ),
                "median_offset_from_literature_track_deg": separation,
                "track_ra": " ".join(f"{x:.2f}" for x in track_ra[::10]),
                "track_dec": " ".join(f"{x:.2f}" for x in track_dec[::10]),
            }
        )
        print(
            f"{name}: m-M {distances[best_curved]:.1f} ({within.min():.1f}-{within.max():.1f}), "
            f"S/N {snr_curved[best_curved]:.1f}; straight {distances[best]:.1f}; "
            f"width {width:.2f}; {len(fitted)} bins; literature {literature}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    if names is not None and (LEADS_DIR / "fits.csv").exists():
        kept = pd.read_csv(LEADS_DIR / "fits.csv")
        table = pd.concat([kept[~kept.lead.isin(table.lead)], table], ignore_index=True)
    table.to_csv(LEADS_DIR / "fits.csv", index=False)


# The DES 2018 streams where DES 2018 found them, each method side by side:
# what the streamobs-trained models recover of the streams found by eye
DES2018 = OUT / "des2018"
DES2018_CONFIGS = ("hough/band2 residual x4", "hough/band2 residual long x4")
# Not in our data, even with the paper's own cuts, filter and track
# (docs: real_des/des2018_reproduction)
DES2018_NOT_IN_DATA = ("Molonglo", "Ravi")


def des2018_known(config="hough/band2 residual long x4", mask_objects=True):
    """The copies' two tests (`evaluate_hough`) on the real streams, where
    DES 2018 found them.

    Each stream's band -- within one width of its DES 2018 track (the arc of
    Shipp et al. 2018, `real_des.detection_tracks`) -- on the inference sky
    as the sky search sees it (`inference_sky`, the bright objects masked),
    at the queried distance nearest the stream's own, in the search's windows
    that hold at least HOUGH_MIN_LENGTH_DEG of it. Each window is scored by
    the models that never trained on its fold, and by the matched filter's
    line sums; the lines along the track in a window are those `hough_target`
    makes of the band there. Against the copies' stream-free windows
    (`null_windows`, on the calibration sky of the window's fold, scored by
    the same models):

    - known track: in the window holding the longest stretch of the band, the
      best line along the track against the same lines in the stream-free
      windows; found when at most 1 / (N_NULL_BANDS + 1) of them score as
      high;
    - blind: found when a line along the track, in any of its windows, scores
      above the level the best line of HOUGH_FALSE_ALARM of the stream-free
      windows reaches;
    - and the combination, as the sky search's: either search at half its
      rate (for the known track, either p-value at half the level).

    Writes des2018/known_<config>.csv.
    """
    import numpy as np
    import pandas as pd

    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.matched_filter import WindowProjection
    from streamgoggles.models.hough import hough_target
    from streamgoggles.objects_overlap import get_footprint, spatial_fold

    rd = real_des()
    sp = rd.stream_parameters_module()
    sky = inference_sky(config, mask_objects)
    pix, valid, grid = sky.pix, sky.valid, sky.grid
    usable = get_footprint("des_yr6_inference", nside=sky.nside)[0]
    min_length_pix = HOUGH_MIN_LENGTH_DEG / pix.pixel_scale_deg
    projections = [WindowProjection.for_window(t, pix, valid) for t in sky.tiles]
    folds = spatial_fold(
        np.array([t.center_ra for t in sky.tiles]), rd.STRIPE_DEG, rd.N_FOLDS
    )
    tracks = rd.detection_tracks()
    scorers = ("network", "matched filter")
    # each stream's windows on the inference sky, scored out of fold
    rows, longest = [], {}
    for name, (width, _, distance, _) in sp.DES_STREAMS.items():
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
        points = _unit(*tracks[name][0])
        reach = track_band(tracks[name], width, sky.nside) & usable
        band = reach & valid
        row = {
            "config": config,
            "stream": name,
            "distance_modulus": distance,
            "query": query,
            "width": width,
            "arc_deg": float(
                np.degrees(
                    np.arccos(np.clip((points[1:] * points[:-1]).sum(1), -1, 1))
                ).sum()
            ),
            "input_snr": rd.REAL_INPUT_SNR[name],
            "band_valid_share": float(band.sum() / max(reach.sum(), 1)),
            "n_windows": 0,
        }
        margins, stretches = {}, []
        for i, projection in enumerate(projections):
            on = band[projection.pixnums]
            if not on.any():
                continue
            inside = projection.image(on.astype(float)) > 0.5
            stretch = _run_length_pix(inside, grid)
            if stretch < min_length_pix:
                continue
            region = hough_target(inside, grid) > 0
            scoring = f"fold{1 - folds[i]}"
            answers = sky.score[scoring](
                sky.maps, valid, sky.smooth[query], [projection], query
            )
            along = {s: float(np.nanmax(answers[s][0][region])) for s in scorers}
            row["n_windows"] += 1
            for s in scorers:
                key = (scoring, s, round(query, 1))
                for rate, levels in (("", sky.levels), (" half", sky.half)):
                    margins[s + rate] = max(
                        margins.get(s + rate, -np.inf), along[s] - levels[key]
                    )
            stretches.append((stretch, i, scoring, region, along))
        for s in scorers:
            row[f"{s} blind margin"] = margins.get(s, np.nan)
            row[f"{s} blind"] = bool(margins.get(s, -np.inf) >= 0)
        row["combined blind"] = any(
            margins.get(s + " half", -np.inf) >= 0 for s in scorers
        )
        if stretches:
            stretch, i, scoring, region, along = max(stretches, key=lambda x: x[0])
            row.update(
                {
                    "longest_deg": stretch * pix.pixel_scale_deg,
                    "window_ra": sky.tiles[i].center_ra,
                    "window_dec": sky.tiles[i].center_dec,
                    "models": scoring,
                    **{f"{s} along": along[s] for s in scorers},
                }
            )
            longest[name] = (scoring, query, region, along)
        rows.append(row)
        print(
            f"{name}: {row['n_windows']} windows, "
            + ", ".join(f"{s} blind {row[f'{s} blind']}" for s in scorers),
            flush=True,
        )
    # the known track: against the same lines in the stream-free windows of
    # the models that scored the stream's longest stretch
    table = pd.DataFrame(rows).set_index("stream")
    for train_sky in ("fold0", "fold1"):
        names = [n for n, v in longest.items() if v[0] == train_sky]
        if not names:
            continue
        fold_sky = window_level_sky(EVALUATED_ON[train_sky], pix.image_size_pix[0])
        windows = null_windows(fold_sky)
        maps = [
            fold_sky.background.raw_map_full_dict[c["filter"]][c["distance_modulus"]]
            for c in channels()
        ]
        queries = sorted({longest[n][1] for n in names})
        smooth = smooth_backgrounds(fold_sky.background, queries)
        for q in queries:
            start = time.time()
            null = sky.score[train_sky](maps, fold_sky.valid, smooth[q], windows, q)
            for n in names:
                _, query, region, along = longest[n]
                if query != q:
                    continue
                for s in scorers:
                    null_along = np.nanmax(null[s][:, region], axis=1)
                    exceed = int(np.sum(null_along >= along[s]))
                    table.loc[n, f"{s} known exceed"] = exceed
                    table.loc[n, f"{s} known p"] = (1 + exceed) / (1 + len(null_along))
                    table.loc[n, f"{s} null median"] = float(np.median(null_along))
            print(
                f"{train_sky} m-M {q:.1f}: stream-free windows in "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
        del fold_sky, maps, smooth
    known_level = 1.0 / (N_NULL_BANDS + 1)
    for s in scorers:
        table[f"{s} known"] = table[f"{s} known p"] <= known_level + 1e-12
    table["combined known"] = (
        table[[f"{s} known p" for s in scorers]].min(axis=1) <= known_level / 2 + 1e-12
    )
    DES2018.mkdir(parents=True, exist_ok=True)
    table = table.reset_index()
    table.to_csv(
        DES2018 / f"known_{config.replace('/', '_')}{_suffix(mask_objects)}.csv",
        index=False,
    )
    pd.set_option("display.width", 220)
    print(
        table[
            ["stream", "query", "input_snr", "n_windows", "longest_deg"]
            + [f"{s} {t}" for t in ("known p", "known", "blind") for s in scorers]
            + ["combined known", "combined blind"]
        ]
        .round(3)
        .to_string(index=False),
        flush=True,
    )
    return table


def _copies(config):
    """The DES 2018 copies of `evaluate_hough` for a configuration, both
    folds, at Table 1's surface brightness and FAINTER_BY fainter: a row per
    copy, with its stream and ``fainter`` (0 at Table 1's)."""
    import pandas as pd

    name = config.replace("/", "_")
    tables = [
        pd.read_csv(path)
        for train_sky in ("fold0", "fold1")
        for suffix in ("", "__fainter")
        if (path := result_dir(train_sky) / f"hough_{name}{suffix}.csv").exists()
    ]
    copies = pd.concat(tables, ignore_index=True)
    copies = copies[copies.set.isin(["DES 2018", "fainter"])].copy()
    parts = copies.stream.str.extract(r"^(?P<stream>.*?)(?: \+(?P<fainter>[\d.]+))?$")
    copies["stream"] = parts.stream
    copies["fainter"] = parts.fainter.astype(float).fillna(0.0)
    return copies


def des2018_strength(config="hough/band2 residual long x4"):
    """How strong each DES 2018 stream is in our data, on the simulation's
    scale: its copies' median input S/N at Table 1's surface brightness and
    FAINTER_BY fainter, a line in log S/N against magnitudes fainter (the S/N
    falls as the flux, 0.4 dex a magnitude), read at the real stream's S/N
    (`real_des.REAL_INPUT_SNR`) -- how much fainter than Table 1's the real
    stream is, and its surface brightness on streamobs's scale. Writes
    des2018/strength.csv."""
    import numpy as np
    import pandas as pd

    rd = real_des()
    sp = rd.stream_parameters_module()
    copies = _copies(config)
    rows = []
    for name, (width, length, distance, sb) in sp.DES_STREAMS.items():
        levels = copies[copies.stream == name].groupby("fainter").input_snr.median()
        slope, intercept = np.polyfit(levels.index, np.log10(levels.to_numpy()), 1)
        real = rd.REAL_INPUT_SNR[name]
        fainter = (np.log10(real) - intercept) / slope if real > 0 else np.nan
        rows.append(
            {
                "stream": name,
                "distance_modulus": distance,
                "width": width,
                "length": length,
                "table1_sb": sb,
                "copies_snr": float(levels.loc[0.0]),
                "real_snr": real,
                "dex_per_mag": float(slope),
                "fainter_by": float(fainter),
                "effective_sb": float(sb + fainter),
            }
        )
    table = pd.DataFrame(rows).sort_values("effective_sb")
    DES2018.mkdir(parents=True, exist_ok=True)
    table.to_csv(DES2018 / "strength.csv", index=False)
    return table


def _at_strength(copies, column, snr):
    """(share, side): the share of a stream's copies found (``column``) at
    the input S/N ``snr`` -- each brightness's share at the median input S/N
    of its copies, interpolated in log S/N; beyond the copies' range, the
    nearest brightness's share, and on which side ("fainter", "brighter")."""
    import numpy as np

    levels = (
        copies.groupby("fainter")
        .agg(snr=("input_snr", "median"), found=(column, "mean"))
        .sort_values("snr")
    )
    if snr < levels.snr.iloc[0]:
        return float(levels.found.iloc[0]), "fainter"
    if snr > levels.snr.iloc[-1]:
        return float(levels.found.iloc[-1]), "brighter"
    return float(np.interp(np.log(snr), np.log(levels.snr), levels.found)), ""


def des2018_table(configs=DES2018_CONFIGS):
    """The DES 2018 streams, each method side by side, in long form: per
    stream, configuration, scorer and test (known track, blind), whether the
    real stream is found (`des2018_known`), and the share of its copies
    found at its strength in our data (`real_des.REAL_INPUT_SNR`;
    `_at_strength`). With the per-pixel network's test (the first training,
    along the known track) and the sky search's verdict (lines along the
    track beyond chance, `line_sky_summary`). Writes des2018/streams.csv."""
    import numpy as np
    import pandas as pd

    rd = real_des()
    rows = []
    for config in configs:
        name = config.replace("/", "_")
        known = pd.read_csv(DES2018 / f"known_{name}.csv")
        copies = _copies(config)
        matches = pd.read_csv(LINE_SKY / f"matches_{name}.csv").set_index("stream")
        for _, stream in known.iterrows():
            mine = copies[copies.stream == stream.stream]
            table1 = mine[mine.fainter == 0]
            for scorer in ("network", "matched filter", "combined"):
                for test in ("known", "blind"):
                    share, side = (np.nan, "")
                    if scorer != "combined":
                        share, side = _at_strength(
                            mine, f"{scorer} {test}", rd.REAL_INPUT_SNR[stream.stream]
                        )
                    sky = np.nan
                    if test == "blind" and f"{scorer} found" in matches:
                        found = matches.loc[stream.stream]
                        sky = bool(
                            found[f"{scorer} found"]
                            and found[f"{scorer} p"] <= LINE_SKY_SIGNIFICANCE
                        )
                    rows.append(
                        {
                            "config": config,
                            "stream": stream.stream,
                            "distance_modulus": stream.distance_modulus,
                            "input_snr": rd.REAL_INPUT_SNR[stream.stream],
                            "scorer": scorer,
                            "test": test,
                            "found": bool(stream[f"{scorer} {test}"]),
                            "copies_at_strength": share,
                            "copies_side": side,
                            "copies_table1": float(table1[f"{scorer} {test}"].mean())
                            if scorer != "combined"
                            else np.nan,
                            "copies_table1_snr": float(table1.input_snr.median()),
                            "sky_search": sky,
                            "per_pixel": bool(
                                matches.loc[stream.stream, "per-pixel found"]
                            ),
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(DES2018 / "streams.csv", index=False)
    return table


# The methods of the DES 2018 figures: (configuration, scorer) -> label, colour
DES2018_METHODS = {
    ("hough/band2 residual long x4", "matched filter"): (
        "matched-filter\nline sums",
        "#e07b39",
    ),
    ("hough/band2 residual x4", "network"): (
        "line network,\n4 quick models",
        "#88c999",
    ),
    ("hough/band2 residual long x4", "network"): (
        "line network,\n4 long models",
        "#2e8b57",
    ),
    ("hough/band2 residual long x4", "combined"): (
        "both, half\nthe rate each",
        "#1f6fb4",
    ),
}


def des2018_figures(configs=DES2018_CONFIGS):
    """des2018_methods.png: the fourteen DES 2018 streams, strongest in our
    data first, and which method finds each where DES 2018 found it -- along
    the known track and without it (`des2018_known`), with the per-pixel
    network's test (the first training, along the track) for reference.
    des2018_copies.png: each stream found or not against the share of its
    copies found at its strength in our data (`des2018_table`)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    des2018_strength()
    table = des2018_table(configs)
    streams = (
        table.drop_duplicates("stream")
        .sort_values("input_snr", ascending=False)
        .reset_index(drop=True)
    )
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    def found(config, scorer, test):
        chosen = table[
            (table.config == config) & (table.scorer == scorer) & (table.test == test)
        ].set_index("stream")
        return chosen.loc[streams.stream, "found"].to_numpy(bool)

    columns = [("known", "per-pixel network\n(first training)", "#4d4d4d", None)]
    columns += [
        ("known", label, colour, key)
        for key, (label, colour) in DES2018_METHODS.items()
    ]
    columns += [
        ("blind", label, colour, key)
        for key, (label, colour) in DES2018_METHODS.items()
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    x = 0.0
    centres = {"known": [], "blind": []}
    for test, label, colour, key in columns:
        if test == "blind" and not centres["blind"]:
            x += 0.8  # a gap between the two tests
        if key is None:
            hits = (
                table.drop_duplicates("stream")
                .set_index("stream")
                .loc[streams.stream, "per_pixel"]
                .to_numpy(bool)
            )
        else:
            hits = found(key[0], key[1], test)
        for y, hit in enumerate(hits):
            ax.scatter(
                x,
                y,
                s=140,
                facecolors=colour if hit else "white",
                edgecolors=colour,
                linewidths=1.5,
            )
        ax.text(
            x,
            len(streams) - 0.2,
            f"{int(hits.sum())} of {len(hits)}",
            ha="center",
            va="top",
            fontsize=9,
        )
        ax.text(x, -0.9, label, ha="center", va="bottom", fontsize=8.5)
        centres[test].append(x)
        x += 1.0
    for test, title in (
        ("known", "along the known track"),
        ("blind", "without the track (1% false lines per window)"),
    ):
        ax.text(
            np.mean(centres[test]),
            -2.3,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            weight="bold",
        )
    ax.set_yticks(range(len(streams)))
    ax.set_yticklabels(
        [
            f"{row.stream}  (S/N {row.input_snr:.1f}, m−M {row.distance_modulus:g})"
            for row in streams.itertuples()
        ],
        fontsize=9,
    )
    ax.set_ylim(len(streams) + 0.4, -2.6)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / "des2018_methods.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # each stream against its copies at its strength
    panels = [
        (config, scorer, test)
        for test in ("known", "blind")
        for config, scorer in (
            ("hough/band2 residual long x4", "network"),
            ("hough/band2 residual long x4", "matched filter"),
        )
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    for ax, (config, scorer, test) in zip(axes.flat, panels, strict=True):
        chosen = table[
            (table.config == config)
            & (table.scorer == scorer)
            & (table.test == test)
            & ~table.stream.isin(DES2018_NOT_IN_DATA)
        ].sort_values("copies_at_strength", ascending=False)
        label, colour = DES2018_METHODS[(config, scorer)]
        placed = []  # the labels' boxes so far: (log10 x, y) corners
        for row in chosen.itertuples():
            ax.scatter(
                row.input_snr,
                row.copies_at_strength,
                s=70,
                marker="o"
                if not row.copies_side
                else ("<" if row.copies_side == "fainter" else ">"),
                facecolors=colour if row.found else "white",
                edgecolors=colour,
                linewidths=1.5,
                zorder=3,
            )
            # the first of four places beside the point whose box (in log S/N
            # and share, about 0.02 dex a letter and 0.035 high) meets no other
            x0, y0 = np.log10(row.input_snr), row.copies_at_strength
            width = 0.021 * len(row.stream)
            for dx, dy, side in ((5, 3, "left"), (5, -10, "left"),
                                 (-5, 3, "right"), (-5, -10, "right")):  # fmt: skip
                left = x0 + 0.012 if side == "left" else x0 - 0.012 - width
                bottom = y0 + (0.01 if dy > 0 else -0.045)
                box = (left, bottom, left + width, bottom + 0.035)
                if not any(
                    box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                    for b in placed
                ):
                    break
            placed.append(box)
            ax.annotate(
                row.stream,
                (row.input_snr, row.copies_at_strength),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=side,
                fontsize=7.5,
            )
        ax.set_xscale("log")
        ax.set_xlim(3, 30)
        ax.set_xticks([3, 4, 6, 10, 20, 30])
        ax.set_xticklabels(["3", "4", "6", "10", "20", "30"])
        ax.minorticks_off()
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(0.5, color="#b0b0b0", lw=0.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"{label.replace(chr(10), ' ')}, "
            + ("along the known track" if test == "known" else "without the track"),
            fontsize=10,
        )
    for ax in axes[1]:
        ax.set_xlabel("the real stream's S/N in our data")
    for ax in axes[:, 0]:
        ax.set_ylabel("its copies found at that S/N")
    fig.suptitle(
        "The twelve DES 2018 streams in our data against their copies at their "
        "strength\nfilled: the real stream found; open: missed; ◁ fainter than all "
        "its copies (the faintest's share, an upper bound), ▷ brighter",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / "des2018_copies.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return table


# The trainings compared: the same 2-degree line model, two quick models per
# fold, its band label from S/N 2 or 5 in the window, trained on the
# population set (32-34.5 mag/arcsec2) or at the DES 2018 streams' strength in
# our data (32.5-35.5)
DES2018_TRAININGS = {
    "hough/band2 residual": ("lines from S/N 2,\n32-34.5", "#88c999"),
    "hough/band2 residual des": ("lines from S/N 2,\n32.5-35.5", "#1b5e20"),
    "hough/band2s5 residual": ("lines from S/N 5,\n32-34.5", "#7b3294"),
    "hough/band2s5 residual des": ("lines from S/N 5,\n32.5-35.5", "#3f007d"),
}
# More short streams in training, against the uniform lengths (S/N-5 label,
# two quick models per fold each)
DES2018_SHORT = {
    "hough/band2s5 residual": ("lengths uniform,\n4-30°", "#7b3294"),
    "hough/band2s5 residual short": ("lengths log-uniform,\n4-30°", "#c51b7d"),
}
# The quick S/N-5 models' seeds: the first pair, the second, and all four
DES2018_SEEDS = {
    "hough/band2s5 residual": ("lines from S/N 5,\nseeds 42-43", "#7b3294"),
    "hough/band2s5 residual s44": ("lines from S/N 5,\nseeds 44-45", "#c2a5cf"),
    "hough/band2s5 residual x4": ("lines from S/N 5,\nall four", "#40004b"),
}
# Segment lines against window lines (S/N-5 label, two quick models per fold)
DES2018_SEG = {
    "hough/band2s5 residual": ("window lines", "#7b3294"),
    "hough/band2s5 residual seg": ("window and\nsub-window lines", "#e66101"),
}
# The segment-line model's seeds, against four window-line models per fold
DES2018_SEG_SEEDS = {
    "hough/band2s5 residual x4": ("window lines,\nall four", "#40004b"),
    "hough/band2s5 residual seg": ("segment lines,\nseeds 42-43", "#e66101"),
    "hough/band2s5 residual seg s44": ("segment lines,\nseeds 44-45", "#fdb863"),
    "hough/band2s5 residual seg x4": ("segment lines,\nall four", "#b35806"),
}
# What longer training buys the S/N-5 label: two quick models per fold, two
# long ones, and the four long S/N-2 models per fold (the best before)
DES2018_LONG = {
    "hough/band2s5 residual": ("lines from S/N 5,\n2 quick models", "#7b3294"),
    "hough/band2s5 residual long": ("lines from S/N 5,\n2 long models", "#3f007d"),
    "hough/band2 residual long x4": ("lines from S/N 2,\n4 long models", "#2e8b57"),
}


def des2018_training(trainings=DES2018_TRAININGS, name="training"):
    """The line network under several trainings, for each configuration of
    ``trainings``: on the copies (fold-0 models on fold 1: the input S/N
    where half are found, near and far; the DES 2018 copies found; and, if
    evaluated, the bright streams of the length scan found by length), on
    the sky (the DES 2018 streams found beyond chance, `line_sky_summary`),
    and on the real streams where DES 2018 found them (`des2018_known`).
    Writes des2018/<name>.csv and des2018_<name>.png: each stream found or
    not under each training."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rows, known = [], {}
    for config in trainings:
        name_ = config.replace("/", "_")
        copies = pd.concat(
            [
                pd.read_csv(path)
                for suffix in ("", "__fainter")
                if (path := result_dir("fold0") / f"hough_{name_}{suffix}.csv").exists()
            ]
        )
        copies = copies[copies.set.isin(["DES 2018", "fainter"])]
        row = {"config": config}
        for near, part in ((True, "near"), (False, "far")):
            mine = copies[(copies.distance_modulus < 16.5) == near]
            for scorer, test in (("network", "blind"), ("network", "known")):
                rated = mine.assign(detected=mine[f"{scorer} {test}"].astype(float))
                row[f"{test} half-recovery S/N, {part}"] = half_recovery_snr(rated)
            row[f"DES 2018 copies found blind, {part}"] = float(
                mine[mine.set == "DES 2018"]["network blind"].mean()
            )
        matches = pd.read_csv(LINE_SKY / f"matches_{name_}.csv")
        significant = {
            scorer: matches[f"{scorer} found"]
            & (matches[f"{scorer} p"] <= LINE_SKY_SIGNIFICANCE)
            for scorer in ("network", "matched filter", "combined")
        }
        row["sky: network"] = int(significant["network"].sum())
        row["sky: both, half rate"] = int(significant["combined"].sum())
        row["sky: either"] = int(
            (significant["network"] | significant["matched filter"]).sum()
        )
        known[config] = pd.read_csv(DES2018 / f"known_{name_}.csv").set_index("stream")
        for test in ("known", "blind"):
            row[f"DES 2018 streams, network {test}"] = int(
                known[config][f"network {test}"].sum()
            )
        scan = result_dir("fold0") / f"hough_{name_}__length-scan.csv"
        if scan.exists():
            bright = pd.read_csv(scan)
            bright = bright[
                bright.stream.str.endswith(f"SB {_patches.LENGTH_SCAN['sb'][0]:g}")
            ]
            length = bright.stream.str.split().str[1].astype(float)
            for degrees in (4.0, 5.0, 6.0, 8.0):
                row[f"bright {degrees:g} deg streams found blind"] = float(
                    bright[length == degrees]["network blind"].mean()
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DES2018 / f"{name}.csv", index=False)

    strength = pd.read_csv(DES2018 / "strength.csv").set_index("stream")
    order = strength.sort_values("real_snr", ascending=False).index
    fig, ax = plt.subplots(figsize=(4.5 + 1.1 * len(trainings) * 2, 6.8))
    x = 0.0
    centres = {"known": [], "blind": []}
    for test in ("known", "blind"):
        if test == "blind":
            x += 0.8
        for config, (label, colour) in trainings.items():
            hits = known[config].loc[order, f"network {test}"].to_numpy(bool)
            for y, hit in enumerate(hits):
                ax.scatter(
                    x,
                    y,
                    s=140,
                    facecolors=colour if hit else "white",
                    edgecolors=colour,
                    linewidths=1.5,
                )
            ax.text(x, len(order) - 0.2, f"{int(hits.sum())} of {len(hits)}",
                    ha="center", va="top", fontsize=9)  # fmt: skip
            ax.text(x, -0.9, label, ha="center",
                    va="bottom", fontsize=8.5)  # fmt: skip
            centres[test].append(x)
            x += 1.0
    for test, title in (("known", "along the known track"), ("blind", "without it")):
        ax.text(np.mean(centres[test]), -2.3, title, ha="center", va="bottom",
                fontsize=10.5, weight="bold")  # fmt: skip
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [
            f"{s}  (S/N {strength.loc[s, 'real_snr']:.1f}, "
            f"{strength.loc[s, 'effective_sb']:.1f} mag/arcsec²)"
            if np.isfinite(strength.loc[s, "effective_sb"])
            else f"{s}  (S/N {strength.loc[s, 'real_snr']:.1f})"
            for s in order
        ],
        fontsize=9,
    )
    ax.set_ylim(len(order) + 0.4, -2.6)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / f"des2018_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(table.round(2).T.to_string(), flush=True)
    return table


# The bright dwarfs whose outskirts the search masks, and the radii compared
DES2018_DWARFS = {
    "Fornax": (39.96, -34.50, 3.98),
    "Sculptor": (15.02, -33.72, 2.23),
}


def des2018_dwarf_profiles(queries=(16.0, 17.5, 19.0)):
    """des2018_dwarf_profiles.png: Fornax's and Sculptor's stars in the
    matched filter, against the distance from their centres -- the counts in
    annuli over those 6-9 degrees out, at a few queried distances, with the
    radius the object mask reaches (12 half-light radii) and
    LINE_SKY_DWARF_MAX_DEG. Aliqa Uma's band (three widths) and the other
    masked objects are left out of the annuli."""
    import itertools

    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.objects_overlap import des2018_arc

    rd = real_des()
    background, _, pix = build_sky("inference", "count", IMAGE_PIX)
    nside = pix.nside
    valid = background.valid_mask_full & ~track_band(
        [des2018_arc("Aliqa Uma", n=400)], 3 * 0.26, nside
    )
    others = rd.object_mask(nside, max_dwarf_mv=LINE_SKY_DWARF_MV)
    edges = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    fig, axes = plt.subplots(1, len(DES2018_DWARFS), figsize=(11, 4.2), sharey=True)
    colours = ("#9ecae1", "#3182bd", "#08519c")
    for ax, (name, (ra, dec, radius)) in zip(axes, DES2018_DWARFS.items(), strict=True):
        centre = hp.ang2vec(ra, dec, lonlat=True)
        pixels = np.flatnonzero(valid)
        sep = np.degrees(
            np.arccos(
                np.clip(
                    hp.ang2vec(*hp.pix2ang(nside, pixels, lonlat=True), lonlat=True)
                    @ centre,
                    -1,
                    1,
                )
            )
        )
        keep = (sep < 9) & ~(others[pixels] & (sep > radius + 0.1))
        pixels, sep = pixels[keep], sep[keep]
        middle = (edges[:-1] + edges[1:]) / 2
        for q, colour in zip(queries, colours, strict=True):
            counts = background.raw_map_full_dict["good"][q][pixels]
            ring = counts[(sep >= 6) & (sep < 9)].mean()
            excess = [
                counts[(sep >= a) & (sep < b)].mean() / ring - 1
                for a, b in itertools.pairwise(edges)
            ]
            ax.plot(middle, 100 * np.array(excess), "o-", color=colour, lw=2,
                    label=f"m−M {q:g}")  # fmt: skip
        ax.axvline(radius, color="#7f7f7f", ls="--", lw=1)
        ax.text(radius, 0.97, " mask: 12 half-light radii", rotation=90, va="top",
                ha="right", fontsize=8, transform=ax.get_xaxis_transform())  # fmt: skip
        ax.axvline(LINE_SKY_DWARF_MAX_DEG, color="#d95f02", ls="--", lw=1)
        ax.text(LINE_SKY_DWARF_MAX_DEG, 0.97, " tight mask", rotation=90, va="top",
                ha="right", fontsize=8, color="#d95f02",
                transform=ax.get_xaxis_transform())  # fmt: skip
        ax.axhline(0, color="#b0b0b0", lw=0.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([-5, 0, 5, 10, 100])
        ax.set_yticklabels(["−5", "0", "5", "10", "100"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("distance from the centre (degrees)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("matched-filter counts over the 6-9° ring (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        DOC_FIGURES / "des2018_dwarf_profiles.png", dpi=110, bbox_inches="tight"
    )
    plt.close(fig)


# The selection's faint limits compared (g and r; ours is 24.5)
DES2018_DEPTHS = (23.0, 23.5, 24.0, 24.5)


def des2018_depth(limits=DES2018_DEPTHS, own_distance=False):
    """Each DES 2018 stream's S/N in our matched filter against the
    selection's faint limit: the inference catalogue's stars, both bands
    between 16 and the limit, selected by the matched filter at the queried
    distance nearest the stream's; their density across its DES 2018 track,
    on the search's valid sky (the contaminants' pixels and the tight object
    mask out) away from the other DES 2018 streams (two widths of theirs), in
    bins of a quarter width out to four widths; a quadratic
    fitted to it beyond two widths; the stars within one width against it.
    With ``own_distance``, the matched filter at the stream's own distance
    (Table 1) instead, as `real_des.REAL_INPUT_SNR` was measured. Writes
    des2018/depth.csv (depth_own.csv)."""
    import healpy as hp
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree

    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.matched_filter import build_matched_filters
    from streamgoggles.objects_overlap import des2018_arc

    rd = real_des()
    sp = rd.stream_parameters_module()
    nside = 512
    background, _, _ = build_sky("inference", "count", IMAGE_PIX)
    excluded = hp.read_map(str(_patches.contaminants())) != 0
    valid = (
        background.valid_mask_full
        & ~excluded
        & ~rd.object_mask(
            nside,
            max_dwarf_mv=LINE_SKY_DWARF_MV,
            max_radius_deg=LINE_SKY_DWARF_MAX_DEG,
        )
    )
    arcs = {name: des2018_arc(name, n=400) for name in sp.DES_STREAMS}
    regions = {
        name: track_band([arcs[name]], 4.5 * width, nside)
        for name, (width, *_) in sp.DES_STREAMS.items()
    }
    anywhere = np.logical_or.reduce(list(regions.values())) & valid
    # the stars near any track, read a row group at a time
    catalogue = pq.ParquetFile(rd.INFERENCE_CATALOGUE)
    kept = []
    for group in range(catalogue.num_row_groups):
        stars = catalogue.read_row_group(group).to_pandas()
        pixel = hp.ang2pix(
            nside, stars.ra.to_numpy(), stars.dec.to_numpy(), lonlat=True
        )
        kept.append(stars[anywhere[pixel]].assign(pixel=pixel[anywhere[pixel]]))
    stars = pd.concat(kept, ignore_index=True)
    good = build_matched_filters(rd.filters_config(), namespace=rd.NAMESPACE)["good"]
    g, r = (stars[f"{rd.NAMESPACE}_{band}_obs"].to_numpy() for band in ("g", "r"))
    area = hp.nside2pixarea(nside, degrees=True)

    def across(vectors, points, tangents, poles):
        """Signed distance from the track (degrees), and whether the nearest
        track point is inside the arc (not an end)."""
        nearest = cKDTree(points).query(vectors)[1]
        phi2 = np.degrees(np.arcsin(np.clip((vectors * poles[nearest]).sum(1), -1, 1)))
        return phi2, (nearest > 0) & (nearest < len(points) - 1)

    rows = []
    for name, (width, _, distance, _) in sp.DES_STREAMS.items():
        query = (
            distance
            if own_distance
            else min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
        )
        points = _unit(*arcs[name])
        tangents = _tangents(points)
        poles = np.cross(points, tangents)
        poles /= np.linalg.norm(poles, axis=1, keepdims=True)
        clear = ~np.logical_or.reduce(
            [
                track_band([arcs[other]], 2 * sp.DES_STREAMS[other][0], nside)
                for other in sp.DES_STREAMS
                if other != name
            ]
        )
        mine = (regions[name] & clear)[stars.pixel.to_numpy()]
        phi2, inside = across(
            _unit(stars.ra.to_numpy()[mine], stars.dec.to_numpy()[mine]),
            points,
            tangents,
            poles,
        )
        selected = good.select(stars[mine], ["g", "r"], query)
        sky = np.flatnonzero(regions[name] & valid & clear)
        sky_phi2, sky_inside = across(
            np.asarray(hp.pix2vec(nside, sky)).T, points, tangents, poles
        )
        edges = np.arange(-4.0, 4.0 + 1e-9, 0.25) * width
        centres = (edges[:-1] + edges[1:]) / 2
        areas = np.histogram(sky_phi2[sky_inside], edges)[0] * area
        side = (np.abs(centres) >= 2 * width) & (areas > 0)
        core = np.abs(centres) < width
        row = {"stream": name, "distance_modulus": distance, "query": query}
        for limit in limits:
            chosen = (
                selected
                & inside
                & (g[mine] >= 16.0)
                & (r[mine] >= 16.0)
                & (g[mine] <= limit)
                & (r[mine] <= limit)
            )
            counts = np.histogram(phi2[chosen], edges)[0]
            fit = np.polyfit(
                centres[side], counts[side] / areas[side], 2, w=np.sqrt(areas[side])
            )
            expected = float((np.polyval(fit, centres[core]) * areas[core]).sum())
            row[f"S/N {limit:g}"] = (counts[core].sum() - expected) / np.sqrt(
                max(expected, 1.0)
            )
            row[f"stars {limit:g}"] = int(counts[core].sum())
        row["S/N reference"] = rd.REAL_INPUT_SNR[name]
        rows.append(row)
        print(
            name,
            {k: round(v, 1) for k, v in row.items() if k.startswith("S/N")},
            flush=True,
        )
    table = pd.DataFrame(rows)
    DES2018.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        DES2018 / ("depth_own.csv" if own_distance else "depth.csv"), index=False
    )
    return table


# What the sky search's lines lie along, and how the summary draws them
DES2018_LINE_CLASSES = {
    "DES 2018": ("along a DES 2018 track", "#1a9850"),
    "known": ("along another known stream (galstreams)", "#0b4fff"),
    "objects": ("around the Magellanic Clouds or a bright dwarf", "#9e9ac8"),
    "unexplained": ("unexplained: false alarm or candidate", "#d7301f"),
}


def des2018_sky_lines(config="hough/band2s5 residual x4", mask_objects="tight"):
    """The sky search's lines (`line_sky`, either search at half its rate:
    about 1% of stream-free windows hold one), each classed by what it lies
    along: a DES 2018 track (`line_sky_matches`), another stream galstreams
    traces (at least MIN_ALONG_DEG of the line within GALSTREAMS_TOLERANCE_DEG
    of it, running along it), the Magellanic Clouds' outskirts or a bright
    dwarf (LINE_SKY_OBJECTS), or nothing known. Writes
    des2018/sky_lines_<config><mask>.csv and returns the table."""
    import pandas as pd

    name = config.replace("/", "_") + _suffix(mask_objects)
    table = pd.read_csv(LINE_SKY / f"detections_{name}.csv")
    table = table[_chosen(table, "combined")].reset_index(drop=True)
    known = {k: _unit(*v) for k, v in _known_tracks().items()}
    known_tangents = {k: _tangents(v) for k, v in known.items()}
    classes = []
    for row in table.itertuples():
        if row.kind == "along a DES 2018 track":
            classes.append("DES 2018")
            continue
        if str(row.kind).startswith("around"):
            classes.append("objects")
            continue
        n = max(int(row.length_deg / 0.1), 2)
        points = _unit(*_arc(row.ra1, row.dec1, row.ra2, row.dec2, n))
        step = row.length_deg / (n - 1)
        along_known = any(
            _aligned_length(
                points, step, reference, GALSTREAMS_TOLERANCE_DEG,
                reference_tangents=known_tangents[k],
            )
            >= MIN_ALONG_DEG
            for k, reference in known.items()
        )  # fmt: skip
        classes.append("known" if along_known else "unexplained")
    table["class"] = classes
    DES2018.mkdir(parents=True, exist_ok=True)
    table.to_csv(DES2018 / f"sky_lines_{name}.csv", index=False)
    return table


def des2018_gif(config="hough/band2s5 residual x4", mask_objects="tight"):
    """des2018_sky.gif: the matched filter over the DES footprint, one frame
    per queried distance -- the counts over their smooth local background,
    minus one, averaged over nside-64 pixels (0.9 degrees) on the search's
    valid sky and interpolated between them -- with the sky
    search's lines found at that distance (`des2018_sky_lines`), coloured by
    what they lie along (solid: the line network's; dotted: the matched
    filter's line sums), and the fourteen DES 2018 tracks (dashed; bold at
    the queried distance nearest their own)."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patheffects
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    from streamgoggles.objects_overlap import des2018_arc

    rd = real_des()
    sp = rd.stream_parameters_module()
    lines = des2018_sky_lines(config, mask_objects)
    background, _, pix = build_sky("inference", "count", IMAGE_PIX)
    nside = pix.nside
    valid = background.valid_mask_full & ~rd.object_mask(
        nside,
        max_dwarf_mv=LINE_SKY_DWARF_MV,
        max_radius_deg=LINE_SKY_DWARF_MAX_DEG if mask_objects == "tight" else None,
    )
    queries = list(sp.QUERY_GRID)
    smooth = smooth_backgrounds(background, queries, valid)
    ra_axis = np.arange(-65.0, 105.0, 0.1)
    dec_axis = np.arange(-72.0, 6.0, 0.1)
    grid_ra, grid_dec = np.meshgrid(ra_axis, dec_axis)
    coarse = 64
    on_sky = hp.get_interp_val(
        hp.ud_grade(valid.astype(float), coarse), grid_ra % 360, grid_dec, lonlat=True
    )
    images = {}
    for q in queries:
        counts = hp.ud_grade(
            np.where(valid, background.raw_map_full_dict["good"][q], 0.0), coarse
        )
        expected = hp.ud_grade(np.where(valid, smooth[q], 0.0), coarse)
        contrast = counts / np.maximum(expected, 1e-9) - 1
        images[q] = np.where(
            on_sky > 0.5,
            hp.get_interp_val(contrast, grid_ra % 360, grid_dec, lonlat=True),
            np.nan,
        )

    def wrap(ra):
        ra = np.asarray(ra, float) % 360
        return np.where(ra > 180, ra - 360, ra)

    nearest = {
        name: min(queries, key=lambda q: abs(q - distance))
        for name, (_, _, distance, _) in sp.DES_STREAMS.items()
    }
    fig, ax = plt.subplots(figsize=(14, 7))

    def draw(k):
        q = queries[k]
        ax.clear()
        ax.imshow(
            images[q],
            origin="lower",
            extent=(ra_axis[0], ra_axis[-1], dec_axis[0], dec_axis[-1]),
            cmap="gray",
            vmin=-0.1,
            vmax=0.1,
            alpha=0.75,
            interpolation="bilinear",
        )
        for name in sp.DES_STREAMS:
            ra, dec = des2018_arc(name, n=200)
            here = nearest[name] == q
            track = ax.plot(wrap(ra), dec, ls="--", color="#fee08b",
                            lw=2.6 if here else 1.4)[0]  # fmt: skip
            track.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=4.5 if here else 3, foreground="black"
                    )
                ]
            )
            middle = len(ra) // 2
            label = ax.text(wrap(ra[middle]) + 1.0, dec[middle], name, color="#fee08b",
                            fontsize=8.5 if here else 7, weight="bold" if here else "normal",
                            alpha=1.0 if here else 0.75)  # fmt: skip
            label.set_path_effects(
                [patheffects.withStroke(linewidth=2.5, foreground="black")]
            )
        mine = lines[np.isclose(lines["query"], q)]
        counts = {}
        for kind, (_, colour) in DES2018_LINE_CLASSES.items():
            chosen = mine[mine["class"] == kind]
            counts[kind] = len(chosen)
            for row in chosen.itertuples():
                ra, dec = _arc(row.ra1, row.dec1, row.ra2, row.dec2, 20)
                ax.plot(wrap(ra), dec, color=colour, lw=2.4,
                        ls="-" if row.scorer == "network" else ":")  # fmt: skip
        ax.set_xlim(ra_axis[-1], ra_axis[0])  # RA increasing to the left
        ax.set_ylim(dec_axis[0], dec_axis[-1])
        ax.set_xlabel("RA (degrees)")
        ax.set_ylabel("Dec (degrees)")
        at = [n for n, d in nearest.items() if d == q]
        ax.set_title(
            f"matched filter at m−M {q:.1f}"
            + (f" — DES 2018 streams at this distance: {', '.join(at)}" if at else "")
            + "\nlines found: "
            + ", ".join(f"{counts[k]} {DES2018_LINE_CLASSES[k][0].split(':')[0]}" for k in DES2018_LINE_CLASSES),
            fontsize=10,
        )  # fmt: skip
        handles = [
            Line2D([], [], color=colour, lw=2.4, label=label)
            for label, colour in DES2018_LINE_CLASSES.values()
        ]
        handles += [
            Line2D([], [], color="#636363", lw=2.4, label="line network"),
            Line2D([], [], color="#636363", lw=2.4, ls=":", label="matched-filter line sums"),
            Line2D([], [], color="#fee08b", lw=1.5, ls="--", label="DES 2018 track"),
        ]  # fmt: skip
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        return []

    animation = FuncAnimation(fig, draw, frames=len(queries), blit=False)
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    animation.save(
        DOC_FIGURES / "des2018_sky.gif", writer=PillowWriter(fps=0.8), dpi=80
    )
    draw(queries.index(17.0))
    fig.savefig(DOC_FIGURES / "des2018_sky_17.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return lines


def hough_figures(train_sky="fold0"):
    """hough_{blind,known}_{sky}.png and hough.csv: the line model, the
    per-pixel network with a line search on top, and the matched filter, on
    the same copies (DES 2018 at full and reduced brightness): found against
    input S/N, near and far, at 1% false lines per stream-free window
    (blind) or along the known track."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    folder = result_dir(train_sky)

    def read(kind, config):
        # the line searches' results here, the per-pixel ones in patches.py's
        prefix = "hough" if kind == "lines" else "evaluation"
        base = folder if kind == "lines" else pixel_result_dir(train_sky)
        name = config.replace("/", "_")
        parts = [
            base / f"{prefix}_{name}.csv",
            base / f"{prefix}_{name}__fainter.csv",
        ]
        frames = [pd.read_csv(p) for p in parts if p.exists()]
        if not frames:  # not evaluated yet
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        if kind == "pixels":
            data = data[data.scorer == "ensemble"]
        return data[data.set.isin(["DES 2018", "fainter"])]

    rows = []
    for figure, curves in HOUGH_FIGURES.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
        for ax, (label, near) in zip(
            axes, (("m−M < 16.5", True), ("m−M ≥ 16.5", False)), strict=True
        ):
            for kind, config, test, legend in curves:
                colour = HOUGH_COLOURS[legend]
                data = read(kind, config)
                if test not in data:  # not evaluated yet
                    continue
                mine = data[(data.distance_modulus < 16.5) == near].assign(
                    detected=lambda d, t=test: d[t].astype(float)
                )
                rate = mine.groupby(
                    pd.cut(mine.input_snr, SNR_EDGES), observed=True
                ).detected.agg(["mean", "size"])
                rate = rate[rate["size"] >= 4]
                centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
                style = "--" if test.startswith("matched") else "-"
                ax.plot(
                    centres,
                    rate["mean"],
                    "o",
                    ls=style,
                    lw=2,
                    color=colour,
                    label=legend,
                )
                full = mine[mine.set == "DES 2018"]
                rows.append(
                    {
                        "figure": figure,
                        "test": legend,
                        "streams": label,
                        "half_recovery_snr": half_recovery_snr(mine),
                        "found": mine.detected.mean(),
                        "DES 2018 copies found": full.detected.mean(),
                    }
                )
            ax.set_xscale("log")
            ax.set_xticks([2, 4, 6, 10, 20, 40])
            ax.set_xticklabels(["2", "4", "6", "10", "20", "40"])
            ax.minorticks_off()
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("S/N of the copy in the matched-filter input")
            ax.axhline(0.5, color="#b0b0b0", lw=0.8, ls=":")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("copies found")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(
            {
                "blind": "without the track (1% false lines per stream-free window)",
                "known": "along the known track",
                "combination": "the per-pixel network searched along lines",
            }[figure],
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            DOC_FIGURES / f"hough_{figure}_{train_sky}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "hough.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "step",
        choices=[
            "train",
            "evaluate",
            "figures",
            "line-sky",
            "leads",
            "fits",
            "des2018",
        ],
    )
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument(
        "--mask",
        choices=["objects", "tight", "none"],
        default="objects",
        help="line-sky, des2018: the search mask (bright dwarfs to 12 half-light "
        "radii and clusters; the same with the dwarfs to 2 degrees at most; none)",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["DES 2018", "distance scan"],
        help="evaluate: which stream sets (DES 2018, distance scan, fainter, length scan)",
    )
    parser.add_argument(
        "--train-sky",
        choices=["fold0", "fold1"],
        default="fold0",
        help="where models train (scored on the other fold)",
    )
    arguments = parser.parse_args()
    mask = {"objects": True, "tight": "tight", "none": False}[arguments.mask]
    if arguments.step == "train":
        train(arguments.config, arguments.seed, arguments.train_sky)
    elif arguments.step == "evaluate":
        evaluate_hough(
            arguments.config, train_sky=arguments.train_sky, sets=tuple(arguments.sets)
        )
    elif arguments.step == "leads":  # the catalogue's leads, one by one
        lead_inspection()
    elif arguments.step == "fits":  # the leads that held, fitted
        lead_fits()
    elif arguments.step == "des2018":  # the DES 2018 streams where they are
        for config in [arguments.config] if arguments.config else DES2018_CONFIGS:
            des2018_known(config, mask)
        if mask is True and all(
            (DES2018 / f"known_{c.replace('/', '_')}.csv").exists()
            for c in DES2018_CONFIGS
        ):
            des2018_figures()
    elif arguments.step == "line-sky":  # the line model over the whole DES sky
        config = arguments.config or LINE_SKY_CONFIG
        line_sky(config, mask)
        line_sky_matches(config, mask)
        line_sky_chance(config, mask)
        line_sky_figures(config, mask)
        line_sky_summary(config, mask)
        line_sky_tracks(config, mask)
        line_sky_track_figure(config, mask)
    else:  # the copies' figures, the stream-free windows', and the guide's
        hough_figures(arguments.train_sky)
        hough_null_figure(arguments.train_sky)
        line_model_figures(arguments.train_sky)
