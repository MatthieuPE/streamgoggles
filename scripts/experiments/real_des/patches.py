"""Label and normalization, tested quickly on two patches of the real DES sky.

The first real-data models found 5 of the 14 DES 2018 streams, answered with
blobs, and saw nothing nearer than m-M 16. An audit of their training windows
showed why the label may be to blame: it marks the pixels where the stream's
selected stars exceed one per pixel, so a near or sparse stream is labelled by
a few random clumps. This experiment compares, on quick models:

- two labels: "count" (the first training's: selected stream stars > 1 per
  pixel) and "band" (the stream's band, one width either side of its track,
  labelled in a channel when the band as a whole stands out of the real
  background there, S/N >= 2, with at least 4 degrees of it on valid sky --
  `StreamInjector(label_policy="stream_band")`);
- two normalizations: "window" (each channel standardized by its own mean and
  spread in the window, as before) and "decoy" (every channel standardized by
  the decoy channel's, `DecoyNormalizer`).

Everything on the real DES Y6 training sky (known streams, clusters, dwarfs
masked), with the nearby-galaxy and artefact mask (`contaminant_mask`) removed
as well. Models train on patch A and are scored on patch B, a different part
of the sky, by the detection test of the real data: false-alarm rate <= 1e-3
against patch B's own stream-free output, >= 20 flagged pixels within one
width of the track, S/N >= 2 against null bands.

Steps (from the repository root, streamml environment):

  python scripts/experiments/real_des/patches.py audit              # label quality, both labels
  python scripts/experiments/real_des/patches.py train --config band/decoy --seed 42
  python scripts/experiments/real_des/patches.py evaluate --config band/decoy
  python scripts/experiments/real_des/patches.py figures
"""

import argparse
import importlib.util
import json
import pickle
import tempfile
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "experiments" / "real_des" / "patches"
MODELS = OUT / "models"
DOC_FIGURES = REPO / "docs" / "source" / "experiments" / "figures" / "real_des_patches"
CONTAMINANTS = OUT / "contaminants_nside512.fits"

# Two 30 x 20 degree patches, almost entirely unmasked training sky, far
# apart: A for training, B for evaluation.
TRAIN_PATCH = {
    "center_ra": 60.0,
    "center_dec": -45.0,
    "width_deg": 30.0,
    "height_deg": 20.0,
}
EVAL_PATCH = {
    "center_ra": 40.0,
    "center_dec": -20.0,
    "width_deg": 30.0,
    "height_deg": 20.0,
}

CONFIGS = {
    "count/window": {"label": "count", "normalizer": "window"},
    "count/decoy": {"label": "count", "normalizer": "decoy"},
    "band/window": {"label": "band", "normalizer": "window"},
    "band/decoy": {"label": "band", "normalizer": "decoy"},
    "count/poisson": {"label": "count", "normalizer": "poisson"},
    "band/poisson": {"label": "band", "normalizer": "poisson"},
    # round 2b: the band label with a stricter visibility cut, S/N >= 5
    "band5/window": {"label": "band5", "normalizer": "window"},
    # round 2c: the band cut into 1-degree segments along the track, each
    # labelled only where the stream stands out locally (S/N >= 1)
    "bandseg/window": {"label": "bandseg", "normalizer": "window"},
    # round 2d: no new training -- the count-label and band-label models,
    # which are complementary stream by stream, averaged into one ensemble
    "count+band/window": {
        "label": None,
        "normalizer": "window",
        "parts": ["count/window", "band/window"],
    },
    # larger windows: 128 x 128 pixels (14.7 deg) instead of 96 (11 deg)
    "count/window 128px": {"label": "count", "normalizer": "window", "image_pix": 128},
    # sensitivity levers: six quick models; two models trained four times longer
    "count/window x6": {
        "label": None,
        "normalizer": "window",
        "parts": ["count/window"],
        "seeds": [42, 43, 44, 45, 46, 47],
    },
    "count/window long": {"label": "count", "normalizer": "window", "windows": 19200},
    # its fair reference: as many count-label models (four seeds)
    "count/window x4": {
        "label": None,
        "normalizer": "window",
        "parts": ["count/window"],
        "seeds": [42, 43, 44, 45],
    },
    # why the network is half as sensitive as its input: the depth-2 U-Net
    # draws 90% of its input from within 1.5 deg, so it sums a stream over a
    # few degrees of track. Depth 4: a field of the whole window.
    "count/window deep": {
        "label": "count",
        "normalizer": "window",
        "model": {"depth": 4},
    },
    # training streams down to 36 mag/arcsec2 instead of 34.5: half the copies
    # at input S/N 4-7 are fainter than anything the reference trained on
    "count/window faint": {
        "label": "count",
        "normalizer": "window",
        "training_set": "population faint",
    },
    "band/window faint": {
        "label": "band",
        "normalizer": "window",
        "training_set": "population faint",
    },
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
    # batch Dice weights bright streams' many pixels over faint ones' few;
    # per-pixel cross-entropy weights every pixel alike (logits head)
    "count/window bce": {
        "label": "count",
        "normalizer": "window",
        "model": {"head": "identity"},
        "loss": "bce",
    },
}
# Where models train and where they are scored. Round 1 trained on patch A and
# was scored on patch B: one 600 deg2 patch proved too narrow a sky (the models
# fire on 9-60% of patch B's stream-free sky). Round 2 trains on the whole of
# fold 0's training sky -- same number of windows, same cost -- and is scored
# on fold 1's calibration sky.
SKIES = {
    "A": {"region": TRAIN_PATCH},
    "B": {"region": EVAL_PATCH},
    "fold0": {"fold": 0},
    "fold1": {"fold": 1},
    # the whole DES sky with the known streams in it (the inference
    # catalogue), for the on-sky maps; galaxies and artefacts masked as in
    # every other sky here
    "inference": {"catalogue": "inference"},
}
EVALUATED_ON = {"A": "B", "fold0": "fold1", "fold1": "fold0"}
ROUND2_CONFIGS = [
    "count/window",
    "band/window",
    "band/decoy",
    "count/poisson",
    "band/poisson",
]


def result_dir(train_sky):
    """Round 1 (patch A) writes to OUT itself; others to OUT/<train sky>."""
    return OUT if train_sky == "A" else OUT / train_sky


SEEDS = [42, 43]
WINDOWS = 4800  # the quick tier
IMAGE_PIX = 96
BAND_MIN_SNR = 2.0
BAND_MIN_LENGTH_DEG = 4.0

# Evaluation streams on patch B: the DES 2018 streams with their own Table 1
# parameters (length capped at 15 degrees to fit the patch), and a distance
# scan at fixed geometry.
N_PLACEMENTS = 8
MAX_LENGTH = 15.0
SCAN = {
    "distances": [15.0, 16.0, 17.0, 18.0, 19.0],
    "width": 0.3,
    "length": 10.0,
    "sb": 33.0,
}
TARGET_FALSE_ALARM_RATE = 1e-3
MIN_FLAGGED_PIXELS = 20
MIN_SNR = 2.0
N_NULL_BANDS = 200
EVAL_SEED = 2028
MF_BACKGROUND_NSIDE = 32  # the matched-filter test's local background: 1.8 deg pixels


def real_des():
    spec = importlib.util.spec_from_file_location(
        "real_des_run", REPO / "scripts/experiments/real_des/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contaminants():
    import healpy as hp
    import numpy as np

    from streamgoggles.objects_overlap import contaminant_mask

    if not CONTAMINANTS.exists():
        CONTAMINANTS.parent.mkdir(parents=True, exist_ok=True)
        hp.write_map(
            CONTAMINANTS, contaminant_mask(512).astype(np.float32), dtype=np.float32
        )
    return CONTAMINANTS


def build_sky(sky, label, image_pix=IMAGE_PIX):
    """Background and injector on a patch or a fold of the real training sky."""
    import numpy as np

    from streamgoggles.background import Background
    from streamgoggles.background_sources import (
        PreparedCatalogBackgroundSource,
        StudyRegion,
    )
    from streamgoggles.injector import StreamInjector
    from streamgoggles.matched_filter import PixelizationSpec, build_matched_filters
    from streamgoggles.storage import BackgroundMapStore
    from streamgoggles.stream_sources import StreamObsSource

    rd = real_des()
    sp = rd.stream_parameters_module()
    pix = PixelizationSpec(nside=512, image_size_pix=(image_pix, image_pix))
    config = rd.filters_config()
    filters = build_matched_filters(config, namespace=rd.NAMESPACE)
    catalogue = (
        rd.INFERENCE_CATALOGUE
        if SKIES[sky].get("catalogue") == "inference"
        else rd.TRAINING_CATALOGUE
    )
    source_cfg = {"path": str(catalogue), "exclude": str(contaminants())}
    if "fold" in SKIES[sky]:
        source_cfg["fold"] = {
            "index": SKIES[sky]["fold"],
            "n_folds": rd.N_FOLDS,
            "stripe_deg": rd.STRIPE_DEG,
            "nside": 512,
        }
    region = SKIES[sky].get("region")
    background = Background.load_or_cache(
        source=PreparedCatalogBackgroundSource(),
        source_cfg=source_cfg,
        study_region=StudyRegion(**region) if region else None,
        cuts=[],
        clipping=rd.CLIPPING,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=sp.CHANNEL_DISTANCES,
        finalize_cfg={"enabled": False},
        store=BackgroundMapStore(OUT / "background_maps"),
        pix=pix,
        survey=rd.SURVEY,
        release=rd.RELEASE,
        filter_configs=config,
    )
    background.catalog = None
    for per_distance in background.raw_map_full_dict.values():
        for dm, counts in per_distance.items():
            per_distance[dm] = counts.astype(np.float32)
    labels = {
        "count": {"label_policy": "stream_detection", "count_threshold": 1.0},
        "band": {
            "label_policy": "stream_band",
            "band_min_snr": BAND_MIN_SNR,
            "band_min_length_deg": BAND_MIN_LENGTH_DEG,
        },
        "band5": {
            "label_policy": "stream_band",
            "band_min_snr": 5.0,
            "band_min_length_deg": BAND_MIN_LENGTH_DEG,
        },
        # the band label down to 2 degrees: a short, clear segment is a line,
        # not a negative example (the line model's own label)
        "band2": {
            "label_policy": "stream_band",
            "band_min_snr": BAND_MIN_SNR,
            "band_min_length_deg": 2.0,
        },
        "bandseg": {
            "label_policy": "stream_band",
            "band_min_snr": BAND_MIN_SNR,
            "band_min_length_deg": BAND_MIN_LENGTH_DEG,
            "band_segment_deg": 1.0,
            "band_segment_min_snr": 1.0,
        },
    }
    injector = StreamInjector(
        background=background,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=rd.CLIPPING,
        pix=pix,
        survey=rd.SURVEY,
        release=rd.RELEASE,
        bands=("g", "r"),
        richness_kind="surface_brightness",
        finalize_cfg={"enabled": False},
        min_stream_length_deg=3.0,
        **labels[label],
    )
    return background, injector, pix


def normalizer(name, channels):
    from streamgoggles.datasets.transforms import DecoyNormalizer, WindowNormalizer

    if name == "window":
        return WindowNormalizer()
    if name == "poisson":
        from streamgoggles.datasets.transforms import PoissonNormalizer

        return PoissonNormalizer()
    if name == "residual":
        from streamgoggles.datasets.transforms import ResidualNormalizer

        return ResidualNormalizer()
    decoy = next(i for i, c in enumerate(channels) if c["filter"] == "decoy")
    return DecoyNormalizer(decoy)


def channels():
    rd = real_des()
    sp = rd.stream_parameters_module()
    return [
        {"filter": name, "distance_modulus": dm}
        for dm in sp.CHANNEL_DISTANCES
        for name in rd.filters_config()
    ]


def model_stem(config, seed, train_sky="A"):
    return result_dir(train_sky) / "models" / f"{config.replace('/', '_')}_seed{seed}"


def hough_parts(config):
    """(model factory, view wrapper) for a Hough configuration: the model from
    `build_model(kind="hough")` with the configuration's options (backbone
    as the per-pixel models'), and the view turning the label into lines."""
    from streamgoggles.models import build_model
    from streamgoggles.models.hough import HoughLines, HoughTargetTransform

    sp = real_des().stream_parameters_module()
    options = CONFIGS[config]["hough"]
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    grid = HoughLines(
        image_pix,
        image_pix,
        options["n_theta"],
        options["rho_step"],
        options["min_pixels"],
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


def train(config, seed, train_sky="A"):
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
    factory, wrapper = None, None
    if "hough" in CONFIGS[config]:
        factory, wrapper = hough_parts(config)
    model, _, result = sp.train(
        seed,
        windows,
        background,
        injector,
        training_set=CONFIGS[config].get("training_set", rd.TRAINING_SET),
        normalizer=normalizer(CONFIGS[config]["normalizer"], channels()),
        model_options=CONFIGS[config].get("model"),
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


def audit(n=300, labels=("count", "band")):
    """Label quality on the training patch: is the label one elongated band?"""
    import numpy as np
    from scipy import ndimage

    from streamgoggles.config import StreamConfig
    from streamgoggles.datasets.stream_map_dataset import StreamMapDataset
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.storage import SimulationStore

    rd = real_des()
    sp = rd.stream_parameters_module()
    rows = []
    for label in labels:
        background, injector, pix = build_sky("A", label)
        config = StreamConfig(
            params=sp.training_parameters(rd.TRAINING_SET),
            background_fraction=0.0,
            persist=False,
            richness_kind="surface_brightness",
        )
        dataset = StreamMapDataset(
            config=config,
            background=background,
            injector=injector,
            store=SimulationStore(tempfile.mkdtemp()),
            eval_mode=False,
            steps_per_epoch=n,
            rng=np.random.default_rng(7),
        )
        view = QueryDistanceTransform(
            StreamMapTransform(
                normalizer=normalizer("window", channels()), augment=False
            ),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            rng=np.random.default_rng(8),
        )
        start = time.time()
        for i in range(n):
            sample = dataset[i]
            viewed = view(sample)
            mask = viewed["label_stack"][0] > 0.5
            params = viewed["params"]
            row = {
                "label": label,
                "distance_modulus": params["distance_modulus"],
                "query": params["query_distance_modulus"],
                "width": params["width"],
                "valid_fraction": float(sample["valid_mask"].mean()),
                "positive": int(mask.sum()),
            }
            if mask.any():
                pieces, n_pieces = ndimage.label(mask, structure=np.ones((3, 3)))
                sizes = ndimage.sum(mask, pieces, range(1, n_pieces + 1))
                y, x = np.nonzero(pieces == np.argmax(sizes) + 1)
                points = np.c_[x, y] * pix.pixel_scale_deg
                spread = (
                    np.linalg.eigvalsh(np.cov(points.T)) if len(points) > 2 else [0, 0]
                )
                row.update(
                    pieces=n_pieces,
                    largest_share=float(sizes.max() / mask.sum()),
                    largest_length_deg=float(4 * np.sqrt(max(spread[-1], 0.0))),
                )
            rows.append(row)
        print(
            f"{label}: {n} windows in {time.time() - start:.0f}s, "
            f"{dataset.invisible_draws} invisible streams redrawn",
            flush=True,
        )
        rows.append(
            {"label": label, "invisible_draws": dataset.invisible_draws, "n": n}
        )
    OUT.mkdir(parents=True, exist_ok=True)
    name = (
        "audit.pkl"
        if tuple(labels) == ("count", "band")
        else f"audit_{'_'.join(labels)}.pkl"
    )
    with open(OUT / name, "wb") as handle:
        pickle.dump(rows, handle)


FAINTER_BY = (
    1.0,
    1.5,
)  # mag/arcsec^2: the "fainter" set, near the real streams' strength


LENGTH_SCAN = {
    "distance": 17.0,
    "width": 0.2,
    "lengths": [4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
    "sb": [32.0, 33.5],
}


def evaluation_streams(sets=("DES 2018", "distance scan")):
    """(set, name, params) injected on the evaluation sky. Sets: "DES 2018"
    (Table 1 parameters), "distance scan", and "fainter" (the DES 2018
    streams made FAINTER_BY mag/arcsec^2 fainter -- the copies are brighter
    than the real streams, so this set reaches their strength and measures
    the detection limit)."""
    from streamgoggles.objects_overlap import DES2018_POPULATIONS

    sp = real_des().stream_parameters_module()
    streams = []
    for name, (width, length, distance, sb) in sp.DES_STREAMS.items():
        age, z = DES2018_POPULATIONS[name]
        streams.append(
            (
                "DES 2018",
                name,
                {
                    "width": width,
                    "length": min(length, MAX_LENGTH),
                    "distance_modulus": distance,
                    "richness": sb,
                    "age": age,
                    "z": z,
                },
            )
        )
    for name, (width, length, distance, sb) in sp.DES_STREAMS.items():
        age, z = DES2018_POPULATIONS[name]
        for fainter in FAINTER_BY:
            streams.append(
                (
                    "fainter",
                    f"{name} +{fainter:g}",
                    {
                        "width": width,
                        "length": min(length, MAX_LENGTH),
                        "distance_modulus": distance,
                        "richness": sb + fainter,
                        "age": age,
                        "z": z,
                    },
                )
            )
    for distance in SCAN["distances"]:
        streams.append(
            (
                "distance scan",
                f"m-M {distance:g}",
                {
                    "width": SCAN["width"],
                    "length": SCAN["length"],
                    "distance_modulus": distance,
                    "richness": SCAN["sb"],
                    "age": 13.0,
                    "z": 0.0002,
                },
            )
        )
    # one stream at m-M 17 and a fixed width, at several lengths, bright and
    # moderate: does the line model miss short streams however bright?
    for sb in LENGTH_SCAN["sb"]:
        for length in LENGTH_SCAN["lengths"]:
            streams.append(
                (
                    "length scan",
                    f"L {length:g} SB {sb:g}",
                    {
                        "width": LENGTH_SCAN["width"],
                        "length": length,
                        "distance_modulus": LENGTH_SCAN["distance"],
                        "richness": sb,
                        "age": 13.0,
                        "z": 0.0002,
                    },
                )
            )
    return [s for s in streams if s[0] in sets]


def shapes(mask, nside):
    """How blob-like a set of HEALPix pixels is: its connected groups and the
    share of its pixels in groups shorter than 2 degrees (length: four
    standard deviations along a group's long axis)."""
    import healpy as hp
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pixels = np.flatnonzero(mask)
    if pixels.size == 0:
        return {"groups": 0, "blob_share": np.nan}
    index = -np.ones(mask.size, dtype=np.int64)
    index[pixels] = np.arange(pixels.size)
    neighbours = hp.get_all_neighbours(nside, pixels).T  # (n, 8)
    rows, cols = np.nonzero(neighbours >= 0)
    targets = index[neighbours[rows, cols]]
    keep = targets >= 0
    graph = coo_matrix(
        (np.ones(keep.sum()), (rows[keep], targets[keep])), shape=(pixels.size,) * 2
    )
    n_groups, labels = connected_components(graph, directed=False)
    vectors = np.asarray(hp.pix2vec(nside, pixels)).T
    short = 0
    for group in range(n_groups):
        members = labels == group
        if members.sum() < 3:
            short += members.sum()
            continue
        spread = np.linalg.eigvalsh(np.cov(vectors[members].T))[-1]
        if np.degrees(4 * np.sqrt(max(spread, 0.0))) < 2.0:
            short += members.sum()
    return {"groups": int(n_groups), "blob_share": float(short / pixels.size)}


def smooth_backgrounds(background, queries, valid=None):
    """{query: the stream-free matched-filter counts smoothed to ~2 degrees}:
    the valid-weighted mean over nside-32 pixels (1.8 deg), interpolated back
    to every pixel. The coarse sums and the coarse valid fraction are
    interpolated apart and divided, so sky outside the footprint weighs
    nothing (not hp.smoothing, whose OpenMP runtime clashes with torch's)."""
    import healpy as hp
    import numpy as np

    if valid is None:
        valid = background.valid_mask_full
    nside = hp.npix2nside(valid.size)
    coarse_weight = hp.ud_grade(valid.astype(float), MF_BACKGROUND_NSIDE)
    lon, lat = hp.pix2ang(nside, np.arange(valid.size), lonlat=True)
    weight = np.maximum(hp.get_interp_val(coarse_weight, lon, lat, lonlat=True), 1e-6)
    smooth = {}
    for q in queries:
        counts = np.where(valid, background.raw_map_full_dict["good"][q], 0.0)
        coarse = hp.ud_grade(counts, MF_BACKGROUND_NSIDE)
        smooth[q] = hp.get_interp_val(coarse, lon, lat, lonlat=True) / weight
    return smooth


def evaluate(config, seeds=SEEDS, train_sky="A", sets=("DES 2018", "distance scan")):
    """Score each seed's model, and their average, on patch B."""
    import healpy as hp
    import numpy as np
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.evaluation.footprint import (
        band_mean_statistics,
        null_band_placements,
        real_track_statistics,
        track_band,
    )
    from streamgoggles.matched_filter import (
        WindowProjection,
        stitch_windows_to_healpix,
        window_to_healpix_indices,
    )
    from streamgoggles.models.unet import UNet
    from streamgoggles.windows import tile_footprint

    rd = real_des()
    sp = rd.stream_parameters_module()
    configure_torch_threads(num_workers=0)
    eval_sky = EVALUATED_ON[train_sky]
    background, injector, pix = build_sky(
        eval_sky, "count", CONFIGS[config].get("image_pix", IMAGE_PIX)
    )
    nside = pix.nside
    valid = background.valid_mask_full
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        valid, nside, tile_size_deg=window_deg, stride_deg=window_deg / 2
    )
    tile_pixels = [window_to_healpix_indices(t, pix, nside)[0] for t in tiles]
    chans = channels()
    norm = normalizer(CONFIGS[config]["normalizer"], chans)
    models = {}
    parts = CONFIGS[config].get("parts", [config])
    seeds = CONFIGS[config].get("seeds", seeds)
    for part in parts:
        for seed in seeds:
            options = {**sp.MODEL, **CONFIGS[part].get("model", {})}
            model = UNet(
                in_channels=QueryDistanceTransform.n_channels,
                out_channels=1,
                **options,
            )
            model.load_state_dict(
                torch.load(model_stem(part, seed, train_sky).with_suffix(".pt"))
            )
            if options["head"] == "identity":  # logits: scored as probabilities
                model = torch.nn.Sequential(model, torch.nn.Sigmoid())
            model.eval()
            key = f"{part} seed{seed}" if len(parts) > 1 else f"seed{seed}"
            models[key] = model
    streams = evaluation_streams(sets)
    queries = sorted(
        {
            min(sp.QUERY_GRID, key=lambda q: abs(q - p["distance_modulus"]))
            for *_, p in streams
        }
    )
    transforms = {
        q: QueryDistanceTransform(
            StreamMapTransform(normalizer=norm, augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=q,
        )
        for q in queries
    }

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(maps_full, tile_ids, query):
        """{scorer: stitched map} for these tiles: each model, and their mean."""
        images = {key: [] for key in [*models, "ensemble"]}
        for i in tile_ids:
            projection = WindowProjection.for_window(tiles[i], pix, valid)
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
            batch = torch.as_tensor(transforms[query](sample)["map_stack"]).unsqueeze(0)
            with torch.no_grad():
                outputs = {k: m(batch)[0, 0].numpy() for k, m in models.items()}
            outputs["ensemble"] = np.mean(list(outputs.values()), axis=0)
            for key, out in outputs.items():
                images[key].append(np.where(projection.valid, out, np.nan))
        chosen = [tiles[i] for i in tile_ids]
        return {
            k: stitch_windows_to_healpix(v, chosen, pix, nside)[0]
            for k, v in images.items()
        }

    # The stream-free patch: every model's output, the reference its
    # false-alarm rates and null bands are measured on, and how blob-like its
    # confident answers are.
    background_maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
    # False-alarm rates and null bands on the stream-free sky: the whole
    # patch, or a fold's calibration sky (its training sky still holds the
    # leftover structure the calibration mask removes).
    calibration = valid.copy()
    if "fold" in SKIES[eval_sky]:
        calibration &= hp.read_map(rd.CALIBRATION_MASK).astype(bool)
    baseline, reference, flagged = {}, {}, {}
    quiet = []
    for q in queries:
        stitched = predict(background_maps, range(len(tiles)), q)
        for key, image in stitched.items():
            image = np.where(valid, image, np.nan)
            baseline[(key, q)] = image
            ref = np.sort(image[np.isfinite(image) & calibration])
            reference[(key, q)] = ref
            rate = np.full(image.size, np.nan)
            finite = np.isfinite(image)
            rate[finite] = (
                ref.size - np.searchsorted(ref, image[finite], side="left")
            ) / ref.size
            flagged[(key, q)] = finite & (rate <= TARGET_FALSE_ALARM_RATE)
            quiet.append(
                {
                    "config": config,
                    "scorer": key,
                    "query": q,
                    "above_half": float((ref > 0.5).mean()),
                    **shapes(finite & calibration & (image > 0.5), nside),
                    "threshold": float(
                        ref[int((1 - TARGET_FALSE_ALARM_RATE) * ref.size)]
                    ),
                }
            )
    calibration &= np.isfinite(baseline[("ensemble", queries[0])])

    # The matched filter's own search, as a classical analysis runs it: counts
    # minus a smooth local background (the stream-free counts smoothed to
    # about 2 degrees within the valid sky: the valid-weighted mean over
    # nside-32 pixels, 1.8 degrees, interpolated back to every pixel -- not
    # hp.smoothing, whose OpenMP runtime clashes with torch's in one process).
    # The background model is taken from the
    # stream-free map, which spares a wide stream the few tens of percent of
    # its excess a smoothing of the injected map would absorb -- slightly in
    # the matched filter's favour.
    smooth_background = smooth_backgrounds(background, queries)

    rows = []
    start = time.time()
    for index, (kind, name, stream) in enumerate(streams):
        rng = np.random.default_rng([EVAL_SEED, index])
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - stream["distance_modulus"]))
        good = next(
            i
            for i, c in enumerate(chans)
            if c["filter"] == "good" and c["distance_modulus"] == query
        )
        candidates = np.flatnonzero(calibration)
        placed = 0
        tries = 0
        while placed < N_PLACEMENTS and tries < 50 * N_PLACEMENTS:
            tries += 1
            ra, dec = hp.pix2ang(nside, int(rng.choice(candidates)), lonlat=True)
            rotation = float(rng.uniform(0, 360))
            band = track_band(
                [rd.great_circle(float(ra), float(dec), rotation, stream["length"])],
                stream["width"],
                nside,
            )
            if band.sum() == 0 or (band & calibration).sum() / band.sum() < 0.9:
                continue
            band &= calibration
            params = {
                "morphology": "uniform",
                "isochrone_model": sp.ISOCHRONE_FAMILY,
                "orientation": rotation,
                **stream,
            }
            sky = injector.inject_streams_full_sky(
                [params],
                np.random.default_rng([EVAL_SEED, index, placed]),
                centers=[(float(ra), float(dec))],
            )
            covering = [i for i, px in enumerate(tile_pixels) if band[px].any()]
            stitched = predict(sky["map_full"], covering, query)
            at = np.flatnonzero(band)
            stream_stars = sky["stream_raw_full"][good][band].sum()
            background_stars = background.raw_map_full_dict["good"][query][band].sum()
            # The integrated test: the band's mean against the same shape on
            # stream-free sky, one set of null bands for every map scored --
            # the network's outputs and the matched filter's own counts.
            usable = calibration & np.isfinite(baseline[("ensemble", query)])
            placements = null_band_placements(
                band,
                usable,
                np.random.default_rng([EVAL_SEED, index, placed, 99]),
                N_NULL_BANDS,
            )
            exceeds_all = 1.0 / (len(placements) + 1) + 1e-12
            counts = background.raw_map_full_dict["good"][query]
            matched = band_mean_statistics(
                sky["map_full"][good] - smooth_background[query],
                at,
                counts - smooth_background[query],
                placements,
            )
            for key, image in stitched.items():
                integrated = band_mean_statistics(
                    np.where(band, image, baseline[(key, query)]),
                    at,
                    baseline[(key, query)],
                    placements,
                )
                # The same test on the logits: a mean of probabilities is ruled
                # by the few noisy peaks, a sum of logits adds up the local
                # evidence (for a calibrated output, the local log-likelihood
                # ratio) as the matched filter adds up counts.
                integrated_logit = band_mean_statistics(
                    logit(np.where(band, image, baseline[(key, query)])),
                    at,
                    logit(baseline[(key, query)]),
                    placements,
                )
                values = image[at]
                ref = reference[(key, query)]
                ok = np.isfinite(values)
                hit = np.zeros(at.size, bool)
                hit[ok] = (
                    ref.size - np.searchsorted(ref, values[ok], side="left")
                ) / ref.size <= TARGET_FALSE_ALARM_RATE
                flags = flagged[(key, query)].copy()
                flags[at] = hit
                scored = np.isfinite(baseline[(key, query)])
                stats = real_track_statistics(
                    flags,
                    scored,
                    band,
                    calibration,
                    np.random.default_rng([EVAL_SEED, index, placed, len(key)]),
                    N_NULL_BANDS,
                )
                rows.append(
                    {
                        "config": config,
                        "scorer": key,
                        "set": kind,
                        "stream": name,
                        "distance_modulus": stream["distance_modulus"],
                        "query": query,
                        "placement": placed,
                        "input_snr": stream_stars / np.sqrt(max(background_stars, 1.0)),
                        "n_flagged": stats["n_flagged"],
                        "snr": stats["snr"],
                        "detected": bool(
                            stats["n_flagged"] >= MIN_FLAGGED_PIXELS
                            and stats["snr"] >= MIN_SNR
                        ),
                        "integrated_snr": integrated["snr"],
                        "integrated_p": integrated["p_value"],
                        "integrated_detected": bool(
                            integrated["p_value"] <= exceeds_all
                        ),
                        "integrated_logit_snr": integrated_logit["snr"],
                        "integrated_logit_detected": bool(
                            integrated_logit["p_value"] <= exceeds_all
                        ),
                        "matched_filter_snr": matched["snr"],
                        "matched_filter_p": matched["p_value"],
                        "matched_filter_detected": bool(
                            matched["p_value"] <= exceeds_all
                        ),
                    }
                )
            placed += 1
        print(
            f"{config} {name}: {placed} placements, {time.time() - start:.0f}s",
            flush=True,
        )
    out = result_dir(train_sky)
    out.mkdir(parents=True, exist_ok=True)
    name = config.replace("/", "_")
    if tuple(sets) == ("DES 2018", "distance scan"):
        pd.DataFrame(rows).to_csv(out / f"evaluation_{name}.csv", index=False)
        pd.DataFrame(quiet).to_csv(out / f"quiet_{name}.csv", index=False)
    else:  # an extra set: its own file, read with the others by set
        suffix = "_".join(x.replace(" ", "-") for x in sets)
        pd.DataFrame(rows).to_csv(out / f"evaluation_{name}__{suffix}.csv", index=False)


HOUGH_NULL_WINDOWS = 600  # stream-free windows per queried distance
HOUGH_FALSE_ALARM = 0.01  # blind test: share of stream-free windows with a line found
HOUGH_MIN_LENGTH_DEG = 4.0  # a window scores a copy if it holds this much of it
# A stream-free window must lie on calibration sky: at least this share of
# its valid pixels. The per-pixel false-alarm rates are measured on
# calibration pixels alone; a window reaching into the training-only sky
# meets the leftover structure the calibration mask removes -- Sagittarius's
# wing past its 6-degree mask, a real line the line model rightly finds.
HOUGH_NULL_CALIBRATION = 0.95
# The lines of a per-pixel model scored as lines (configurations with "lines")
HOUGH_GRID = {"n_theta": 90, "rho_step": 2.0, "min_pixels": 20}


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
    import healpy as hp
    import numpy as np
    import pandas as pd
    import torch

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.matched_filter import WindowProjection
    from streamgoggles.models.hough import hough_target, line_counts
    from streamgoggles.windows import Window

    configure_torch_threads(num_workers=0)
    rd = real_des()
    sp = rd.stream_parameters_module()
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    sky = window_level_sky(EVALUATED_ON[train_sky], image_pix)
    background, injector, pix = sky.background, sky.injector, sky.pix
    nside, valid, window_deg = sky.nside, sky.valid, sky.window_deg
    tile_pixels, projections = sky.tile_pixels, sky.projections
    calibration = sky.calibration
    chans = channels()
    norm = normalizer(CONFIGS[config]["normalizer"], chans)
    models = []
    if "hough" in CONFIGS[config]:  # a line model: one answer per line
        factory, _ = hough_parts(config)
        for seed in CONFIGS[config].get("seeds", seeds):
            model = factory(QueryDistanceTransform.n_channels)
            model.load_state_dict(
                torch.load(model_stem(config, seed, train_sky).with_suffix(".pt"))
            )
            models.append(model.eval())
        grid = models[0].hough.grid
    else:  # a per-pixel ensemble, its answer summed along lines
        from streamgoggles.models import build_model
        from streamgoggles.models.hough import HoughLines

        for part in CONFIGS[config].get("parts", [config]):
            for seed in CONFIGS[config].get("seeds", seeds):
                model = build_model(
                    QueryDistanceTransform.n_channels,
                    **{**sp.MODEL, **CONFIGS[part].get("model", {})},
                )
                model.load_state_dict(
                    torch.load(model_stem(part, seed, train_sky).with_suffix(".pt"))
                )
                models.append(model.eval())
        grid = HoughLines(image_pix, image_pix, **HOUGH_GRID)
    min_length_pix = HOUGH_MIN_LENGTH_DEG / pix.pixel_scale_deg
    streams = evaluation_streams(sets)
    queries = sorted(
        {
            min(sp.QUERY_GRID, key=lambda q: abs(q - p["distance_modulus"]))
            for *_, p in streams
        }
    )
    transforms = {
        q: QueryDistanceTransform(
            StreamMapTransform(normalizer=norm, augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=q,
        )
        for q in queries
    }
    good_at = {
        q: next(
            i
            for i, c in enumerate(chans)
            if c["filter"] == "good" and c["distance_modulus"] == q
        )
        for q in queries
    }
    smooth_background = smooth_backgrounds(background, queries)

    def scores(maps_full, windows, query):
        """Line scores of the model and of the matched filter, (n, n_theta,
        n_rho) each, NaN on lines too short to count."""
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
            smooth = projection.image(
                np.where(valid_at, smooth_background[query][projection.pixnums], 0.0)
            )
            residual = (stack[good_at[query]] - smooth) / np.sqrt(
                np.maximum(smooth, 0.5)
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

    # Stream-free windows: centred on random calibration pixels, at least
    # half valid, lying on calibration sky, unrotated like the tiles.
    rng = np.random.default_rng([EVAL_SEED, 7])
    candidates = np.flatnonzero(calibration)
    null_windows = []
    while len(null_windows) < HOUGH_NULL_WINDOWS:
        ra, dec = hp.pix2ang(nside, int(rng.choice(candidates)), lonlat=True)
        window = Window(
            center_ra=float(ra),
            center_dec=float(dec),
            width_deg=window_deg,
            height_deg=window_deg,
        )
        projection = WindowProjection.for_window(window, pix, valid)
        if projection.valid.mean() < 0.5:
            continue
        on_calibration = projection.image(calibration[projection.pixnums].astype(float))
        share = (on_calibration > 0.99)[projection.valid].mean()
        if share >= HOUGH_NULL_CALIBRATION:
            null_windows.append(projection)
    background_maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
    null, thresholds, null_rows, null_best = {}, {}, [], {}
    start = time.time()
    for q in queries:
        null[q] = scores(background_maps, null_windows, q)
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
                length = line_counts(inside, grid).max()
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
                answers = scores(injected["map_full"], scored, query)
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
    factory, _ = hough_parts(config)
    models = []
    for seed in CONFIGS[config].get("seeds", SEEDS):
        model = factory(QueryDistanceTransform.n_channels)
        model.load_state_dict(
            torch.load(model_stem(config, seed, train_sky).with_suffix(".pt"))
        )
        models.append(model.eval())
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

    factory, _ = hough_parts(config)
    line_models = []
    for s in CONFIGS[config].get("seeds", SEEDS):
        model = factory(QueryDistanceTransform.n_channels)
        model.load_state_dict(
            torch.load(model_stem(config, s, train_sky).with_suffix(".pt"))
        )
        line_models.append(model.eval())
    grid = line_models[0].hough.grid
    pixel_models = []
    for s in CONFIGS["count/window x4"]["seeds"]:
        model = build_model(QueryDistanceTransform.n_channels, **sp.MODEL)
        model.load_state_dict(
            torch.load(model_stem("count/window", s, train_sky).with_suffix(".pt"))
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
    return "" if mask_objects else "__unmasked"


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
    import torch
    from scipy import ndimage

    from streamgoggles.datasets.stream_map_dataset import configure_torch_threads
    from streamgoggles.datasets.transforms import (
        QueryDistanceTransform,
        StreamMapTransform,
    )
    from streamgoggles.matched_filter import WindowProjection, _tangent_plane_radec
    from streamgoggles.objects_overlap import get_footprint, spatial_fold
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
        valid = valid & ~rd.object_mask(nside, max_dwarf_mv=LINE_SKY_DWARF_MV)
    usable = get_footprint("des_yr6_inference", nside=nside)[0]
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        usable & valid, nside, tile_size_deg=window_deg, stride_deg=window_deg / 2
    )
    chans = channels()
    name = config.replace("/", "_")
    factory, _ = hough_parts(config)
    models, levels = {}, {}
    for train_sky in ("fold0", "fold1"):
        models[train_sky] = []
        for seed in CONFIGS[config].get("seeds", SEEDS):
            model = factory(QueryDistanceTransform.n_channels)
            model.load_state_dict(
                torch.load(model_stem(config, seed, train_sky).with_suffix(".pt"))
            )
            models[train_sky].append(model.eval())
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
    grid = models["fold0"][0].hough.grid
    queries = list(sp.QUERY_GRID)
    norm = normalizer(CONFIGS[config]["normalizer"], chans)
    transforms = {
        q: QueryDistanceTransform(
            StreamMapTransform(normalizer=norm, augment=False),
            query_grid=sp.QUERY_GRID,
            step=sp.STEP,
            query=q,
        )
        for q in queries
    }
    good_at = {
        q: next(
            i
            for i, c in enumerate(chans)
            if c["filter"] == "good" and c["distance_modulus"] == q
        )
        for q in queries
    }
    smooth_background = smooth_backgrounds(background, queries, valid)
    maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
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
        valid_at = valid[projection.pixnums]
        stack = np.stack(
            [
                projection.image(np.where(valid_at, m[projection.pixnums], 0.0))
                for m in maps
            ]
        ).astype(np.float32)
        sample = {
            "map_stack": stack,
            "label_stack": np.zeros_like(stack),
            "valid_mask": projection.valid,
            "params": {},
            "metadata": {"channels": chans},
        }
        ra_grid, dec_grid = _tangent_plane_radec(
            dataclasses.replace(
                pix,
                center_ra=tile.center_ra,
                center_dec=tile.center_dec,
                rotation_deg=tile.rotation_deg,
            )
        )
        with torch.no_grad():
            x = torch.as_tensor(
                np.stack([transforms[q](dict(sample))["map_stack"] for q in queries])
            )
            network = np.mean(
                [torch.sigmoid(m(x))[:, 0].numpy() for m in models[scoring]], 0
            )
        for k, q in enumerate(queries):
            smooth = projection.image(
                np.where(valid_at, smooth_background[q][projection.pixnums], 0.0)
            )
            residual = (stack[good_at[q]] - smooth) / np.sqrt(np.maximum(smooth, 0.5))
            matched = grid(np.where(projection.valid, residual, 0.0))
            for scorer, values in (
                ("network", network[k]),
                ("matched filter", matched),
            ):
                values = np.where(grid.valid, values, -np.inf)
                level = levels[(scoring, scorer, round(q, 1))]
                peaks = (values == ndimage.maximum_filter(values, size=5)) & (
                    values >= level
                )
                for t, r in zip(*np.nonzero(peaks), strict=True):
                    theta, rho = grid.thetas[t], grid.rhos[r]
                    line = np.abs(xx * np.cos(theta) + yy * np.sin(theta) - rho) < 1.0
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
                            "score": float(values[t, r]),
                            "level": float(level),
                            "level_half": half[(scoring, scorer, round(q, 1))],
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
    combination = pd.DataFrame(combination)
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
    the track as the paper draws it. With the per-pixel network's result
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
    pixel_test = pd.read_csv(rd.DETECTIONS)
    table["along"] = ""
    rows = []
    for stream, (width, _, distance, _) in sp.DES_STREAMS.items():
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - distance))
        track = _unit(*des2018_arc(stream, n=600))
        tolerance = np.radians(max(TRACK_TOLERANCE_DEG, 2 * width))
        along = np.array(
            [
                np.sum(np.arccos(np.clip((seg @ track.T).max(1), -1, 1)) <= tolerance)
                * s
                for seg, s in zip(segments, step, strict=True)
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
            DOC_FIGURES / f"line_sky_{stem}{suffix}.png", dpi=100, bbox_inches="tight"
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
                near = tree.query_ball_point(_unit(*track), chord)
                hits = np.unique(np.concatenate([np.asarray(h, int) for h in near]))
                along = np.bincount(owner[hits], minlength=len(steps)) * steps
                counts.append(int((along >= MIN_ALONG_DEG).sum()))
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
        DOC_FIGURES / f"line_sky_streams{suffix}.png", dpi=110, bbox_inches="tight"
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

    def overlap(track_points, reference, tolerance_deg):
        """Degrees of the track within tolerance of a reference track."""
        close = (track_points @ reference.T).max(1) >= np.cos(np.radians(tolerance_deg))
        return close.sum() * 0.25

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
            degrees = overlap(arc, reference, max(TRACK_TOLERANCE_DEG, 2 * width))
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
            degrees = overlap(arc, reference, GALSTREAMS_TOLERANCE_DEG)
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
        DOC_FIGURES / f"line_sky_tracks{suffix}.png", dpi=100, bbox_inches="tight"
    )
    plt.close(fig)


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
        prefix = "hough" if kind == "lines" else "evaluation"
        name = config.replace("/", "_")
        parts = [
            folder / f"{prefix}_{name}.csv",
            folder / f"{prefix}_{name}__fainter.csv",
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


def label_figures():
    """The two labels: how often each is a scatter of clumps, by distance; and
    both on the same injected streams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from streamgoggles.datasets.transforms import WindowNormalizer

    rd = real_des()
    sp = rd.stream_parameters_module()
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    with open(OUT / "audit.pkl", "rb") as handle:
        rows = [r for r in pickle.load(handle) if "positive" in r]
    audit_rows = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = [15, 16, 17, 18, 19.01]
    for label, colour in (("count", "#9e9e9e"), ("band", "#1f3b73")):
        mine = audit_rows[(audit_rows.label == label) & (audit_rows.positive > 0)]
        clumps = mine.groupby(pd.cut(mine.distance_modulus, bins), observed=True).apply(
            lambda g: (g.largest_length_deg < 2).mean()
        )
        centres = [0.5 * (b.left + b.right) for b in clumps.index]
        ax.plot(
            centres, clumps.values, "o-", color=colour, lw=2, label=f"{label} label"
        )
    ax.set_xlabel("stream distance modulus")
    ax.set_ylabel("labels whose largest piece is < 2 deg")
    ax.set_ylim(-0.03, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(DOC_FIGURES / "label_clumps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # the same streams, both labels: each stream realized and windowed once
    # (band injector), the count label computed on the same window
    _, band_injector, _ = build_sky("A", "band")
    examples = [
        {"width": 0.8, "distance_modulus": 15.4, "richness": 33.0},
        {"width": 0.3, "distance_modulus": 16.2, "richness": 33.5},
        {"width": 0.2, "distance_modulus": 17.0, "richness": 33.0},
        {"width": 0.5, "distance_modulus": 18.2, "richness": 33.5},
    ]
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    for row, extra in zip(axes, examples, strict=True):
        params = {
            "morphology": "uniform",
            "length": 15.0,
            "age": 12.0,
            "z": 0.0003,
            "isochrone_model": sp.ISOCHRONE_FAMILY,
            "distance_gradient": 0.0,
            **extra,
        }
        sample = band_injector.inject_single_stream(params, np.random.default_rng(11))
        channel = next(
            i
            for i, c in enumerate(sample.metadata["channels"])
            if c["filter"] == "good"
            and np.isclose(
                c["distance_modulus"],
                min(
                    sp.CHANNEL_DISTANCES,
                    key=lambda d: abs(d - extra["distance_modulus"]),
                ),
            )
        )
        valid = sample.valid_mask
        normed = WindowNormalizer()(sample.map_stack, valid)[channel]
        # the count label on the same stars and window: the same seed repeats
        # the realization and placement (the first placement was accepted)
        from streamgoggles.windows import Window

        window = Window(**sample.metadata["window"])
        detected, _, _ = band_injector._realize_and_inject(
            params, np.random.default_rng(11)
        )
        _, raw, _, _ = band_injector._build_window_channels(detected, window)
        count = raw[channel] > 1.0
        panels = (
            (np.where(valid, normed, np.nan), "input (m-M channel)", "Greys"),
            (np.where(valid, count, np.nan), "count label", "Blues"),
            (
                np.where(valid, sample.label_stack[channel], np.nan),
                "band label",
                "Blues",
            ),
        )
        for ax, (image, title, cmap) in zip(row, panels, strict=True):
            lim = np.nanpercentile(np.abs(image), 99) if cmap == "Greys" else 1
            ax.imshow(
                image,
                origin="lower",
                cmap=cmap,
                vmin=-lim if cmap == "Greys" else 0,
                vmax=lim,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{title}: m-M {extra['distance_modulus']}, w {extra['width']}",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / "label_examples.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# One figure per question, each against the reference configuration (the
# first training's: count label, window normalization, two quick models).
COMPARISONS = {
    "labels": ["count/window", "band/window", "band5/window", "bandseg/window"],
    "normalization": [
        "count/window",
        "count/poisson",
        "band/window",
        "band/poisson",
        "band/decoy",
    ],
    "models": ["count/window", "count/window x4", "count+band/window"],
    "window_size": ["count/window", "count/window 128px"],
    "models_and_length": [
        "count/window",
        "count/window x4",
        "count/window x6",
        "count/window long",
    ],
}
# the reference in dark grey, the others in a fixed order
COMPARISON_COLOURS = ["#4d4d4d", "#1f6fb4", "#e07b39", "#2e8b57", "#8e44ad"]


def comparison_figures(train_sky="fold0"):
    """For each question in COMPARISONS: DES 2018 copies recovered, recovery
    against distance, and how often the ensemble fires on empty sky."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    folder = result_dir(train_sky)
    data = pd.concat(
        [pd.read_csv(p) for p in sorted(folder.glob("evaluation_*.csv"))],
        ignore_index=True,
    )
    quiet = pd.concat(
        [pd.read_csv(p) for p in sorted(folder.glob("quiet_*.csv"))], ignore_index=True
    )
    for question, configs in COMPARISONS.items():
        configs = [c for c in configs if c in set(data.config)]
        if len(configs) < 2:
            continue
        colours = dict(zip(configs, COMPARISON_COLOURS, strict=False))
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.2, 1]}
        )
        for i, config in enumerate(configs):
            mine = data[(data.config == config) & (data.set == "DES 2018")]
            ensemble = mine[mine.scorer == "ensemble"].detected.mean()
            seeds = [
                g.detected.mean()
                for _, g in mine[mine.scorer != "ensemble"].groupby("scorer")
            ]
            axes[0].bar(i, ensemble, color=colours[config], width=0.6)
            axes[0].plot([i] * len(seeds), seeds, "o", color="black", ms=4)
            axes[0].text(i, ensemble + 0.02, f"{ensemble:.0%}", ha="center", fontsize=9)
            scan = (
                data[
                    (data.config == config)
                    & (data.set == "distance scan")
                    & (data.scorer == "ensemble")
                ]
                .groupby("distance_modulus")
                .detected.mean()
            )
            axes[1].plot(
                scan.index, scan.values, "o-", color=colours[config], lw=2, label=config
            )
            empty = quiet[(quiet.config == config) & (quiet.scorer == "ensemble")]
            axes[2].bar(
                i, 100 * empty.above_half.median(), color=colours[config], width=0.6
            )
        for ax in (axes[0], axes[2]):
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, fontsize=8, rotation=25, ha="right")
        axes[0].set_ylim(0, 1.08)
        axes[0].set_ylabel("DES 2018 copies recovered")
        axes[0].set_title("bar: ensemble, dots: single models", fontsize=9)
        axes[1].set_ylim(-0.03, 1.05)
        axes[1].set_xlabel("distance modulus (0.3 deg wide, 10 deg long, SB 33)")
        axes[1].set_ylabel("recovered (ensemble)")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
        axes[2].set_ylabel("% of empty sky above 0.5")
        axes[2].set_title("false alarms", fontsize=9)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(question.replace("_", " "), fontsize=11)
        fig.tight_layout()
        fig.savefig(
            DOC_FIGURES / f"compare_{question}_{train_sky}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)


SENSITIVITY_CONFIGS = [
    "count/window",
    "count/window x4",
    "count/window x6",
    "count/window long",
    "count/window 128px",
]
SNR_EDGES = [0, 4, 6, 8, 10, 13, 17, 22, 30, 60]


def half_recovery_snr(rows):
    """Input S/N at which half the copies are found: binned recovery,
    interpolated where it crosses 0.5 (NaN if it never does)."""
    import numpy as np
    import pandas as pd

    rate = rows.groupby(pd.cut(rows.input_snr, SNR_EDGES), observed=True).detected.agg(
        ["mean", "size"]
    )
    rate = rate[rate["size"] >= 4]
    centres = np.array([np.sqrt(max(b.left, 1) * b.right) for b in rate.index])
    values = rate["mean"].to_numpy()
    for i in range(1, len(values)):
        if values[i - 1] < 0.5 <= values[i]:
            f = (0.5 - values[i - 1]) / (values[i] - values[i - 1])
            return float(
                np.exp(np.log(centres[i - 1]) + f * np.log(centres[i] / centres[i - 1]))
            )
    return float("nan") if values.size == 0 or values[0] < 0.5 else float(centres[0])


def sensitivity(train_sky="fold0"):
    """Recovery against input S/N for each sensitivity configuration, near and
    far, from the DES 2018 and fainter copies (ensembles), and its 50% point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    folder = result_dir(train_sky)
    data = pd.concat(
        [pd.read_csv(p) for p in sorted(folder.glob("evaluation_*.csv"))],
        ignore_index=True,
    )
    data = data[(data.scorer == "ensemble") & data.set.isin(["DES 2018", "fainter"])]
    configs = [
        c
        for c in SENSITIVITY_CONFIGS
        if c in set(data.config) and (data[data.config == c].set == "fainter").any()
    ]
    colours = dict(zip(configs, COMPARISON_COLOURS, strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    rows = []
    for ax, (label, near) in zip(
        axes, (("m−M < 16.5", True), ("m−M ≥ 16.5", False)), strict=True
    ):
        for config in configs:
            mine = data[
                (data.config == config) & ((data.distance_modulus < 16.5) == near)
            ]
            rate = mine.groupby(
                pd.cut(mine.input_snr, SNR_EDGES), observed=True
            ).detected.agg(["mean", "size"])
            rate = rate[rate["size"] >= 4]
            centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
            ax.plot(
                centres, rate["mean"], "o-", color=colours[config], lw=2, label=config
            )
            rows.append(
                {
                    "config": config,
                    "streams": label,
                    "half_recovery_snr": half_recovery_snr(mine),
                    "recovered": mine.detected.mean(),
                    "copies": len(mine),
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
    axes[0].set_ylabel("copies recovered (ensemble)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "sensitivity: DES 2018 copies at full and reduced brightness", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(
        DOC_FIGURES / f"compare_sensitivity_{train_sky}.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "sensitivity.csv", index=False)
    print(
        table.pivot(index="config", columns="streams", values="half_recovery_snr")
        .round(1)
        .to_string()
    )
    print(
        table.pivot(index="config", columns="streams", values="recovered")
        .round(2)
        .to_string()
    )


# Why the network is half as sensitive as its input: one lever per figure,
# each against the reference (two count/window models) and the matched filter
# (the ceiling: the same copies, the band's background-subtracted counts).
LEVERS = {
    "depth": ("count/window deep", "depth 4 (field: the window)"),
    "loss": ("count/window bce", "cross-entropy loss"),
    "label": ("band/window", "band label"),
    "faint_count": ("count/window faint", "training down to 36 mag/arcsec²"),
    "faint_band": ("band/window faint", "band label, training down to 36"),
}


def levers(train_sky="fold0"):
    """compare_lever_{name}_{sky}.png and levers.csv: half-recovery input S/N
    of the reference, the lever and the matched filter, near and far."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    folder = result_dir(train_sky)
    data = pd.concat(
        [pd.read_csv(p) for p in sorted(folder.glob("evaluation_*.csv"))],
        ignore_index=True,
    )
    data = data[(data.scorer == "ensemble") & data.set.isin(["DES 2018", "fainter"])]
    reference = "count/window"
    rows = []
    for lever, (config, name) in LEVERS.items():
        mine = data[data.config == config]
        if not (mine.set == "fainter").any() or "matched_filter_detected" not in mine:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
        for ax, (label, near) in zip(
            axes, (("m−M < 16.5", True), ("m−M ≥ 16.5", False)), strict=True
        ):
            curves = (
                (
                    reference,
                    "detected",
                    "reference (count label, depth 2, Dice)",
                    {"color": "#4d4d4d", "ls": "-"},
                ),
                (config, "detected", name, {"color": "#1f6fb4", "ls": "-"}),
                (
                    config,
                    "matched_filter_detected",
                    "matched filter alone",
                    {"color": "#9e9e9e", "ls": "--"},
                ),
            )
            for which, test, legend, style in curves:
                part = data[
                    (data.config == which) & ((data.distance_modulus < 16.5) == near)
                ].dropna(subset=[test])
                rate = part.groupby(pd.cut(part.input_snr, SNR_EDGES), observed=True)[
                    test
                ].agg(["mean", "size"])
                rate = rate[rate["size"] >= 4]
                centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
                ax.plot(
                    centres,
                    rate["mean"].astype(float),
                    "o-",
                    lw=2,
                    label=legend,
                    **style,
                )
                rows.append(
                    {
                        "lever": lever,
                        "curve": legend,
                        "streams": label,
                        "half_recovery_snr": half_recovery_snr(
                            part.assign(detected=part[test].astype(float))
                        ),
                        "recovered": part[test].astype(float).mean(),
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
        axes[0].set_ylabel("copies found (two models)")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(f"{name} against the reference", fontsize=11)
        fig.tight_layout()
        fig.savefig(
            DOC_FIGURES / f"compare_lever_{lever}_{train_sky}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)
    table = pd.DataFrame(rows).drop_duplicates(subset=["curve", "streams"])
    table.to_csv(folder / "levers.csv", index=False)
    print(table.round(2).to_string(index=False))


TESTS = {
    "detected": "network, >= 20 pixels at FAR 1e-3 (current test)",
    "integrated_detected": "network, mean output along the band",
    "integrated_logit_detected": "network, mean logit along the band",
    "matched_filter_detected": "matched filter, background-subtracted counts along the band",
}
TEST_STYLES = {
    "detected": {"color": "#1f6fb4", "ls": "-"},
    "integrated_detected": {"color": "#e07b39", "ls": "-"},
    "integrated_logit_detected": {"color": "#2e8b57", "ls": "-"},
    "matched_filter_detected": {"color": "#4d4d4d", "ls": "--"},
}


def detection_tests(train_sky="fold0", config="count/window x4"):
    """The same copies under three tests, at the same false-positive level
    for the two integrated ones (the band beats all its null bands):
    recovery against input S/N, near and far, and the half-recovery points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    folder = result_dir(train_sky)
    data = pd.concat(
        [pd.read_csv(p) for p in sorted(folder.glob("evaluation_*.csv"))],
        ignore_index=True,
    )
    if "integrated_detected" not in data:
        return
    data = data[
        (data.scorer == "ensemble")
        & data.set.isin(["DES 2018", "fainter"])
        & data.integrated_detected.notna()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    rows = []
    mine_all = data[data.config == config]
    for ax, (label, near) in zip(
        axes, (("m−M < 16.5", True), ("m−M ≥ 16.5", False)), strict=True
    ):
        mine = mine_all[(mine_all.distance_modulus < 16.5) == near]
        for test, description in TESTS.items():
            if test not in mine or mine[test].isna().all():
                continue
            rate = mine.groupby(pd.cut(mine.input_snr, SNR_EDGES), observed=True)[
                test
            ].agg(["mean", "size"])
            rate = rate[rate["size"] >= 4]
            centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
            ax.plot(
                centres, rate["mean"], "o", lw=2, label=description, **TEST_STYLES[test]
            )
            renamed = mine.assign(detected=mine[test].astype(bool))
            rows.append(
                {
                    "config": config,
                    "streams": label,
                    "test": test,
                    "half_recovery_snr": half_recovery_snr(renamed),
                    "recovered": float(mine[test].mean()),
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
    fig.suptitle(f"three tests on the same copies ({config})", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        DOC_FIGURES / f"detection_tests_{train_sky}.png", dpi=120, bbox_inches="tight"
    )
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "detection_tests.csv", index=False)
    print(
        table.pivot(index="test", columns="streams", values="half_recovery_snr")
        .round(1)
        .to_string()
    )
    print(
        table.pivot(index="test", columns="streams", values="recovered")
        .round(2)
        .to_string()
    )
    full = mine_all[mine_all.set == "DES 2018"]
    print(full.groupby("stream")[list(TESTS)].mean().round(2).to_string())


def figures(train_sky="A"):
    """Recovery per configuration: DES 2018 streams, and against distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frames = [
        pd.read_csv(p) for p in sorted(result_dir(train_sky).glob("evaluation_*.csv"))
    ]
    data = pd.concat(frames, ignore_index=True)
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    order = [c for c in CONFIGS if c in set(data.config)]
    colours = {
        "count/window": "#9e9e9e",
        "count/decoy": "#5b5b5b",
        "band/window": "#8fb4e3",
        "band/decoy": "#1f3b73",
        "count/poisson": "#d9a066",
        "band/poisson": "#b5651d",
        "band5/window": "#4a7fc1",
        "bandseg/window": "#2e8b57",
        "count+band/window": "#6a3d9a",
        "count/window x4": "#555555",
        "count/window 128px": "#c0392b",
        "count/window x6": "#1b1b1b",
        "count/window long": "#16a085",
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    ax = axes[0]
    des = data[data.set == "DES 2018"]
    for i, config in enumerate(order):
        mine = des[des.config == config]
        ensemble = mine[mine.scorer == "ensemble"].detected.mean()
        seeds = [
            g.detected.mean()
            for k, g in mine[mine.scorer != "ensemble"].groupby("scorer")
        ]
        ax.bar(i, ensemble, color=colours[config], width=0.6)
        ax.plot([i] * len(seeds), seeds, "o", color="black", ms=4)
        ax.text(i, ensemble + 0.02, f"{ensemble:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of copies recovered")
    ax.set_title(
        f"DES 2018 streams, scored on {EVALUATED_ON[train_sky]} (bar: ensemble, dots: seeds)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    scan = data[(data.set == "distance scan") & (data.scorer == "ensemble")]
    for config in order:
        mine = scan[scan.config == config].groupby("distance_modulus").detected.mean()
        ax.plot(
            mine.index, mine.values, "o-", color=colours[config], label=config, lw=2
        )
    ax.set_xlabel("distance modulus (width 0.3 deg, 10 deg long, SB 33)")
    ax.set_ylabel("fraction recovered (ensemble)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("recovery against distance", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC_FIGURES / f"recovery_{train_sky}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # What each configuration answers on sky with no stream: how often it is
    # confident (output > 0.5), and how much of that is blobs (< 2 degrees).
    quiet = pd.concat(
        [pd.read_csv(p) for p in sorted(result_dir(train_sky).glob("quiet_*.csv"))],
        ignore_index=True,
    )
    quiet = quiet[quiet.scorer == "ensemble"]
    if "blob_share" in quiet:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, config in enumerate(order):
            mine = quiet[quiet.config == config]
            axes[0].bar(
                i, 100 * mine.above_half.median(), color=colours[config], width=0.6
            )
            axes[1].bar(
                i, 100 * mine.blob_share.median(), color=colours[config], width=0.6
            )
        axes[0].set_ylabel("% of stream-free pixels above 0.5")
        axes[1].set_ylabel("% of those in blobs shorter than 2 deg")
        for ax in axes:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, fontsize=8, rotation=35, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_title("how often the ensemble fires on empty sky", fontsize=10)
        axes[1].set_title("how blob-like those answers are", fontsize=10)
        fig.tight_layout()
        fig.savefig(
            DOC_FIGURES / f"false_alarms_{train_sky}.png", dpi=120, bbox_inches="tight"
        )
        plt.close(fig)

    table = (
        data[data.scorer == "ensemble"]
        .groupby(["config", "set"])
        .detected.mean()
        .unstack("set")
        .reindex(order)
    )
    print(table.round(2).to_string())
    per_stream = (
        des[des.scorer == "ensemble"]
        .groupby(["stream", "config"])
        .detected.mean()
        .unstack("config")[order]
    )
    print(per_stream.round(2).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "step",
        choices=["audit", "train", "evaluate", "figures", "label-figures", "line-sky"],
    )
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--n", type=int, default=300, help="audit: windows per label")
    parser.add_argument(
        "--labels", nargs="+", default=["count", "band"], help="audit: labels"
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["DES 2018", "distance scan"],
        help="evaluate: which stream sets (DES 2018, distance scan, fainter)",
    )
    parser.add_argument(
        "--train-sky",
        choices=list(EVALUATED_ON),
        default="fold0",
        help="where models train (A: round 1, scored on B; fold0: scored on fold 1)",
    )
    arguments = parser.parse_args()
    if arguments.step == "audit":
        audit(arguments.n, tuple(arguments.labels))
    elif arguments.step == "train":
        train(arguments.config, arguments.seed, arguments.train_sky)
    elif arguments.step == "evaluate":
        # a line model, or a per-pixel ensemble searched along lines, answers
        # per window; the others per pixel
        lines = {"hough", "lines"} & set(CONFIGS[arguments.config])
        scorer = evaluate_hough if lines else evaluate
        scorer(
            arguments.config, train_sky=arguments.train_sky, sets=tuple(arguments.sets)
        )
    elif arguments.step == "label-figures":
        label_figures()
    elif arguments.step == "line-sky":  # the line model over the whole DES sky
        config = arguments.config or LINE_SKY_CONFIG
        line_sky(config)
        line_sky_matches(config)
        line_sky_chance(config)
        line_sky_figures(config)
        line_sky_summary(config)
        line_sky_tracks(config)
        line_sky_track_figure(config)
    else:
        figures(arguments.train_sky)
        if arguments.train_sky != "A":
            comparison_figures(arguments.train_sky)
            sensitivity(arguments.train_sky)
            detection_tests(arguments.train_sky)
            levers(arguments.train_sky)
            if (result_dir(arguments.train_sky) / "hough_hough_band.csv").exists():
                hough_figures(arguments.train_sky)
                hough_null_figure(arguments.train_sky)
                line_model_figures(arguments.train_sky)  # the guide's figures
