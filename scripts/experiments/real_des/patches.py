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
    "hough/band": {
        "label": "band",
        "normalizer": "window",
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
}
EVALUATED_ON = {"A": "B", "fold0": "fold1"}
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
    source_cfg = {"path": str(rd.TRAINING_CATALOGUE), "exclude": str(contaminants())}
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


def smooth_backgrounds(background, queries):
    """{query: the stream-free matched-filter counts smoothed to ~2 degrees}:
    the valid-weighted mean over nside-32 pixels (1.8 deg), interpolated back
    to every pixel. The coarse sums and the coarse valid fraction are
    interpolated apart and divided, so sky outside the footprint weighs
    nothing (not hp.smoothing, whose OpenMP runtime clashes with torch's)."""
    import healpy as hp
    import numpy as np

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


def evaluate_hough(
    config, seeds=SEEDS, train_sky="fold0", sets=("DES 2018", "distance scan")
):
    """Window-level scoring of a line model, and of the matched filter's own
    line sums (a classical line search), on the same copies as `evaluate`.

    Each window answers with one score per line through it: the model's
    probability (the mean over its seeds), or, for the matched filter, the
    counts at the queried distance minus the smooth local background, over
    the background's square root, summed along the line (`HoughLines`). The
    lines "along the track" of a copy, in a window, are those `hough_target`
    makes of the copy's band there; a window scores a copy if it holds at
    least HOUGH_MIN_LENGTH_DEG of it.

    Two tests, against HOUGH_NULL_WINDOWS stream-free windows centred on
    random points of the calibration sky, at the same queried distance:

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
    from streamgoggles.evaluation.footprint import track_band
    from streamgoggles.matched_filter import (
        WindowProjection,
        stitch_windows_to_healpix,
        window_to_healpix_indices,
    )
    from streamgoggles.models.hough import hough_target, line_counts
    from streamgoggles.windows import Window, tile_footprint

    configure_torch_threads(num_workers=0)
    rd = real_des()
    sp = rd.stream_parameters_module()
    eval_sky = EVALUATED_ON[train_sky]
    image_pix = CONFIGS[config].get("image_pix", IMAGE_PIX)
    background, injector, pix = build_sky(eval_sky, "count", image_pix)
    nside = pix.nside
    valid = background.valid_mask_full
    window_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    tiles = tile_footprint(
        valid, nside, tile_size_deg=window_deg, stride_deg=window_deg / 2
    )
    tile_pixels = [window_to_healpix_indices(t, pix, nside)[0] for t in tiles]
    projections = [WindowProjection.for_window(t, pix, valid) for t in tiles]
    chans = channels()
    norm = normalizer(CONFIGS[config]["normalizer"], chans)
    factory, _ = hough_parts(config)
    models = []
    for seed in CONFIGS[config].get("seeds", seeds):
        model = factory(QueryDistanceTransform.n_channels)
        model.load_state_dict(
            torch.load(model_stem(config, seed, train_sky).with_suffix(".pt"))
        )
        models.append(model.eval())
    grid = models[0].hough.grid
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
    # The calibration sky exactly as `evaluate` has it -- valid, calibration
    # mask, and seen by the window most central to it -- so the copies land
    # on the same places and the two evaluations compare copy by copy.
    calibration = valid.copy()
    if "fold" in SKIES[eval_sky]:
        calibration &= hp.read_map(rd.CALIBRATION_MASK).astype(bool)
    seen = stitch_windows_to_healpix(
        [np.where(p.valid, 1.0, np.nan) for p in projections], tiles, pix, nside
    )[0]
    calibration &= valid & np.isfinite(seen)
    smooth_background = smooth_backgrounds(background, queries)

    def scores(maps_full, windows, query):
        """Line scores of the model and of the matched filter, (n, n_theta,
        n_rho) each, NaN on lines too short to count."""
        inputs, matched = [], []
        for projection in windows:
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
                network.append(
                    np.mean([torch.sigmoid(m(x))[:, 0].numpy() for m in models], 0)
                )
        network = np.concatenate(network).astype(np.float32)
        matched = np.stack(matched).astype(np.float32)
        network[:, ~grid.valid] = np.nan
        matched[:, ~grid.valid] = np.nan
        return {"network": network, "matched filter": matched}

    # Stream-free windows: centred on random calibration pixels, at least
    # half valid, unrotated like the tiles.
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
        if projection.valid.mean() >= 0.5:
            null_windows.append(projection)
    background_maps = [
        background.raw_map_full_dict[c["filter"]][c["distance_modulus"]] for c in chans
    ]
    null, thresholds, null_rows = {}, {}, []
    start = time.time()
    for q in queries:
        null[q] = scores(background_maps, null_windows, q)
        for scorer, values in null[q].items():
            best = np.nanmax(values.reshape(len(values), -1), axis=1)
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
        rng = np.random.default_rng([EVAL_SEED, index])
        query = min(sp.QUERY_GRID, key=lambda q: abs(q - stream["distance_modulus"]))
        good = good_at[query]
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
            stream_stars = sky["stream_raw_full"][good][band].sum()
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
                answers = scores(sky["map_full"], scored, query)
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
            placed += 1
        print(
            f"{config} {name}: {placed} placements, {time.time() - start:.0f}s",
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


HOUGH_FIGURES = {
    # without the track: what a search can claim
    "blind": [
        ("per pixel", "count/window x4", "detected", "per-pixel network, 4 models"),
        ("lines", None, "network blind", "line network, 1% false lines per window"),
        ("lines", None, "matched filter blind", "matched-filter lines, same rate"),
    ],
    # along the known track: the most the data allow
    "known": [
        ("lines", None, "network known", "line network, known track"),
        ("lines", None, "matched filter known", "matched-filter lines, known track"),
        (
            "per pixel",
            "count/window x4",
            "matched_filter_detected",
            "matched-filter band",
        ),
    ],
}
HOUGH_STYLES = ["#4d4d4d", "#1f6fb4", "#e07b39"]


def hough_figures(train_sky="fold0", config="hough/band"):
    """hough_{blind,known}_{sky}.png and hough.csv: the line model against the
    per-pixel network and the matched filter on the same copies (DES 2018 at
    full and reduced brightness), found against input S/N, near and far."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    folder = result_dir(train_sky)

    def read(prefix, name, ensemble):
        parts = [
            folder / f"{prefix}_{name}.csv",
            folder / f"{prefix}_{name}__fainter.csv",
        ]
        data = pd.concat(
            [pd.read_csv(p) for p in parts if p.exists()], ignore_index=True
        )
        if ensemble:
            data = data[data.scorer == "ensemble"]
        return data[data.set.isin(["DES 2018", "fainter"])]

    lines = read("hough", config.replace("/", "_"), ensemble=False)
    rows = []
    for figure, curves in HOUGH_FIGURES.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
        for ax, (label, near) in zip(
            axes, (("m−M < 16.5", True), ("m−M ≥ 16.5", False)), strict=True
        ):
            for (source, other, test, legend), colour in zip(
                curves, HOUGH_STYLES, strict=True
            ):
                data = (
                    lines
                    if source == "lines"
                    else read("evaluation", other.replace("/", "_"), ensemble=True)
                )
                mine = data[(data.distance_modulus < 16.5) == near].assign(
                    detected=lambda d, t=test: d[t].astype(float)
                )
                rate = mine.groupby(
                    pd.cut(mine.input_snr, SNR_EDGES), observed=True
                ).detected.agg(["mean", "size"])
                rate = rate[rate["size"] >= 4]
                centres = [np.sqrt(max(b.left, 1) * b.right) for b in rate.index]
                style = "--" if "matched" in test and figure == "known" else "-"
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
            "without the track" if figure == "blind" else "along the known track",
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
        "step", choices=["audit", "train", "evaluate", "figures", "label-figures"]
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
        # a line model answers per window, the others per pixel
        scorer = evaluate_hough if "hough" in CONFIGS[arguments.config] else evaluate
        scorer(
            arguments.config, train_sky=arguments.train_sky, sets=tuple(arguments.sets)
        )
    elif arguments.step == "label-figures":
        label_figures()
    else:
        figures(arguments.train_sky)
        if arguments.train_sky != "A":
            comparison_figures(arguments.train_sky)
            sensitivity(arguments.train_sky)
            detection_tests(arguments.train_sky)
            levers(arguments.train_sky)
            if (result_dir(arguments.train_sky) / "hough_hough_band.csv").exists():
                hough_figures(arguments.train_sky)
