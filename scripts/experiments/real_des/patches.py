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


def build_sky(sky, label):
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
    pix = PixelizationSpec(nside=512, image_size_pix=(IMAGE_PIX, IMAGE_PIX))
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


def train(config, seed, train_sky="A"):
    import torch

    rd = real_des()
    sp = rd.stream_parameters_module()
    stem = model_stem(config, seed, train_sky)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.with_suffix(".pt").exists():
        print(f"{stem.name}: already trained", flush=True)
        return
    background, injector, _ = build_sky(train_sky, CONFIGS[config]["label"])
    start = time.time()
    model, _, result = sp.train(
        seed,
        WINDOWS,
        background,
        injector,
        training_set=rd.TRAINING_SET,
        normalizer=normalizer(CONFIGS[config]["normalizer"], channels()),
    )
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": config,
                "seed": seed,
                "windows": WINDOWS,
                "train_s": time.time() - start,
                "train_losses": result["train_losses"],
                "val_losses": result["val_losses"],
            }
        )
    )
    print(f"{stem.name}: trained in {time.time() - start:.0f}s", flush=True)


def audit(n=300):
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
    for label in ("count", "band"):
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
    with open(OUT / "audit.pkl", "wb") as handle:
        pickle.dump(rows, handle)


def evaluation_streams():
    """(set, name, params) injected on patch B."""
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
    return streams


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


def evaluate(config, seeds=SEEDS, train_sky="A"):
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
    from streamgoggles.evaluation.footprint import real_track_statistics, track_band
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
    background, injector, pix = build_sky(eval_sky, "count")
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
    for seed in seeds:
        model = UNet(
            in_channels=QueryDistanceTransform.n_channels, out_channels=1, **sp.MODEL
        )
        model.load_state_dict(
            torch.load(model_stem(config, seed, train_sky).with_suffix(".pt"))
        )
        model.eval()
        models[f"seed{seed}"] = model
    streams = evaluation_streams()
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
            for key, image in stitched.items():
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
    pd.DataFrame(rows).to_csv(out / f"evaluation_{name}.csv", index=False)
    pd.DataFrame(quiet).to_csv(out / f"quiet_{name}.csv", index=False)


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
    ax.set_xticklabels([c.replace("/", "\n") for c in order], fontsize=9)
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
    ax.legend(frameon=False, fontsize=9)
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
            ax.set_xticklabels([c.replace("/", "\n") for c in order], fontsize=9)
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
        "--train-sky",
        choices=list(EVALUATED_ON),
        default="fold0",
        help="where models train (A: round 1, scored on B; fold0: scored on fold 1)",
    )
    arguments = parser.parse_args()
    if arguments.step == "audit":
        audit(arguments.n)
    elif arguments.step == "train":
        train(arguments.config, arguments.seed, arguments.train_sky)
    elif arguments.step == "evaluate":
        evaluate(arguments.config, train_sky=arguments.train_sky)
    elif arguments.step == "label-figures":
        label_figures()
    else:
        figures(arguments.train_sky)
