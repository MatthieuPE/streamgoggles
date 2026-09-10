"""Test datasets and transforms.

StreamMapDataset (real streamobs throughout, no mocking): training mode
(on-the-fly generation, background_fraction, optional persistence) and eval
mode (fixed eval_grid, always persisted, deterministic across accesses).

RobustNormalizer/StreamMapTransform (pure numpy, no streamobs needed):
valid_mask-aware normalization, and augmentation that keeps map_stack,
label_stack, and valid_mask in geometric lockstep. No noise injection --
considered and removed (map_stack is count data with its own realistic
survey noise already; synthetic Gaussian noise would teach a mismatched
noise model for a counting/denoising target).
"""

import numpy as np
import pytest

from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.config import DistributionType, EvalGrid, ParameterSpec, StreamConfig
from streamgoggles.datasets.stream_map_dataset import (
    StreamMapDataset,
    stream_map_collate_fn,
)
from streamgoggles.datasets.transforms import RobustNormalizer, StreamMapTransform
from streamgoggles.injector import StreamInjector, inject_background_only
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.storage import BackgroundMapStore, SimulationStore
from streamgoggles.stream_sources import StreamObsSource
from streamgoggles.windows import sample_random_window

pytestmark = pytest.mark.datasets

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_background(tmp_path_factory):
    region = StudyRegion(
        center_ra=0.0, center_dec=-30.0, width_deg=25.0, height_deg=18.0
    )
    pix = PixelizationSpec(nside=64, pixel_scale_deg=1.0, image_size_pix=(15, 15))
    store = BackgroundMapStore(tmp_path_factory.mktemp("bgmaps"))
    good = StreamobsSplineFilter(
        iso_config={"age": 12.5, "z": 0.0002}, namespace="lsst_yr1"
    )
    decoy = ShiftedColorBoxFilter(reference_filter=good, color_shift=0.5)
    filters = {"good": good, "decoy": decoy}
    bg = Background.load_or_cache(
        source=StreamObsLightBackgroundSource(),
        source_cfg={},
        study_region=region,
        cuts=[],
        clipping=None,
        matched_filters=filters,
        bands=["g", "r"],
        distance_moduli=[16.8],
        finalize_cfg=None,
        store=store,
        pix=pix,
        survey="lsst",
        release="yr1",
        filter_configs={"good": {"age": 12.5, "z": 0.0002}, "decoy": {"shift": 0.5}},
    )
    return bg, filters, pix


@pytest.fixture
def injector(real_background):
    bg, filters, pix = real_background
    return StreamInjector(
        background=bg,
        matched_filters=filters,
        stream_source=StreamObsSource(),
        cuts=[],
        clipping=None,
        pix=pix,
        survey="lsst",
        release="yr1",
    )


def _fixed_params_dict(**overrides):
    base = {
        "morphology": "uniform",
        "richness": 2500,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }
    base.update(overrides)
    return {
        name: ParameterSpec(name=name, dist_type=DistributionType.FIXED, value=value)
        for name, value in base.items()
    }


@pytest.fixture
def fixed_config():
    """All parameters fixed -- deterministic shape, only the RNG draw varies."""
    return StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=0.0,
        persist=False,
        richness_kind="nstars",
    )


@pytest.fixture
def free_richness_config():
    """richness is DISCRETE with 2 values -> a small, fast 2-point eval grid."""
    params = _fixed_params_dict()
    params["richness"] = ParameterSpec(
        name="richness", dist_type=DistributionType.DISCRETE, values=[2000, 4000]
    )
    return StreamConfig(
        params=params, background_fraction=0.0, persist=False, richness_kind="nstars"
    )


@pytest.fixture
def sim_store(tmp_path):
    return SimulationStore(tmp_path / "simulations")


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------


def test_training_mode_length_is_steps_per_epoch(
    real_background, injector, fixed_config, sim_store
):
    bg, _filters, _pix = real_background
    dataset = StreamMapDataset(
        config=fixed_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=7,
        rng=np.random.default_rng(0),
    )
    assert len(dataset) == 7


def test_training_mode_returns_valid_item(
    real_background, injector, fixed_config, sim_store
):
    bg, _filters, pix = real_background
    dataset = StreamMapDataset(
        config=fixed_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=5,
        rng=np.random.default_rng(0),
    )
    item = dataset[0]
    assert set(item) == {"map_stack", "label_stack", "valid_mask", "params", "metadata"}
    ny, nx = pix.image_size_pix
    assert item["map_stack"].shape == (2, ny, nx)  # 1 distance x 2 filters
    assert item["map_stack"].dtype == np.float32
    assert item["label_stack"].shape == (2, ny, nx)
    assert item["valid_mask"].shape == (ny, nx)
    assert item["valid_mask"].dtype == bool
    assert item["params"]["nstars"] == 2500


def test_training_mode_index_out_of_range_raises(
    real_background, injector, fixed_config, sim_store
):
    bg, _filters, _pix = real_background
    dataset = StreamMapDataset(
        config=fixed_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=3,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(IndexError):
        dataset[3]
    with pytest.raises(IndexError):
        dataset[-1]


def test_training_mode_background_fraction_one_is_always_background_only(
    real_background, injector, sim_store
):
    bg, _filters, _pix = real_background
    config = StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=1.0,
        persist=False,
        richness_kind="nstars",
    )
    dataset = StreamMapDataset(
        config=config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=5,
        rng=np.random.default_rng(0),
    )
    for i in range(5):
        item = dataset[i]
        assert item["params"] == {}
        assert np.all(item["label_stack"] == 0.0)


def test_stream_map_collate_fn_handles_mixed_background_and_stream_samples(
    real_background, injector, fixed_config
):
    """Found via real training with background_fraction > 0 through a real
    DataLoader: torch's default collate_fn crashes on a batch mixing a
    background-only item (params={}) with a stream item (params={richness,
    morphology, ...}) -- differing key sets. stream_map_collate_fn is the
    fix; this proves it actually handles the mix, using two real samples
    (not hand-built dicts)."""
    bg, _filters, pix = real_background
    stream_item = injector.inject_single_stream(
        {name: spec.value for name, spec in fixed_config.params.items()},
        np.random.default_rng(0),
    )
    size_deg = pix.image_size_pix[0] * pix.pixel_scale_deg
    window = sample_random_window(
        bg.footprint, pix.nside, size_deg=size_deg, rng=np.random.default_rng(1)
    )
    bg_item = inject_background_only(bg, window, pix)

    batch = [
        {
            "map_stack": stream_item.map_stack,
            "label_stack": stream_item.label_stack,
            "valid_mask": stream_item.valid_mask,
            "params": stream_item.params,
            "metadata": stream_item.metadata,
        },
        {
            "map_stack": bg_item.map_stack,
            "label_stack": bg_item.label_stack,
            "valid_mask": bg_item.valid_mask,
            "params": bg_item.params,
            "metadata": bg_item.metadata,
        },
    ]

    collated = stream_map_collate_fn(batch)

    assert collated["map_stack"].shape[0] == 2
    assert collated["label_stack"].shape[0] == 2
    assert collated["valid_mask"].shape[0] == 2
    assert collated["params"] == [stream_item.params, bg_item.params]
    assert collated["metadata"] == [stream_item.metadata, bg_item.metadata]


def test_default_collate_fails_on_the_same_mixed_batch_without_the_fix():
    """Documents the exact bug stream_map_collate_fn fixes: torch's own
    default_collate cannot batch dicts whose "params" key has different
    sub-keys across the batch."""
    from torch.utils.data import default_collate

    batch = [
        {"params": {"richness": 100, "morphology": "uniform"}},
        {"params": {}},
    ]
    with pytest.raises(KeyError):
        default_collate(batch)


def test_training_mode_background_fraction_zero_is_never_background_only(
    real_background, injector, sim_store
):
    bg, _filters, _pix = real_background
    config = StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=0.0,
        persist=False,
        richness_kind="nstars",
    )
    dataset = StreamMapDataset(
        config=config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=5,
        rng=np.random.default_rng(0),
    )
    for i in range(5):
        item = dataset[i]
        assert item["params"] != {}
        assert item["params"]["nstars"] == 2500


def test_training_mode_persists_stream_samples_when_configured(
    real_background, injector, sim_store
):
    bg, _filters, _pix = real_background
    config = StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=0.0,
        persist=True,
        richness_kind="nstars",
    )
    dataset = StreamMapDataset(
        config=config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=1,
        rng=np.random.default_rng(0),
    )
    item = dataset[0]
    assert sim_store.exists(item["params"])


def test_training_mode_does_not_persist_background_only_samples(
    real_background, injector, sim_store
):
    bg, _filters, _pix = real_background
    config = StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=1.0,
        persist=True,
        richness_kind="nstars",
    )
    dataset = StreamMapDataset(
        config=config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=3,
        rng=np.random.default_rng(0),
    )
    for i in range(3):
        dataset[i]
    manifest = sim_store._load_manifest()
    assert manifest.empty  # nothing with params={} ever gets a store key


def test_training_mode_does_not_persist_by_default(
    real_background, injector, fixed_config, sim_store
):
    bg, _filters, _pix = real_background
    assert fixed_config.persist is False
    dataset = StreamMapDataset(
        config=fixed_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=1,
        rng=np.random.default_rng(0),
    )
    item = dataset[0]
    assert not sim_store.exists(item["params"])


def test_training_mode_rejects_eval_grid(
    real_background, injector, fixed_config, sim_store
):
    bg, _filters, _pix = real_background
    with pytest.raises(ValueError, match="eval_mode"):
        StreamMapDataset(
            config=fixed_config,
            background=bg,
            injector=injector,
            store=sim_store,
            eval_mode=False,
            eval_grid=EvalGrid(points=[{}], seeds=[0]),
        )


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------


def test_eval_mode_auto_builds_grid_and_length_matches(
    real_background, injector, free_richness_config, sim_store
):
    bg, _filters, _pix = real_background
    dataset = StreamMapDataset(
        config=free_richness_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
    )
    assert dataset.eval_grid is not None
    assert (
        len(dataset) == len(dataset.eval_grid.points) == 2
    )  # 2 discrete richness values


def test_eval_mode_returns_valid_item(
    real_background, injector, free_richness_config, sim_store
):
    bg, _filters, pix = real_background
    dataset = StreamMapDataset(
        config=free_richness_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
    )
    item = dataset[0]
    ny, nx = pix.image_size_pix
    assert item["map_stack"].shape == (2, ny, nx)
    assert item["params"]["nstars"] in (2000, 4000)


def test_eval_mode_deterministic_across_accesses(
    real_background, injector, free_richness_config, sim_store
):
    bg, _filters, _pix = real_background
    dataset = StreamMapDataset(
        config=free_richness_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
    )
    item_a = dataset[0]
    item_b = dataset[0]
    np.testing.assert_array_equal(item_a["map_stack"], item_b["map_stack"])
    np.testing.assert_array_equal(item_a["label_stack"], item_b["label_stack"])
    assert item_a["params"] == item_b["params"]


def test_eval_mode_persists_regardless_of_config_persist(
    real_background, injector, free_richness_config, sim_store
):
    """The store is keyed on the *pre-resolution* full_params (config fixed
    values + the eval_grid point) -- checked before generation even runs, so
    a cache hit can skip it -- not on item["params"] (injector's *resolved*
    output: richness -> nstars, band_1/band_2 defaults added). Reconstruct
    the same key the dataset itself uses (config.params + eval_grid point,
    exactly as documented in StreamMapDataset._get_eval_sample) rather than
    the returned item's own params."""
    bg, _filters, _pix = real_background
    assert free_richness_config.persist is False
    dataset = StreamMapDataset(
        config=free_richness_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
    )
    dataset[0]
    full_params = {
        name: (spec.value if spec.is_fixed() else dataset.eval_grid.points[0][name])
        for name, spec in free_richness_config.params.items()
    }
    assert sim_store.exists(full_params)


def test_eval_mode_uses_explicitly_provided_grid(real_background, injector, sim_store):
    bg, _filters, pix = real_background
    config = StreamConfig(
        params=_fixed_params_dict(),
        background_fraction=0.0,
        persist=False,
        richness_kind="nstars",
    )
    grid = EvalGrid(points=[{}], seeds=[123])
    dataset = StreamMapDataset(
        config=config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
        eval_grid=grid,
    )
    assert dataset.eval_grid is grid
    assert len(dataset) == 1
    item = dataset[0]
    ny, nx = pix.image_size_pix
    assert item["map_stack"].shape == (2, ny, nx)


def test_eval_mode_index_out_of_range_raises(
    real_background, injector, free_richness_config, sim_store
):
    bg, _filters, _pix = real_background
    dataset = StreamMapDataset(
        config=free_richness_config,
        background=bg,
        injector=injector,
        store=sim_store,
        eval_mode=True,
    )
    with pytest.raises(IndexError):
        dataset[len(dataset)]


# ---------------------------------------------------------------------------
# RobustNormalizer
# ---------------------------------------------------------------------------


@pytest.fixture
def map_stack_with_invalid_pixel():
    rng = np.random.default_rng(0)
    map_stack = rng.uniform(0, 100, size=(3, 10, 10)).astype(np.float32)
    valid_mask = np.ones((10, 10), dtype=bool)
    valid_mask[0, 0] = False
    # A wild value only an invalid pixel has -- must never influence fit()
    # or get touched by __call__().
    map_stack[:, 0, 0] = 9999.0
    return map_stack, valid_mask


def test_robust_normalizer_fit_ignores_invalid_pixels(map_stack_with_invalid_pixel):
    map_stack, valid_mask = map_stack_with_invalid_pixel
    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    # 9999.0 would massively inflate mean/std if it leaked into the fit.
    assert np.all(norm.mean < 100)
    assert np.all(norm.std < 100)


def test_robust_normalizer_call_leaves_invalid_pixels_unchanged(
    map_stack_with_invalid_pixel,
):
    map_stack, valid_mask = map_stack_with_invalid_pixel
    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    out = norm(map_stack, valid_mask)
    np.testing.assert_array_equal(out[:, 0, 0], map_stack[:, 0, 0])


def test_robust_normalizer_call_normalizes_valid_pixels(map_stack_with_invalid_pixel):
    map_stack, valid_mask = map_stack_with_invalid_pixel
    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    out = norm(map_stack, valid_mask)
    for c in range(3):
        assert out[c][valid_mask].mean() == pytest.approx(0.0, abs=1e-4)
        assert out[c][valid_mask].std() == pytest.approx(1.0, rel=1e-4)


def test_robust_normalizer_call_before_fit_raises():
    norm = RobustNormalizer()
    map_stack = np.ones((2, 4, 4), dtype=np.float32)
    valid_mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(RuntimeError):
        norm(map_stack, valid_mask)


def test_robust_normalizer_per_channel_independent():
    map_stack = np.zeros((2, 4, 4), dtype=np.float32)
    map_stack[0] = 10.0  # channel 0: constant-ish, small scale
    map_stack[1] = 1000.0  # channel 1: large scale
    map_stack[0, 0, 0] = 12.0
    map_stack[1, 0, 0] = 1200.0
    valid_mask = np.ones((4, 4), dtype=bool)

    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    assert norm.mean[1] > norm.mean[0] * 10  # channels never mixed
    out = norm(map_stack, valid_mask)
    # Both channels end up on a comparable normalized scale despite the
    # 100x difference in raw magnitude.
    assert abs(out[0].std() - out[1].std()) < 0.5


def test_robust_normalizer_constant_channel_no_divide_by_zero():
    map_stack = np.full((1, 4, 4), 5.0, dtype=np.float32)
    valid_mask = np.ones((4, 4), dtype=bool)
    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    assert norm.std[0] == 1.0  # fallback, not 0
    out = norm(map_stack, valid_mask)
    assert np.all(np.isfinite(out))


def test_robust_normalizer_fit_requires_3d_map_stack():
    norm = RobustNormalizer()
    with pytest.raises(ValueError, match="3D"):
        norm.fit(np.ones((4, 4)), np.ones((4, 4), dtype=bool))


def test_robust_normalizer_fit_shape_mismatch_raises():
    norm = RobustNormalizer()
    with pytest.raises(ValueError, match="valid_mask shape"):
        norm.fit(np.ones((2, 4, 4)), np.ones((5, 5), dtype=bool))


def test_robust_normalizer_call_wrong_channel_count_raises():
    norm = RobustNormalizer()
    norm.fit(np.ones((2, 4, 4)), np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="channels"):
        norm(np.ones((3, 4, 4)), np.ones((4, 4), dtype=bool))


def test_robust_normalizer_fit_no_valid_pixels_raises():
    norm = RobustNormalizer()
    with pytest.raises(ValueError, match="valid pixels"):
        norm.fit(np.ones((1, 4, 4)), np.zeros((4, 4), dtype=bool))


def test_robust_normalizer_call_does_not_mutate_input(map_stack_with_invalid_pixel):
    map_stack, valid_mask = map_stack_with_invalid_pixel
    original = map_stack.copy()
    norm = RobustNormalizer()
    norm.fit(map_stack, valid_mask)
    norm(map_stack, valid_mask)
    np.testing.assert_array_equal(map_stack, original)


# ---------------------------------------------------------------------------
# StreamMapTransform
# ---------------------------------------------------------------------------


@pytest.fixture
def transform_sample():
    map_stack = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
    label_stack = map_stack.copy() * 10.0
    valid_mask = np.ones((6, 6), dtype=bool)
    valid_mask[0, 0] = False
    return {
        "map_stack": map_stack,
        "label_stack": label_stack,
        "valid_mask": valid_mask,
        "params": {"a": 1},
        "metadata": {"b": 2},
    }


def test_stream_map_transform_missing_key_raises_keyerror():
    transform = StreamMapTransform()
    with pytest.raises(KeyError):
        transform({"map_stack": np.zeros((1, 2, 2))})


def test_stream_map_transform_identity_when_disabled(transform_sample):
    transform = StreamMapTransform(normalizer=None, augment=False)
    out = transform(transform_sample)
    np.testing.assert_array_equal(out["map_stack"], transform_sample["map_stack"])
    np.testing.assert_array_equal(out["label_stack"], transform_sample["label_stack"])
    np.testing.assert_array_equal(out["valid_mask"], transform_sample["valid_mask"])


def test_stream_map_transform_does_not_mutate_input_sample(transform_sample):
    original_map = transform_sample["map_stack"].copy()
    transform = StreamMapTransform(augment=True, rng=np.random.default_rng(0))
    transform(transform_sample)
    np.testing.assert_array_equal(transform_sample["map_stack"], original_map)


def test_stream_map_transform_passes_through_extra_keys(transform_sample):
    transform = StreamMapTransform()
    out = transform(transform_sample)
    assert out["params"] == {"a": 1}
    assert out["metadata"] == {"b": 2}


def test_stream_map_transform_output_dtypes(transform_sample):
    transform = StreamMapTransform(augment=True, rng=np.random.default_rng(0))
    out = transform(transform_sample)
    assert out["map_stack"].dtype == np.float32
    assert out["label_stack"].dtype == np.float32
    assert out["valid_mask"].dtype == bool


def test_stream_map_transform_applies_normalizer(transform_sample):
    norm = RobustNormalizer()
    norm.fit(transform_sample["map_stack"], transform_sample["valid_mask"])
    transform = StreamMapTransform(normalizer=norm, augment=False)
    out = transform(transform_sample)
    assert not np.array_equal(out["map_stack"], transform_sample["map_stack"])
    # label_stack is never normalized -- stays in its raw count unit.
    np.testing.assert_array_equal(out["label_stack"], transform_sample["label_stack"])


def test_stream_map_transform_augmentation_keeps_map_label_mask_in_lockstep(
    transform_sample,
):
    """A marker at one pixel in map_stack/label_stack, and the ONLY invalid
    pixel in valid_mask, must all land at the same new location after any
    random flip/rotation -- proving the three arrays are transformed
    identically, not independently."""
    map_stack = transform_sample["map_stack"].copy()
    label_stack = transform_sample["label_stack"].copy()
    valid_mask = np.ones((6, 6), dtype=bool)
    marker_pos = (2, 3)
    map_stack[:] = 0.0
    map_stack[0, marker_pos[0], marker_pos[1]] = 1.0
    label_stack[:] = 0.0
    label_stack[0, marker_pos[0], marker_pos[1]] = 1.0
    valid_mask[marker_pos] = False

    sample = {
        "map_stack": map_stack,
        "label_stack": label_stack,
        "valid_mask": valid_mask,
    }

    for seed in range(20):
        transform = StreamMapTransform(augment=True, rng=np.random.default_rng(seed))
        out = transform(sample)
        map_pos = tuple(np.argwhere(out["map_stack"][0] == 1.0)[0])
        label_pos = tuple(np.argwhere(out["label_stack"][0] == 1.0)[0])
        mask_pos = tuple(np.argwhere(~out["valid_mask"])[0])
        assert map_pos == label_pos == mask_pos


def test_stream_map_transform_augmentation_output_is_contiguous(transform_sample):
    transform = StreamMapTransform(augment=True, rng=np.random.default_rng(2))
    out = transform(transform_sample)
    assert out["map_stack"].flags["C_CONTIGUOUS"]
    assert out["label_stack"].flags["C_CONTIGUOUS"]
    assert out["valid_mask"].flags["C_CONTIGUOUS"]
