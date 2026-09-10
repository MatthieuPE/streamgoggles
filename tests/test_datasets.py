"""Test datasets and transforms.

StreamMapDataset (real streamobs throughout, no mocking): training mode
(on-the-fly generation, background_fraction, optional persistence) and eval
mode (fixed eval_grid, always persisted, deterministic across accesses).
transforms.py isn't implemented yet -- no tests for it here until it is.
"""

import numpy as np
import pytest

from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.config import DistributionType, EvalGrid, ParameterSpec, StreamConfig
from streamgoggles.datasets.stream_map_dataset import StreamMapDataset
from streamgoggles.injector import StreamInjector
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.storage import BackgroundMapStore, SimulationStore
from streamgoggles.stream_sources import StreamObsSource

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
