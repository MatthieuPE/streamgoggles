"""Test training/plain_runner.py.

PlainTrainer (real torch throughout, no mocking): train_epoch/validate on a
small synthetic in-memory dataset (fast, isolated -- proves the loop's own
mechanics: masked-loss wiring, optimizer stepping, AMP on CPU, metrics_fn
weighting, checkpointing via a real ModelStore), plus one full end-to-end
integration test wiring a real StreamMapDataset (real streamobs background
+ injector, same fixture pattern as test_datasets.py) through a real UNet
and a real DataLoader.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from streamgoggles.background import Background
from streamgoggles.background_sources import StreamObsLightBackgroundSource, StudyRegion
from streamgoggles.config import DistributionType, ParameterSpec, StreamConfig
from streamgoggles.datasets.stream_map_dataset import StreamMapDataset
from streamgoggles.injector import StreamInjector
from streamgoggles.matched_filter import (
    PixelizationSpec,
    ShiftedColorBoxFilter,
    StreamobsSplineFilter,
)
from streamgoggles.models.losses import MSELoss, get_loss
from streamgoggles.models.unet import UNet
from streamgoggles.storage import BackgroundMapStore, ModelStore, SimulationStore
from streamgoggles.stream_sources import StreamObsSource
from streamgoggles.training.plain_runner import PlainTrainer

pytestmark = pytest.mark.training

IN_CHANNELS = 2
SIZE = 6


class TinyMapDataset:
    """Small deterministic in-memory dataset: fixed map_stack/label_stack
    per index, so a trainer can be made to visibly overfit in a handful of
    epochs without any streamobs/healpy dependency."""

    def __init__(self, n: int = 8, seed: int = 0, target_value: float = 1.0):
        rng = np.random.default_rng(seed)
        self._items = []
        for _ in range(n):
            map_stack = rng.normal(size=(IN_CHANNELS, SIZE, SIZE)).astype(np.float32)
            label_stack = np.full(
                (IN_CHANNELS, SIZE, SIZE), target_value, dtype=np.float32
            )
            valid_mask = np.ones((SIZE, SIZE), dtype=bool)
            self._items.append(
                {
                    "map_stack": map_stack,
                    "label_stack": label_stack,
                    "valid_mask": valid_mask,
                    "params": {},
                    "metadata": {},
                }
            )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def _make_trainer(**kwargs):
    model = UNet(
        in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, base_width=4, depth=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()
    return PlainTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, **kwargs)


@pytest.fixture
def tiny_dl():
    return DataLoader(TinyMapDataset(n=8), batch_size=4)


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


def test_train_epoch_returns_expected_keys(tiny_dl):
    trainer = _make_trainer()
    result = trainer.train_epoch(tiny_dl)
    assert set(result) == {"loss", "n_batches", "lr"}
    assert result["n_batches"] == 2
    assert result["lr"] == pytest.approx(0.05)
    assert np.isfinite(result["loss"])


def test_train_epoch_updates_model_weights(tiny_dl):
    trainer = _make_trainer()
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.train_epoch(tiny_dl)
    after = trainer.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_train_epoch_loss_decreases_when_overfitting_fixed_target():
    dl = DataLoader(TinyMapDataset(n=8, target_value=2.0), batch_size=8)
    trainer = _make_trainer()
    first_loss = trainer.train_epoch(dl)["loss"]
    for _ in range(20):
        last_loss = trainer.train_epoch(dl)["loss"]
    assert last_loss < first_loss


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_does_not_change_weights(tiny_dl):
    trainer = _make_trainer()
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.validate(tiny_dl)
    after = trainer.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_validate_returns_loss_only_without_metrics_fn(tiny_dl):
    trainer = _make_trainer()
    result = trainer.validate(tiny_dl)
    assert set(result) == {"loss"}


def test_validate_metrics_fn_is_batch_size_weighted():
    # Two batches of different sizes; metrics_fn returns a constant that
    # differs per call via a mutable counter, so a naive unweighted mean
    # over batches (rather than over samples) would give a different
    # answer than the batch-size-weighted one this method promises.
    dl = DataLoader(TinyMapDataset(n=6), batch_size=4, drop_last=False)
    trainer = _make_trainer()

    calls = []

    def metrics_fn(pred, target, valid_mask):
        calls.append(pred.shape[0])
        return {"dummy": torch.tensor(float(pred.shape[0]))}

    result = trainer.validate(dl, metrics_fn=metrics_fn)
    # batch sizes are [4, 2]; weighted mean of value==batch_size is
    # (4*4 + 2*2) / 6 = 20/6
    assert result["dummy"] == pytest.approx(20 / 6)
    assert calls == [4, 2]


# ---------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------


def test_amp_true_on_cpu_runs_without_crashing(tiny_dl):
    trainer = _make_trainer(amp=True)
    assert trainer._device_type == "cpu"
    assert not trainer.scaler.is_enabled()
    result = trainer.train_epoch(tiny_dl)
    assert np.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# train() full loop
# ---------------------------------------------------------------------------


def test_train_without_val_dl_val_losses_is_none(tiny_dl):
    trainer = _make_trainer()
    result = trainer.train(tiny_dl, val_dl=None, epochs=2)
    assert result["val_losses"] is None
    assert len(result["train_losses"]) == 2
    assert result["epochs_run"] == 2
    assert result["final_checkpoint_path"] is None


def test_train_with_val_dl_tracks_both_loss_lists(tiny_dl):
    trainer = _make_trainer()
    result = trainer.train(tiny_dl, val_dl=tiny_dl, epochs=3)
    assert len(result["train_losses"]) == 3
    assert len(result["val_losses"]) == 3


def test_train_callbacks_invoked_each_epoch_with_metrics(tiny_dl):
    trainer = _make_trainer()
    seen = []
    callback = lambda epoch, trainer_, metrics: seen.append((epoch, metrics))
    trainer.train(tiny_dl, val_dl=tiny_dl, epochs=2, callbacks=[callback])
    assert [epoch for epoch, _ in seen] == [0, 1]
    assert all("train" in metrics and "val" in metrics for _, metrics in seen)


def test_train_missing_model_config_raises():
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="model_config"):
        trainer.train(
            DataLoader(TinyMapDataset(n=2), batch_size=2),
            model_store=object(),
            config={"label_policy": "stream_count"},
            epochs=1,
        )


def test_train_none_config_with_model_store_raises():
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="model_config"):
        trainer.train(
            DataLoader(TinyMapDataset(n=2), batch_size=2),
            model_store=object(),
            config=None,
            epochs=1,
        )


# ---------------------------------------------------------------------------
# Checkpointing with a real ModelStore
# ---------------------------------------------------------------------------


def test_train_saves_checkpoint_and_reloads_identically(tmp_path, tiny_dl):
    trainer = _make_trainer()
    model_store = ModelStore(tmp_path / "models")
    model_config = {
        "in_channels": IN_CHANNELS,
        "out_channels": IN_CHANNELS,
        "base_width": 4,
        "depth": 1,
    }
    config = {"model_config": model_config, "label_policy": "stream_count"}

    result = trainer.train(tiny_dl, epochs=2, model_store=model_store, config=config)

    assert result["final_checkpoint_path"] is not None
    assert result["final_checkpoint_path"].exists()

    loaded = model_store.load_model(UNet, config)
    trainer.model.eval()
    loaded.eval()
    x = torch.randn(1, IN_CHANNELS, SIZE, SIZE)
    with torch.no_grad():
        expected = trainer.model(x)
        actual = loaded(x)
    torch.testing.assert_close(actual, expected)


def test_train_save_every_only_saves_on_multiples_and_final_epoch(tmp_path, tiny_dl):
    trainer = _make_trainer()
    model_store = ModelStore(tmp_path / "models")
    config = {
        "model_config": {
            "in_channels": IN_CHANNELS,
            "out_channels": IN_CHANNELS,
            "base_width": 4,
            "depth": 1,
        }
    }
    save_calls = []
    original_save_model = model_store.save_model

    def spy_save_model(*args, **kwargs):
        save_calls.append(kwargs.get("metrics", {}).get("epoch"))
        return original_save_model(*args, **kwargs)

    model_store.save_model = spy_save_model
    trainer.train(
        tiny_dl, epochs=5, model_store=model_store, config=config, save_every=3
    )

    # save_every=3 over 5 epochs -> saves at epoch 3, plus the final epoch 5
    # (which isn't itself a multiple of 3) since the last epoch always saves.
    assert save_calls == [3, 5]


# ---------------------------------------------------------------------------
# End-to-end: real StreamMapDataset + real UNet
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
def real_injector(real_background):
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


@pytest.fixture
def real_dataset(real_background, real_injector, tmp_path):
    bg, _filters, _pix = real_background
    params = {
        "morphology": "uniform",
        "richness": 2500,
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 16.8,
        "age": 12.5,
        "z": 0.0002,
    }
    config = StreamConfig(
        params={
            name: ParameterSpec(
                name=name, dist_type=DistributionType.FIXED, value=value
            )
            for name, value in params.items()
        },
        background_fraction=0.0,
        persist=False,
        richness_kind="nstars",
    )
    sim_store = SimulationStore(tmp_path / "sims")
    return StreamMapDataset(
        config=config,
        background=bg,
        injector=real_injector,
        store=sim_store,
        eval_mode=False,
        steps_per_epoch=4,
    )


def test_end_to_end_real_dataset_trains_one_epoch(real_dataset):
    dl = DataLoader(real_dataset, batch_size=2)
    n_channels = 2  # 1 distance_modulus x 2 filters (good + decoy)
    model = UNet(
        in_channels=n_channels,
        out_channels=n_channels,
        base_width=4,
        depth=1,
        head="softplus",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = PlainTrainer(
        model=model, optimizer=optimizer, loss_fn=get_loss("weighted_mse")
    )

    result = trainer.train(dl, epochs=1)

    assert result["epochs_run"] == 1
    assert np.isfinite(result["train_losses"][0])
