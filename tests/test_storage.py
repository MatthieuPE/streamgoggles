"""Test parameter-addressed storage: hashing, manifest queries, and the three stores."""

import numpy as np
import pytest

from streamgoggles.storage import (
    BackgroundMapStore,
    ModelStore,
    ParameterStore,
    SimulationStore,
)

pytestmark = pytest.mark.storage

# ---------------------------------------------------------------------------
# ParameterStore: key / path / exists / generic save-load / manifest query
# ---------------------------------------------------------------------------


def test_key_deterministic(tmp_path):
    store = ParameterStore(tmp_path)
    params = {"width": 0.2, "age": 12.0}
    assert store.key(params) == store.key(dict(params))


def test_key_order_independent(tmp_path):
    store = ParameterStore(tmp_path)
    a = {"width": 0.2, "age": 12.0, "nested": {"x": 1, "y": 2}}
    b = {"age": 12.0, "nested": {"y": 2, "x": 1}, "width": 0.2}
    assert store.key(a) == store.key(b)


def test_key_differs_for_different_params(tmp_path):
    store = ParameterStore(tmp_path)
    assert store.key({"width": 0.2}) != store.key({"width": 0.3})


def test_path_is_sharded_under_root(tmp_path):
    store = ParameterStore(tmp_path)
    params = {"width": 0.2}
    path = store.path(params)
    key = store.key(params)
    assert path.parent.name == key[:4]
    assert path.name == key[4:]
    assert store.root in path.parents


def test_exists_false_then_true_after_save(tmp_path):
    store = ParameterStore(tmp_path)
    params = {"width": 0.2}
    assert not store.exists(params)
    store.save({"a": 1}, params)
    assert store.exists(params)


def test_save_load_roundtrip_generic_object(tmp_path):
    store = ParameterStore(tmp_path)
    params = {"width": 0.2}
    obj = {"array": np.arange(5), "note": "hello"}
    store.save(obj, params, metadata={"source": "test"})
    loaded = store.load(params)
    np.testing.assert_array_equal(loaded["array"], obj["array"])
    assert loaded["note"] == "hello"


def test_load_missing_raises_file_not_found(tmp_path):
    store = ParameterStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load({"width": 0.2})


def test_manifest_upsert_same_params_single_row(tmp_path):
    store = ParameterStore(tmp_path)
    params = {"width": 0.2}
    store.save({"v": 1}, params)
    store.save({"v": 2}, params)  # re-save same params -> overwrite, not duplicate
    manifest = store._load_manifest()
    assert len(manifest) == 1
    assert store.load(params)["v"] == 2


def test_manifest_upsert_survives_many_resaves(tmp_path):
    """Repeated re-saves under the same params must never accumulate rows —
    this is the specific "does the parquet manifest stack duplicates instead
    of overwriting" failure mode."""
    store = ParameterStore(tmp_path)
    params = {"width": 0.2}
    for i in range(5):
        store.save({"v": i}, params)
    manifest = store._load_manifest()
    assert len(manifest) == 1
    assert store.load(params)["v"] == 4  # latest save wins, not a stale earlier one


def test_manifest_upsert_preserves_unrelated_entries(tmp_path):
    """Re-saving one params dict must not disturb other entries' manifest rows
    or on-disk content."""
    store = ParameterStore(tmp_path)
    params_a = {"width": 0.2}
    params_b = {"width": 0.3}

    store.save({"v": "a1"}, params_a)
    store.save({"v": "b1"}, params_b)
    store.save({"v": "a2"}, params_a)  # re-save A only

    manifest = store._load_manifest()
    assert len(manifest) == 2
    assert store.load(params_a)["v"] == "a2"
    assert store.load(params_b)["v"] == "b1"  # untouched by A's resave


def test_query_equality(tmp_path):
    store = ParameterStore(tmp_path)
    store.save({}, {"width": 0.2, "age": 12.0})
    store.save({}, {"width": 0.3, "age": 12.0})
    result = store.query(width=0.2)
    assert len(result) == 1
    assert result.iloc[0]["width"] == 0.2


def test_query_range_tuple(tmp_path):
    store = ParameterStore(tmp_path)
    for width in (0.1, 0.2, 0.3, 0.5):
        store.save({}, {"width": width})
    result = store.query(width=(0.15, 0.35))
    assert sorted(result["width"]) == [0.2, 0.3]


def test_query_isin_list(tmp_path):
    store = ParameterStore(tmp_path)
    for age in (10.0, 12.0, 13.5):
        store.save({}, {"age": age})
    result = store.query(age=[10.0, 13.5])
    assert sorted(result["age"]) == [10.0, 13.5]


def test_query_unknown_column_raises_keyerror(tmp_path):
    store = ParameterStore(tmp_path)
    store.save({}, {"width": 0.2})
    with pytest.raises(KeyError):
        store.query(does_not_exist=1)


def test_query_empty_manifest_returns_empty_dataframe(tmp_path):
    store = ParameterStore(tmp_path)
    result = store.query(width=0.2)
    assert result.empty


# ---------------------------------------------------------------------------
# SimulationStore
# ---------------------------------------------------------------------------


def test_simulation_store_save_load_roundtrip(tmp_path, tiny_sample):
    store = SimulationStore(tmp_path / "simulations")
    params = {"width": 0.2, "age": 12.0}
    store.save_sample(tiny_sample, params)

    loaded = store.load_sample(params)
    np.testing.assert_array_equal(loaded.map_stack, tiny_sample.map_stack)
    np.testing.assert_array_equal(loaded.label_stack, tiny_sample.label_stack)
    np.testing.assert_array_equal(loaded.valid_mask, tiny_sample.valid_mask)
    assert loaded.params == tiny_sample.params
    assert loaded.metadata == tiny_sample.metadata


def test_simulation_store_load_missing_raises(tmp_path):
    store = SimulationStore(tmp_path / "simulations")
    with pytest.raises(FileNotFoundError):
        store.load_sample({"width": 0.2})


def test_simulation_store_resave_overwrites_not_stacks(tmp_path, tiny_sample):
    """Re-saving a sample under the same params must replace the .npz content
    and the manifest row, not leave stale data alongside the new save."""
    from streamgoggles.sample import Sample

    store = SimulationStore(tmp_path / "simulations")
    params = {"width": 0.2, "age": 12.0}

    store.save_sample(tiny_sample, params)

    second_sample = Sample(
        map_stack=tiny_sample.map_stack + 100.0,
        label_stack=tiny_sample.label_stack,
        valid_mask=tiny_sample.valid_mask,
        params={"width": 0.2, "age": 12.0, "extra": "changed"},
        metadata={"seed": 999},
    )
    store.save_sample(second_sample, params)

    loaded = store.load_sample(params)
    np.testing.assert_array_equal(loaded.map_stack, second_sample.map_stack)
    assert loaded.params == second_sample.params
    assert loaded.metadata == second_sample.metadata
    assert len(store._load_manifest()) == 1


def test_simulation_store_get_or_generate_no_persist_always_generates(
    tmp_path, tiny_sample
):
    store = SimulationStore(tmp_path / "simulations")
    calls = []

    def generator(params):
        calls.append(params)
        return tiny_sample

    params = {"width": 0.2}
    store.get_or_generate(params, generator, persist=False)
    store.get_or_generate(params, generator, persist=False)

    assert len(calls) == 2
    assert not store.exists(params)


def test_simulation_store_get_or_generate_persist_caches(tmp_path, tiny_sample):
    store = SimulationStore(tmp_path / "simulations")
    calls = []

    def generator(params):
        calls.append(params)
        return tiny_sample

    params = {"width": 0.2}
    first = store.get_or_generate(params, generator, persist=True)
    second = store.get_or_generate(params, generator, persist=True)

    assert len(calls) == 1  # second call was served from cache
    np.testing.assert_array_equal(first.map_stack, second.map_stack)


# ---------------------------------------------------------------------------
# BackgroundMapStore
# ---------------------------------------------------------------------------


def test_background_map_store_roundtrip_with_finalized(tmp_path):
    store = BackgroundMapStore(tmp_path / "background_maps")
    raw_map = np.arange(12, dtype=np.float64)
    valid_mask = np.ones(12, dtype=bool)
    finalized_map = raw_map * 2
    params = {"survey": "lsst", "release": "yr1", "distance_modulus": 17.5}

    store.save_background(raw_map, valid_mask, finalized_map, params)
    loaded_raw, loaded_valid, loaded_finalized = store.load_background(params)

    np.testing.assert_array_equal(loaded_raw, raw_map)
    np.testing.assert_array_equal(loaded_valid, valid_mask)
    np.testing.assert_array_equal(loaded_finalized, finalized_map)


def test_background_map_store_roundtrip_without_finalized(tmp_path):
    store = BackgroundMapStore(tmp_path / "background_maps")
    raw_map = np.arange(12, dtype=np.float64)
    valid_mask = np.ones(12, dtype=bool)
    params = {"survey": "lsst", "release": "dp2", "distance_modulus": 17.5}

    store.save_background(raw_map, valid_mask, None, params)
    _, _, loaded_finalized = store.load_background(params)

    assert loaded_finalized is None


def test_background_map_store_resave_overwrites_values(tmp_path):
    store = BackgroundMapStore(tmp_path / "background_maps")
    params = {"survey": "lsst", "release": "yr1", "distance_modulus": 17.5}

    store.save_background(np.zeros(12), np.ones(12, dtype=bool), None, params)
    store.save_background(np.ones(12) * 7, np.ones(12, dtype=bool), None, params)

    raw_map, _, _ = store.load_background(params)
    np.testing.assert_array_equal(raw_map, np.ones(12) * 7)
    assert len(store._load_manifest()) == 1


def test_background_map_store_resave_drops_stale_finalized_map(tmp_path):
    """The .npz file is fully rewritten on save, not appended to — resaving
    WITHOUT a finalized_map must not leave a stale one from an earlier save
    lying around in the reloaded object."""
    store = BackgroundMapStore(tmp_path / "background_maps")
    params = {"survey": "lsst", "release": "yr1", "distance_modulus": 17.5}
    raw_map = np.arange(12, dtype=np.float64)
    valid_mask = np.ones(12, dtype=bool)

    store.save_background(raw_map, valid_mask, raw_map * 2, params)
    _, _, finalized_first = store.load_background(params)
    assert finalized_first is not None

    store.save_background(raw_map, valid_mask, None, params)  # resave without finalized
    _, _, finalized_second = store.load_background(params)

    assert finalized_second is None
    assert len(store._load_manifest()) == 1


def test_background_map_store_load_missing_raises(tmp_path):
    store = BackgroundMapStore(tmp_path / "background_maps")
    with pytest.raises(FileNotFoundError):
        store.load_background({"survey": "lsst"})


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------


def test_model_store_save_load_produces_identical_outputs(
    tmp_path, tiny_torch_model_class
):
    import torch

    store = ModelStore(tmp_path / "models")
    model = tiny_torch_model_class(in_features=3, out_features=2)
    x = torch.randn(5, 3)
    with torch.no_grad():
        expected = model(x)

    params = {
        "model_config": {"in_features": 3, "out_features": 2},
        "label_policy": "density",
    }
    store.save_model(
        model,
        config={"in_features": 3, "out_features": 2},
        metrics={"loss": 0.42},
        params=params,
    )

    loaded = store.load_model(tiny_torch_model_class, params)
    with torch.no_grad():
        actual = loaded(x)

    torch.testing.assert_close(actual, expected)


def test_model_store_resave_overwrites_weights_and_metrics(
    tmp_path, tiny_torch_model_class
):
    """Re-saving under the same params must replace the checkpoint directory's
    contents (weights, config, metrics) — not leave the first save's files
    mixed in with the second's."""
    import json

    import torch

    store = ModelStore(tmp_path / "models")
    params = {"model_config": {"in_features": 3, "out_features": 2}}

    first_model = tiny_torch_model_class(in_features=3, out_features=2)
    store.save_model(
        first_model,
        config={"in_features": 3, "out_features": 2},
        metrics={"loss": 1.0},
        params=params,
    )

    second_model = tiny_torch_model_class(in_features=3, out_features=2)
    with torch.no_grad():
        for p in second_model.parameters():
            p.add_(1.0)  # make weights provably different from first_model's
    x = torch.randn(4, 3)
    with torch.no_grad():
        expected_second = second_model(x)

    store.save_model(
        second_model,
        config={"in_features": 3, "out_features": 2},
        metrics={"loss": 0.1},
        params=params,
    )

    loaded = store.load_model(tiny_torch_model_class, params)
    with torch.no_grad():
        actual = loaded(x)
    torch.testing.assert_close(actual, expected_second)

    metrics_path = store.path(params) / "metrics.json"
    with open(metrics_path) as f:
        saved_metrics = json.load(f)
    assert saved_metrics == {"loss": 0.1}  # latest metrics, not the first save's

    assert len(store._load_manifest()) == 1


def test_model_store_save_non_module_raises(tmp_path):
    store = ModelStore(tmp_path / "models")
    with pytest.raises(TypeError):
        store.save_model(object(), config={}, metrics={}, params={"model_config": {}})


def test_model_store_load_missing_raises(tmp_path, tiny_torch_model_class):
    store = ModelStore(tmp_path / "models")
    with pytest.raises(FileNotFoundError):
        store.load_model(tiny_torch_model_class, {"model_config": {}})
