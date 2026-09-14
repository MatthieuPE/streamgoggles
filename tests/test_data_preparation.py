"""Test data_preparation.py's shared quality-cut infrastructure: Cut,
apply_cuts, apply_magnitude_clipping. These live here (not in background.py
or injector.py) because they're applied identically to both the background
catalog and the injected stream catalog (decision 13).
"""

import numpy as np
import pandas as pd
import pytest

from streamgoggles.data_preparation import Cut, apply_cuts, apply_magnitude_clipping

pytestmark = pytest.mark.data_preparation

# ---------------------------------------------------------------------------
# Cut / _resolve_cut_quantity
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_df():
    return pd.DataFrame(
        {
            "lsst_yr1_g_obs": np.array([20.0, 22.0, 24.0, 26.0]),
            "lsst_yr1_g_err": np.array([0.01, 0.05, 0.15, 0.3]),
            "extendedness": np.array([0.0, 1.0, 0.0, 0.9]),
        }
    )


def test_cut_mag_quantity(catalog_df):
    cut = Cut(quantity="mag", band="g", op="<", value=25.0)
    mask = cut.apply(catalog_df, namespace="lsst_yr1")
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_cut_snr_quantity(catalog_df):
    # snr = 1.0857/err -> [108.57, 21.7, 7.24, 3.62]; threshold >5 keeps first 3.
    cut = Cut(quantity="snr", band="g", op=">", value=5.0)
    mask = cut.apply(catalog_df, namespace="lsst_yr1")
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_cut_generic_column_quantity(catalog_df):
    cut = Cut(quantity="extendedness", op="<", value=0.5)
    mask = cut.apply(catalog_df)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_cut_mag_requires_band():
    with pytest.raises(ValueError, match="band"):
        Cut(quantity="mag", op="<", value=25.0).apply(pd.DataFrame({"x": [1]}))


def test_cut_snr_requires_band():
    with pytest.raises(ValueError, match="band"):
        Cut(quantity="snr", op=">", value=5.0).apply(pd.DataFrame({"x": [1]}))


def test_cut_unknown_column_raises(catalog_df):
    with pytest.raises(ValueError, match="not 'mag'/'snr'"):
        Cut(quantity="not_a_real_column", op=">", value=1.0).apply(catalog_df)


def test_cut_unknown_operator_raises(catalog_df):
    with pytest.raises(ValueError, match="operator"):
        Cut(quantity="extendedness", op="~=", value=0.5).apply(catalog_df)


def test_cut_incomplete_spec_raises(catalog_df):
    with pytest.raises(ValueError, match="callable"):
        Cut(quantity="extendedness", op="<").apply(catalog_df)  # no value


def test_cut_all_operators(catalog_df):
    ext = catalog_df["extendedness"]
    thresholds = {">": 0.5, "<": 0.5, ">=": 1.0, "<=": 0.0, "==": 0.0, "!=": 0.0}
    pandas_method = {
        ">": "gt",
        "<": "lt",
        ">=": "ge",
        "<=": "le",
        "==": "eq",
        "!=": "ne",
    }
    for op, value in thresholds.items():
        mask = Cut(quantity="extendedness", op=op, value=value).apply(catalog_df)
        expected = getattr(ext, pandas_method[op])(value)
        np.testing.assert_array_equal(mask, expected.to_numpy())


def test_cut_callable_based(catalog_df):
    cut = Cut(callable="tests.test_data_preparation:_even_index_cut")
    mask = cut.apply(catalog_df)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_cut_callable_malformed_string_raises(catalog_df):
    with pytest.raises(ValueError, match="module:function"):
        Cut(callable="not_a_valid_spec").apply(catalog_df)


def test_cut_callable_wrong_shape_raises(catalog_df):
    with pytest.raises(ValueError, match="shape"):
        Cut(callable="tests.test_data_preparation:_wrong_shape_cut").apply(catalog_df)


def _even_index_cut(df: pd.DataFrame) -> np.ndarray:
    return (np.arange(len(df)) % 2) == 0


def _wrong_shape_cut(df: pd.DataFrame) -> np.ndarray:
    return np.array([True, False])  # deliberately wrong length


# ---------------------------------------------------------------------------
# apply_cuts
# ---------------------------------------------------------------------------


def test_apply_cuts_sequential_rejection(catalog_df):
    cuts = [
        Cut(quantity="mag", band="g", op="<", value=25.0),
        Cut(quantity="extendedness", op="<", value=0.5),
    ]
    out = apply_cuts(catalog_df, cuts, namespace="lsst_yr1")
    # Row 0: mag<25 T, ext<0.5 T -> kept. Row 1: mag<25 T, ext<0.5 F -> dropped.
    # Row 2: mag<25 T, ext<0.5 T -> kept. Row 3: mag<25 F -> dropped.
    assert len(out) == 2
    np.testing.assert_array_equal(out["lsst_yr1_g_obs"].to_numpy(), [20.0, 24.0])


def test_apply_cuts_empty_list_is_noop(catalog_df):
    out = apply_cuts(catalog_df, [], namespace="lsst_yr1")
    pd.testing.assert_frame_equal(out, catalog_df)


def test_apply_cuts_resets_index(catalog_df):
    cuts = [Cut(quantity="extendedness", op="<", value=0.5)]
    out = apply_cuts(catalog_df, cuts)
    assert list(out.index) == list(range(len(out)))


# ---------------------------------------------------------------------------
# apply_magnitude_clipping
# ---------------------------------------------------------------------------


def test_apply_magnitude_clipping_min_and_max(catalog_df):
    out = apply_magnitude_clipping(
        catalog_df, {"g": {"min": 21.0, "max": 25.0}}, namespace="lsst_yr1"
    )
    np.testing.assert_array_equal(out["lsst_yr1_g_obs"].to_numpy(), [22.0, 24.0])


def test_apply_magnitude_clipping_only_min(catalog_df):
    out = apply_magnitude_clipping(
        catalog_df, {"g": {"min": 23.0}}, namespace="lsst_yr1"
    )
    np.testing.assert_array_equal(out["lsst_yr1_g_obs"].to_numpy(), [24.0, 26.0])


def test_apply_magnitude_clipping_only_max(catalog_df):
    out = apply_magnitude_clipping(
        catalog_df, {"g": {"max": 22.0}}, namespace="lsst_yr1"
    )
    np.testing.assert_array_equal(out["lsst_yr1_g_obs"].to_numpy(), [20.0, 22.0])


def test_apply_magnitude_clipping_empty_dict_is_noop(catalog_df):
    out = apply_magnitude_clipping(catalog_df, {}, namespace="lsst_yr1")
    pd.testing.assert_frame_equal(out, catalog_df)
