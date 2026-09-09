"""
Shared pytest fixtures for the streamobs test suite.
"""

import os

import numpy as np
import pytest
import yaml

# Must be set before healpy or torch actually gets imported by any test
# module (both are only imported lazily, well after conftest.py loads, so
# setting it here at collection time is early enough). healpy's
# hp.smoothing() and torch (tiny_torch_model_class fixture, below) each load
# their own OpenMP runtime, and loading both in the same process aborts with
# a duplicate-OpenMP-runtime segfault on this environment (confirmed via
# isolation: reproducible with just rasterize.py's smoothing path +
# test_storage.py's torch fixtures, nothing else). This is a known, common
# macOS conda interoperability issue (conda's MKL/libomp vs PyTorch's bundled
# libomp), not a bug in either library; safe here since this codebase never
# runs MKL and torch numerics concurrently in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture(scope="session")
def rng(seed):
    """Random number generator initialized with a fixed seed."""
    return np.random.default_rng(seed)


@pytest.fixture(scope="session")
def verbose():
    """Control verbosity of test output."""
    return True


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def write_yaml(tmp_path):
    """Factory fixture: write a dict as YAML to a temp file, return its path.

    Centralizes the write-a-synthetic-config-then-load-it pattern used across
    config.py's tests, so synthetic fixture content lives in one place instead
    of being re-typed as an inline `yaml.safe_dump({...})` in every test.
    """

    def _write(data: dict, filename: str = "config.yaml"):
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


@pytest.fixture
def stream_dict_single_fixed_width():
    """Minimal stream-config dict with exactly one fixed parameter (width)."""
    return {"width": 0.2}


@pytest.fixture
def stream_dict_mixed_free():
    """Stream-config dict with one free uniform param and one free discrete param.

    Shared by tests that need >1 free parameter (free_parameters() detection,
    eval-grid Cartesian product) without caring about the rest of a realistic
    stream config.
    """
    return {
        "orientation": {"min": 0.0, "max": 180.0},
        "age": {"values": [10.0, 12.0, 13.5]},
    }


# ---------------------------------------------------------------------------
# "Default" config dicts
#
# These are frozen, hand-maintained snapshots of a realistic training/eval/
# background/matched-filter config — deliberately NOT read from config/*.yaml.
# Tests built on these run against a fixed, known shape regardless of later
# edits to the real config files; keep the two in sync by hand when the real
# files' schema changes, don't have one load the other.
# ---------------------------------------------------------------------------


@pytest.fixture
def default_injection_grid_dict():
    """Representative training injection-grid config (first milestone shape)."""
    return {
        "morphology": "uniform",
        "richness": {"surface_brightness": 30.0},
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 17.5,
        "age": 12.0,
        "z": 0.0004,
        "orientation": {"min": 0.0, "max": 180.0},
        "position": "uniform_in_footprint",
        "background_fraction": 0.0,
        "label": {"policy": "density", "normalization": "max", "dilate_to_width": True},
        "persist": False,
    }


@pytest.fixture
def default_eval_grid_dict():
    """Representative persisted eval-grid config: only richness is free."""
    return {
        "morphology": "uniform",
        "richness": {
            "surface_brightness": {"values": [26.0, 27.2, 28.4, 29.6, 30.8, 32.0]}
        },
        "width": 0.2,
        "length": 8.0,
        "distance_modulus": 17.5,
        "age": 12.0,
        "z": 0.0004,
        "orientation": 90.0,
        "position": "uniform_in_footprint",
        "background_fraction": 0.0,
        "label": {"policy": "density", "normalization": "max", "dilate_to_width": True},
        "persist": True,
    }


@pytest.fixture
def default_background_dict():
    """Representative background config: LSST yr1 forecast via `light`."""
    return {
        "background": {
            "source": "light",
            "survey": "lsst",
            "release": "yr1",
            "light": {},
            "dust_correction": {"enabled": False},
        },
        "study_region": {
            "center_ra": 0.0,
            "center_dec": -30.0,
            "width_deg": 70.0,
            "height_deg": 20.0,
        },
        "cuts": [
            {"quantity": "snr", "band": "g", "op": ">", "value": 5},
            {"quantity": "snr", "band": "r", "op": ">", "value": 5},
        ],
        "magnitude_clipping": {
            "g": {"min": 16.0, "max": 26.5},
            "r": {"min": 16.0, "max": 26.0},
        },
    }


@pytest.fixture
def default_matched_filter_dict():
    """Representative matched-filter config: fixed distance modulus, finalize off."""
    return {
        "reference_isochrone": {"age": 12.5, "z": 0.0002},
        "bands": ["g", "r"],
        "pixelization": {
            "nside": 128,
            "projection": "gnomonic",
            "rotation_deg": 0.0,
            "image_size_pix": [256, 256],
            "pixel_scale_deg": 0.05,
            "interpolate": True,
        },
        "distance": {"mode": "fixed", "value": 17.5},
        "finalize": {
            "enabled": False,
            "smoothing_deg": 0.1,
            "background_subtract": False,
        },
    }


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_sample():
    """A minimal, valid Sample for storage round-trip tests."""
    from streamgoggles.sample import Sample

    gen = np.random.default_rng(0)
    map_stack = gen.normal(size=(2, 4, 4)).astype(np.float32)
    label_stack = gen.uniform(size=(2, 4, 4)).astype(np.float32)
    valid_mask = np.ones((4, 4), dtype=bool)
    return Sample(
        map_stack=map_stack,
        label_stack=label_stack,
        valid_mask=valid_mask,
        params={"width": 0.2, "age": 12.0},
        metadata={"seed": 0},
    )


@pytest.fixture
def tiny_torch_model_class():
    """A minimal torch.nn.Module class for ModelStore round-trip tests."""
    from torch import nn

    class TinyLinearModel(nn.Module):
        def __init__(self, in_features=2, out_features=1):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    return TinyLinearModel
