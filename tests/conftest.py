"""
Shared pytest fixtures for the streamobs test suite.
"""

import numpy as np
import pytest

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


def test_seed(seed):
    """Test that the seed fixture returns a fixed integer."""
    assert isinstance(seed, int)
    assert seed == 42