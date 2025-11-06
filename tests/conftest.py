"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Generate sample xy data for testing."""
    np.random.seed(42)
    n = 50
    x = np.random.uniform(60, 110, n)
    y = np.random.uniform(45, 70, n)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def known_params():
    """Known parameters for synthetic data generation."""
    return {
        'theta_deg': 25.0,
        'M': 0.01,
        'X': 50.0
    }
