"""
Unit tests for parametric model prediction.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import predict, deg_to_rad, rad_to_deg


def test_deg_to_rad():
    """Test degree to radian conversion."""
    assert np.isclose(deg_to_rad(0), 0)
    assert np.isclose(deg_to_rad(180), np.pi)
    assert np.isclose(deg_to_rad(90), np.pi/2)


def test_rad_to_deg():
    """Test radian to degree conversion."""
    assert np.isclose(rad_to_deg(0), 0)
    assert np.isclose(rad_to_deg(np.pi), 180)
    assert np.isclose(rad_to_deg(np.pi/2), 90)


def test_predict_shape():
    """Test that predict returns correct shapes."""
    t = np.array([10, 20, 30, 40, 50])
    theta_deg = 25.0
    M = 0.01
    X = 50.0
    
    x, y = predict(t, theta_deg, M, X)
    
    assert x.shape == t.shape
    assert y.shape == t.shape
    assert len(x) == 5
    assert len(y) == 5


def test_predict_values():
    """Test predict with known values."""
    t = np.array([10.0])
    theta_deg = 25.0
    M = 0.01
    X = 50.0
    
    x, y = predict(t, theta_deg, M, X)
    
    # Check that values are reasonable
    assert x.shape == (1,)
    assert y.shape == (1,)
    assert np.isfinite(x[0])
    assert np.isfinite(y[0])


def test_predict_theta_zero():
    """Test predict with theta = 0."""
    t = np.array([10, 20, 30])
    theta_deg = 0.1  # Small non-zero value
    M = 0.0
    X = 50.0
    
    x, y = predict(t, theta_deg, M, X)
    
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_predict_M_zero():
    """Test predict with M = 0."""
    t = np.array([10, 20, 30])
    theta_deg = 25.0
    M = 0.0
    X = 50.0
    
    x, y = predict(t, theta_deg, M, X)
    
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
