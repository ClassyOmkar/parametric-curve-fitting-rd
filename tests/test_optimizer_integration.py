"""
Integration tests for optimizer with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import predict
from optimizer import fit_params
from data_loader import get_uniform_t_values


def test_optimizer_with_synthetic_data(known_params):
    """Test optimizer with synthetic data from known parameters."""
    # Generate synthetic data
    t_true = get_uniform_t_values(100)
    x_true, y_true = predict(
        t_true,
        known_params['theta_deg'],
        known_params['M'],
        known_params['X']
    )
    
    # Add small noise
    np.random.seed(42)
    noise_level = 0.1
    x_noisy = x_true + np.random.normal(0, noise_level, len(x_true))
    y_noisy = y_true + np.random.normal(0, noise_level, len(y_true))
    
    df = pd.DataFrame({'x': x_noisy, 'y': y_noisy})
    
    # Fit parameters
    result = fit_params(df, n_samples=50, n_restarts=5, seed=42)
    
    # Check that recovered parameters are close to true values
    assert 'theta_deg' in result
    assert 'M' in result
    assert 'X' in result
    assert 'l1' in result
    
    # Tolerances for parameter recovery
    theta_tolerance = 5.0  # degrees
    M_tolerance = 0.02
    X_tolerance = 10.0
    
    assert abs(result['theta_deg'] - known_params['theta_deg']) < theta_tolerance
    assert abs(result['M'] - known_params['M']) < M_tolerance
    assert abs(result['X'] - known_params['X']) < X_tolerance
    
    # L1 score should be small for synthetic data with small noise
    assert result['l1'] < 5.0


def test_optimizer_convergence(sample_data):
    """Test that optimizer converges."""
    result = fit_params(sample_data, n_samples=30, n_restarts=3, seed=42)
    
    # Check that all parameters are within valid bounds
    assert 0 < result['theta_deg'] < 50
    assert -0.05 < result['M'] < 0.05
    assert 0 < result['X'] < 100
    
    # Check that L1 is finite and positive
    assert np.isfinite(result['l1'])
    assert result['l1'] > 0
