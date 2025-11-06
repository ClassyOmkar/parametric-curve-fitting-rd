"""
Parameter optimization using scipy.optimize with bounds enforcement.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
from loss import l1_loss
from data_loader import get_uniform_t_values
from utils import validate_bounds

logger = logging.getLogger(__name__)


def fit_params(
    xy_df: pd.DataFrame,
    initial_guess: Optional[Dict[str, float]] = None,
    n_samples: int = 100,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_restarts: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Fit parametric curve parameters to observed data.
    
    Args:
        xy_df: DataFrame with x,y columns
        initial_guess: Optional initial parameter guess
        n_samples: Number of uniform t samples
        bounds: Optional parameter bounds
        n_restarts: Number of random restarts
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with best-fit parameters and L1 score
    """
    np.random.seed(seed)
    
    # Default bounds
    if bounds is None:
        bounds = {
            'theta_deg': (0.1, 49.9),
            'M': (-0.049, 0.049),
            'X': (0.1, 99.9)
        }
    
    # Extract observed data
    x_obs = xy_df['x'].values
    y_obs = xy_df['y'].values
    
    # Generate uniform t samples
    t_samples = get_uniform_t_values(n_samples)
    
    # Define bounds for scipy
    scipy_bounds = [
        bounds['theta_deg'],
        bounds['M'],
        bounds['X']
    ]
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            'theta_deg': 25.0,
            'M': 0.0,
            'X': 50.0
        }
    
    best_result = None
    best_loss = float('inf')
    
    logger.info(f"Starting optimization with {n_restarts} restarts")
    
    # Try multiple random restarts
    for i in range(n_restarts):
        if i == 0:
            # First iteration: use provided initial guess
            x0 = np.array([
                initial_guess['theta_deg'],
                initial_guess['M'],
                initial_guess['X']
            ])
        else:
            # Random initialization within bounds
            x0 = np.array([
                np.random.uniform(*bounds['theta_deg']),
                np.random.uniform(*bounds['M']),
                np.random.uniform(*bounds['X'])
            ])
        
        # Optimize using Nelder-Mead (robust to local minima)
        result = minimize(
            l1_loss,
            x0,
            args=(t_samples, x_obs, y_obs),
            method='Nelder-Mead',
            bounds=scipy_bounds,
            options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
            logger.info(f"Restart {i+1}/{n_restarts}: New best L1 = {best_loss:.6f}")
    
    # Also try differential evolution for global optimization
    logger.info("Running differential evolution for global search")
    de_result = differential_evolution(
        l1_loss,
        scipy_bounds,
        args=(t_samples, x_obs, y_obs),
        seed=seed,
        maxiter=1000,
        atol=1e-6,
        tol=1e-6
    )
    
    if de_result.fun < best_loss:
        best_loss = de_result.fun
        best_result = de_result
        logger.info(f"Differential evolution: New best L1 = {best_loss:.6f}")
    
    # Extract best parameters
    theta_deg, M, X = best_result.x
    
    # Validate bounds
    if not validate_bounds(theta_deg, M, X):
        logger.warning("Optimal parameters are outside expected bounds")
    
    # Convert theta to radians for output
    theta_rad = np.deg2rad(theta_deg)
    
    logger.info(f"Optimization complete: θ={theta_deg:.4f}°, M={M:.6f}, X={X:.4f}, L1={best_loss:.6f}")
    
    return {
        'theta_deg': float(theta_deg),
        'theta_rad': float(theta_rad),
        'M': float(M),
        'X': float(X),
        'l1': float(best_loss)
    }
