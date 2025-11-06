"""
Parametric curve model implementation.
"""

import numpy as np
from typing import Tuple


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return np.rad2deg(radians)


def predict(t: np.ndarray, theta_deg: float, M: float, X: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parametric curve x(t), y(t) for given parameters.
    
    Parametric equations:
        x(t) = t * cos(θ) - exp(M * |t|) * sin(0.3 * t) * sin(θ) + X
        y(t) = 42 + t * sin(θ) + exp(M * |t|) * sin(0.3 * t) * cos(θ)
    
    Args:
        t: Array of t values
        theta_deg: Angle parameter in degrees (0 < θ < 50)
        M: Exponential parameter (-0.05 < M < 0.05)
        X: X-offset parameter (0 < X < 100)
        
    Returns:
        Tuple of (x, y) arrays
    """
    # Convert theta to radians
    theta_rad = deg_to_rad(theta_deg)
    
    # Compute common terms
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    exp_term = np.exp(M * np.abs(t))
    sin_03t = np.sin(0.3 * t)
    
    # Compute x(t) and y(t)
    x = t * cos_theta - exp_term * sin_03t * sin_theta + X
    y = 42 + t * sin_theta + exp_term * sin_03t * cos_theta
    
    return x, y
