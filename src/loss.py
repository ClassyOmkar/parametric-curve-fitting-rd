"""
Loss function and evaluation metrics for parametric curve fitting.
"""

import numpy as np
from typing import Tuple
from model import predict


def l1_loss(params: np.ndarray, t_samples: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray) -> float:
    """
    Compute L1 loss (mean absolute error) between predicted and observed points.
    
    For each observed point, find the minimum distance to the predicted curve.
    
    Args:
        params: Array of [theta_deg, M, X]
        t_samples: Array of t values for dense curve sampling
        x_obs: Observed x coordinates
        y_obs: Observed y coordinates
        
    Returns:
        L1 distance (mean of minimum distances from observed points to curve)
    """
    theta_deg, M, X = params
    
    # Predict curve at many t samples for dense coverage
    x_pred, y_pred = predict(t_samples, theta_deg, M, X)
    
    # For each observed point, find minimum distance to predicted curve
    min_distances = []
    for x_o, y_o in zip(x_obs, y_obs):
        # Compute distances from this observed point to all predicted points
        dists = np.sqrt((x_pred - x_o)**2 + (y_pred - y_o)**2)
        # Take minimum distance
        min_distances.append(np.min(dists))
    
    # Return mean of minimum distances
    l1 = np.mean(min_distances)
    
    return l1


def compute_residuals(x_pred: np.ndarray, y_pred: np.ndarray, 
                      x_obs: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """
    Compute spatial residuals between predicted and observed points.
    
    Args:
        x_pred: Predicted x coordinates
        y_pred: Predicted y coordinates
        x_obs: Observed x coordinates
        y_obs: Observed y coordinates
        
    Returns:
        Array of residual distances
    """
    return np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)
