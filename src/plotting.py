"""
Plotting utilities for visualization and saving results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from model import predict
from loss import compute_residuals
from data_loader import get_uniform_t_values


def plot_fit(
    xy_df: pd.DataFrame,
    params: dict,
    n_curve_points: int = 500,
    output_path: Optional[str] = None
) -> None:
    """
    Plot observed data points and fitted parametric curve.
    
    Args:
        xy_df: DataFrame with observed x,y data
        params: Dictionary with theta_deg, M, X parameters
        n_curve_points: Number of points for smooth curve
        output_path: Optional path to save figure
    """
    # Generate smooth curve
    t_curve = get_uniform_t_values(n_curve_points)
    x_curve, y_curve = predict(t_curve, params['theta_deg'], params['M'], params['X'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot observed points
    ax.scatter(xy_df['x'], xy_df['y'], alpha=0.5, s=20, label='Observed Data', color='blue')
    
    # Plot fitted curve
    ax.plot(x_curve, y_curve, 'r-', linewidth=2, label='Fitted Curve', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Parametric Curve Fit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved fit plot to {output_path}")
    
    plt.close()


def plot_residuals(
    xy_df: pd.DataFrame,
    params: dict,
    n_samples: int = 200,
    output_path: Optional[str] = None
) -> None:
    """
    Plot residuals between observed and predicted points.
    
    Args:
        xy_df: DataFrame with observed x,y data
        params: Dictionary with theta_deg, M, X parameters
        n_samples: Number of samples for residual computation
        output_path: Optional path to save figure
    """
    # Generate predictions at uniform t samples
    t_samples = get_uniform_t_values(n_samples)
    x_pred, y_pred = predict(t_samples, params['theta_deg'], params['M'], params['X'])
    
    # Compute residuals (assuming order matches)
    x_obs = xy_df['x'].values[:n_samples]
    y_obs = xy_df['y'].values[:n_samples]
    
    residuals = compute_residuals(x_pred[:len(x_obs)], y_pred[:len(y_obs)], x_obs, y_obs)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs index
    ax1.plot(residuals, 'o-', alpha=0.6, markersize=4)
    ax1.set_xlabel('Data Point Index', fontsize=12)
    ax1.set_ylabel('Residual Distance', fontsize=12)
    ax1.set_title('Residuals vs Data Point', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Residual Distance', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(np.mean(residuals), color='r', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved residuals plot to {output_path}")
    
    plt.close()
