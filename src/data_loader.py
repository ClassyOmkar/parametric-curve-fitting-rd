"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    """
    Load xy data from CSV file and validate format.
    
    Args:
        path: Path to CSV file containing x,y columns
        
    Returns:
        DataFrame with validated x,y data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If data format is invalid
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Validate columns
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'x' and 'y' columns")
    
    # Validate sufficient data
    if len(df) < 10:
        raise ValueError(f"Insufficient data: {len(df)} rows (minimum 10 required)")
    
    # Check for NaN values
    if df[['x', 'y']].isna().any().any():
        raise ValueError("Data contains NaN values")
    
    return df[['x', 'y']]


def get_uniform_t_values(n: int, t_min: float = 6.0, t_max: float = 60.0) -> np.ndarray:
    """
    Generate uniform t values for sampling the parametric curve.
    
    Args:
        n: Number of uniform samples
        t_min: Minimum t value (default: 6.0)
        t_max: Maximum t value (default: 60.0)
        
    Returns:
        Array of n uniformly spaced t values
    """
    return np.linspace(t_min, t_max, n)
