"""
Utility functions for the parametric curve fitting project.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any


def save_params_json(params: Dict[str, Any], output_path: str) -> None:
    """
    Save parameters to JSON file.
    
    Args:
        params: Dictionary of parameters to save
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)


def load_params_json(input_path: str) -> Dict[str, Any]:
    """
    Load parameters from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary of parameters
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def validate_bounds(theta_deg: float, M: float, X: float) -> bool:
    """
    Validate that parameters are within allowed bounds.
    
    Args:
        theta_deg: Angle in degrees
        M: Exponential parameter
        X: X-offset parameter
        
    Returns:
        True if all parameters are within bounds
    """
    theta_valid = 0 < theta_deg < 50
    M_valid = -0.05 < M < 0.05
    X_valid = 0 < X < 100
    
    return theta_valid and M_valid and X_valid
