# Mathematical Derivation and Approach

## Problem Statement

Given a dataset of (x, y) points sampled from a parametric curve, estimate the unknown parameters θ, M, and X.

## Parametric Model

The parametric equations are:

```
x(t) = t * cos(θ) - exp(M * |t|) * sin(0.3 * t) * sin(θ) + X
y(t) = 42 + t * sin(θ) + exp(M * |t|) * sin(0.3 * t) * cos(θ)
```

### Parameter Interpretation

- **θ (theta)**: Rotation angle affecting the curve orientation
- **M**: Controls exponential growth/decay in the oscillatory term
- **X**: Horizontal offset of the curve
- **t**: Parameter variable, t ∈ (6, 60)

### Constraints

The parameters must satisfy:
- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100

## Optimization Formulation

### Objective Function

We minimize the L1 distance (mean absolute error) between observed and predicted points:

```
L1(θ, M, X) = (1/n) * Σ sqrt((x_pred_i - x_obs_i)² + (y_pred_i - y_obs_i)²)
```

where:
- n = number of data points
- (x_obs_i, y_obs_i) = observed data points
- (x_pred_i, y_pred_i) = predicted curve points at corresponding t values

### L1 vs L2 Loss

We chose L1 loss over L2 (mean squared error) because:
- L1 is more robust to outliers
- L1 provides better interpretability (units match the data)
- L1 works well with the dense sampling of the curve

## Optimization Strategy

### 1. Sampling Strategy

We use uniform sampling of the parameter t:
```
t_i = 6 + (54/n) * i, for i = 0, 1, ..., n-1
```

This ensures even coverage of the curve domain.

### 2. Multi-Start Optimization

To avoid local minima, we employ:
- **Random restarts**: Run optimization from multiple random initial points
- **Diverse initialization**: Sample initial guesses uniformly within parameter bounds
- **Best solution selection**: Choose parameters with lowest L1 score

### 3. Optimization Methods

We combine two approaches:

#### Nelder-Mead Simplex
- Derivative-free method
- Robust to noisy gradients
- Good for local refinement

#### Differential Evolution
- Global optimization algorithm
- Population-based stochastic search
- Excellent for finding global minimum

### 4. Convergence Criteria

Optimization terminates when:
- Maximum iterations reached (10,000 for Nelder-Mead)
- Function tolerance achieved (fatol = 1e-6)
- Parameter tolerance achieved (xatol = 1e-6)

## Implementation Details

### Numerical Stability

- Angle conversion: θ is input in degrees but converted to radians internally
- Absolute value: |t| ensures exp(M*|t|) is well-defined for negative t
- Bounds enforcement: scipy bounds prevent parameters from violating constraints

### Reproducibility

- Fixed random seed (default: 42) for all stochastic operations
- Deterministic optimization convergence with same initial conditions

## Validation

### Synthetic Data Test

The optimizer is validated using synthetic data:
1. Generate curve from known parameters
2. Add small Gaussian noise
3. Run optimization
4. Verify recovered parameters are close to ground truth

### Residual Analysis

Post-optimization, we analyze:
- Residual distribution (should be centered near zero)
- Spatial pattern of residuals (should be random, not systematic)
- Outliers (large residuals may indicate data quality issues)

## Results Interpretation

### Parameter Confidence

Without uncertainty quantification, parameter estimates should be interpreted as:
- Point estimates from optimization
- Dependent on data quality and sampling density
- Subject to local minima (mitigated by multi-start)

### Model Assumptions

The approach assumes:
- Data points lie approximately on the parametric curve
- Noise is relatively small compared to signal
- The parametric form is correct

## Computational Complexity

- **Time complexity**: O(k * m * n)
  - k = number of restarts
  - m = iterations per optimization
  - n = number of data points
  
- **Space complexity**: O(n) for data storage

## Future Improvements

Potential enhancements:
- Bayesian optimization for parameter uncertainty
- Adaptive sampling based on curve curvature
- Constraint-aware optimization with penalty methods
- Parallel multi-start using multiprocessing
