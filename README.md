# Robust ARMAX Identification with Propagation-Aware Reweighting

Python implementation of the two-stage robust ARMAX estimator from:

> **Two-Stage Robust ARMAX Identification with Propagation-Aware Reweighting under Adversarial Output Outliers**  
> [Author Name], submitted to *Int. J. Robust and Nonlinear Control*, 2026.

## Method

The proposed estimator identifies ARMAX(p,q,r) models from data corrupted by adversarial output outliers. The key insight is that a single outlier propagates through both the AR lags and the MA residual lags, contaminating (p+r+1) regressor rows. The algorithm:

- **Stage 1:** Propagation-aware reweighted Huber regression for AR and exogenous parameters, with adaptive detection threshold
- **Stage 2:** Damped iterative MA estimation from cleaned residuals

## Files

| File | Description |
|------|-------------|
| `data_generation.py` | ARMAX simulation, outlier injection, regressor construction |
| `proposed_method.py` | Two-stage robust ARMAX estimator (Algorithm 1 in paper) |
| `baseline_methods.py` | 8 baselines: OLS, LAD, Huber, ARX-only, BIP-ARMA, Student-t EM, Huber-PEM |
| `experiments.py` | 9 experiments (Tables 1-10 in paper) |
| `real_data_experiments.py` | DaISy hair dryer benchmark (Table 9 in paper) |

## Requirements

- Python 3.8+
- NumPy
- SciPy

No other dependencies.

## Usage

```bash
# Quick test (5 replications, ~5 minutes)
python experiments.py

# Full experiments for paper (30 replications, ~1 hour)
python experiments.py --full

# DaISy real-data experiment (requires dryer.dat)
# Download from: https://homes.esat.kuleuven.be/~smc/daisy/daisydata/dryer.dat
python real_data_experiments.py --full
```

## Key Results

| Method | Clean | 2% outliers | 5% outliers | 10% outliers |
|--------|-------|-------------|-------------|--------------|
| OLS | 0.316 | 1.546 | 2.192 | 2.427 |
| Huber-PEM | 0.259 | 1.152 | 1.144 | 1.151 |
| BIP-ARMA | 0.264 | 1.123 | 1.010 | 0.967 |
| Student-t EM | 0.318 | 0.903 | 0.931 | 0.916 |
| **Proposed** | **0.361** | **0.359** | **0.376** | **0.369** |

The proposed method is **contamination-invariant**: error changes by <5% from 0% to 10% contamination.

## License

MIT License
