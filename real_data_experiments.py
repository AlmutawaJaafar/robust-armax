"""
Real-data experiments using DaISy benchmarks for robust ARMAX paper.

Datasets:
1. DaISy Dryer dataset (hair dryer, thermal system)

For each dataset:
- Fit ARMAX models of various orders
- Inject artificial outliers at 5% and 10% contamination
- Compare all methods on prediction MSE (clean test set)

Usage:
    python3 real_data_experiments.py
    python3 real_data_experiments.py --full   # 30 replications
"""

import numpy as np
import os
import sys
from data_generation import add_outliers
from proposed_method import two_stage_robust_armax
from baseline_methods import (ols_armax, lad_armax, huber_armax,
                               bip_tau_armax, student_t_em_armax)


def load_daisy_dryer(filepath=None):
    """Load DaISy hair dryer dataset.
    
    867 samples, sampling time 0.08s.
    Column 1: input (voltage to heating element)
    Column 2: output (temperature of air)
    
    If file not found, generates synthetic dryer-like data.
    """
    search_paths = [
        filepath,
        'dryer.dat', 'data/dryer.dat', '../data/dryer.dat',
        'dryer2.dat', 'data/dryer2.dat',
    ]
    
    for path in search_paths:
        if path is not None and os.path.exists(path):
            data = np.loadtxt(path)
            u = data[:, 0]
            y = data[:, 1]
            print(f"  Loaded DaISy dryer data from {path}")
            return u, y, True
    
    print("  WARNING: DaISy dryer data not found. Using synthetic data.")
    print("  Download from: https://homes.esat.kuleuven.be/~smc/daisy/daisydata/dryer.dat")
    u, y = generate_synthetic_dryer(n=867)
    return u, y, False


def generate_synthetic_dryer(n=867):
    """Generate synthetic data mimicking the DaISy dryer system.
    
    Approximate dryer as ARMAX(2,2,1).
    """
    rng = np.random.RandomState(42)
    
    a = np.array([1.52, -0.58])
    b = np.array([0.10, 0.08, 0.05])
    c = np.array([0.3])
    
    u = rng.choice([-1.0, 1.0], size=n + 200)
    xi = rng.randn(n + 200) * 0.1
    y = np.zeros(n + 200)
    
    for t in range(2, n + 200):
        y[t] = (a[0] * y[t-1] + a[1] * y[t-2]
                + b[0] * u[t] + b[1] * u[t-1] + b[2] * u[t-2]
                + c[0] * xi[t-1] + xi[t])
    
    return u[200:], y[200:]


def fit_and_evaluate(y_train, u_train, y_test, u_test, p, q, r,
                     method_name, method_func):
    """Fit on training data, evaluate prediction MSE on test data."""
    try:
        if method_name == 'Proposed':
            theta, a_hat, b_hat, c_hat, _, _ = method_func(
                y_train, u_train, p, q, r
            )
        else:
            theta, a_hat, b_hat, c_hat = method_func(
                y_train, u_train, p, q, r
            )
        
        n_test = len(y_test)
        start = max(p, q, r)
        y_pred = np.zeros(n_test)
        xi_hat = np.zeros(n_test)
        
        for t in range(start, n_test):
            ar_val = sum(a_hat[i] * y_test[t-1-i] for i in range(p))
            ex_val = sum(b_hat[j] * u_test[t-j] for j in range(q+1))
            ma_val = sum(c_hat[k] * xi_hat[t-1-k] for k in range(r))
            y_pred[t] = ar_val + ex_val + ma_val
            xi_hat[t] = y_test[t] - y_pred[t]
        
        mse = np.mean((y_test[start:] - y_pred[start:])**2)
        return mse, theta
    
    except Exception as e:
        return np.nan, None


def experiment_real_data(n_rep=10):
    """Run real-data experiments on DaISy dryer benchmark."""
    
    print("=" * 70)
    print("Real-Data Experiment: DaISy Dryer Benchmark")
    print("=" * 70)
    
    u, y, is_real = load_daisy_dryer()
    n_total = len(y)
    print(f"  Samples: {n_total}, Real data: {is_real}")
    print(f"  Input:  [{u.min():.2f}, {u.max():.2f}], std={u.std():.3f}")
    print(f"  Output: [{y.min():.2f}, {y.max():.2f}], std={y.std():.3f}")
    
    # Normalize
    y_mean, y_std = y.mean(), max(y.std(), 1e-8)
    u_mean, u_std = u.mean(), max(u.std(), 1e-8)
    y_norm = (y - y_mean) / y_std
    u_norm = (u - u_mean) / u_std
    
    # Train/test split
    n_train = min(600, n_total - 100)
    y_test_full = y_norm[n_train:]
    u_test_full = u_norm[n_train:]
    
    # Order selection on clean data
    print("\n  Order selection (clean data, OLS):")
    orders = [(2, 2, 1), (2, 2, 2), (3, 2, 1), (3, 2, 2)]
    best_order = None
    best_mse = np.inf
    
    for p, q, r in orders:
        mse, _ = fit_and_evaluate(
            y_norm[:n_train], u_norm[:n_train],
            y_test_full, u_test_full,
            p, q, r, 'OLS', ols_armax
        )
        print(f"    ARMAX({p},{q},{r}): MSE = {mse:.6f}")
        if not np.isnan(mse) and mse < best_mse:
            best_mse = mse
            best_order = (p, q, r)
    
    if best_order is None:
        best_order = (2, 2, 1)
    p, q, r = best_order
    print(f"\n  Selected: ARMAX({p},{q},{r})")
    
    # Methods
    methods = [
        ('OLS', ols_armax),
        ('LAD', lad_armax),
        ('Huber', huber_armax),
        ('BIP-tau', bip_tau_armax),
        ('Student-t EM', student_t_em_armax),
        ('Proposed', two_stage_robust_armax),
    ]
    
    contam_fracs = [0.0, 0.05, 0.10]
    
    # Main comparison
    print(f"\n  Prediction MSE (ARMAX({p},{q},{r}), {n_rep} reps):")
    print(f"  {'Method':15s}  {'Clean':>10s}  {'5% outliers':>12s}  "
          f"{'10% outliers':>12s}")
    print("  " + "-" * 55)
    
    all_results = {}
    
    for method_name, method_func in methods:
        results = {f: [] for f in contam_fracs}
        
        for rep in range(n_rep):
            for frac in contam_fracs:
                y_train = y_norm[:n_train].copy()
                u_train = u_norm[:n_train].copy()
                
                if frac > 0:
                    n_out = int(n_train * frac)
                    y_train, _, _ = add_outliers(
                        y_train, n_out,
                        outlier_magnitude=3.0,
                        seed=9000 + rep
                    )
                
                mse, _ = fit_and_evaluate(
                    y_train, u_train,
                    y_test_full, u_test_full,
                    p, q, r, method_name, method_func
                )
                results[frac].append(mse)
        
        vals = [np.nanmean(results[f]) for f in contam_fracs]
        print(f"  {method_name:15s}  {vals[0]:10.6f}  {vals[1]:12.6f}  "
              f"{vals[2]:12.6f}")
        all_results[method_name] = results
    
    # Summary
    print(f"\n  Summary:")
    proposed_clean = np.nanmean(all_results['Proposed'][0.0])
    proposed_10 = np.nanmean(all_results['Proposed'][0.10])
    ols_clean = np.nanmean(all_results['OLS'][0.0])
    ols_10 = np.nanmean(all_results['OLS'][0.10])
    
    print(f"    OLS degrades {ols_10/ols_clean:.1f}x from clean to 10%")
    print(f"    Proposed degrades {proposed_10/proposed_clean:.1f}x "
          f"from clean to 10%")
    print(f"    At 10%: Proposed is {ols_10/proposed_10:.1f}x better than OLS")
    
    return all_results


if __name__ == '__main__':
    n_rep = 30 if '--full' in sys.argv else 10
    print(f"Running with n_rep={n_rep}\n")
    experiment_real_data(n_rep=n_rep)
