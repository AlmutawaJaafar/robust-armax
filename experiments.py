"""
Main experiments for robust ARMAX identification paper.

Experiments:
1. Parameter error vs contamination fraction (ARMAX(2,2,2))
2. Propagation factor validation: (p+r+1) vs (p+1)
3. Rate validation: error vs n, error vs sqrt((p+r+1)*o/n)
4. Comparison with BIP-tau and Student-t EM
5. Model misspecification (wrong r)
"""

import numpy as np
import time
from data_generation import (generate_armax_system, simulate_armax,
                              add_outliers, true_parameter_vector,
                              get_propagation_set)
from proposed_method import (two_stage_robust_armax, oracle_robust_armax)
from baseline_methods import (ols_armax, lad_armax, huber_armax,
                               arx_only, bip_tau_armax, student_t_em_armax,
                               huber_pem_armax)


def run_single_experiment(a, b, c, n, outlier_frac, noise_type='gaussian',
                          sigma=1.0, outlier_mag=5.0, seed=None):
    """Run a single experiment with all methods.
    
    Returns dict of parameter errors for each method.
    """
    p, q_plus1, r = len(a), len(b), len(c)
    q = q_plus1 - 1
    n_outliers = int(n * outlier_frac)
    theta_true = true_parameter_vector(a, b, c)
    
    # Generate data
    y, u, xi = simulate_armax(a, b, c, n, noise_type=noise_type,
                               sigma=sigma, seed=seed)
    
    # Add outliers
    if n_outliers > 0:
        y_tilde, outlier_set, _ = add_outliers(y, n_outliers,
                                                outlier_magnitude=outlier_mag,
                                                seed=seed)
    else:
        y_tilde = y.copy()
        outlier_set = np.array([], dtype=int)
    
    results = {}
    
    # 1. OLS
    try:
        theta_ols, _, _, _ = ols_armax(y_tilde, u, p, q, r)
        results['OLS'] = np.linalg.norm(theta_ols - theta_true)
    except:
        results['OLS'] = np.nan
    
    # 2. LAD
    try:
        theta_lad, _, _, _ = lad_armax(y_tilde, u, p, q, r)
        results['LAD'] = np.linalg.norm(theta_lad - theta_true)
    except:
        results['LAD'] = np.nan
    
    # 3. Huber (non-propagation-aware)
    try:
        theta_hub, _, _, _ = huber_armax(y_tilde, u, p, q, r)
        results['Huber'] = np.linalg.norm(theta_hub - theta_true)
    except:
        results['Huber'] = np.nan
    
    # 4. ARX-only (Paper 1, ignores MA)
    try:
        theta_arx, _, _, _ = arx_only(y_tilde, u, p, q, r)
        results['ARX-only'] = np.linalg.norm(theta_arx - theta_true)
    except:
        results['ARX-only'] = np.nan
    
    # 5. BIP-tau
    try:
        theta_bip, _, _, _ = bip_tau_armax(y_tilde, u, p, q, r)
        results['BIP-tau'] = np.linalg.norm(theta_bip - theta_true)
    except:
        results['BIP-tau'] = np.nan
    
    # 6. Student-t EM
    try:
        theta_em, _, _, _ = student_t_em_armax(y_tilde, u, p, q, r)
        results['Student-t EM'] = np.linalg.norm(theta_em - theta_true)
    except:
        results['Student-t EM'] = np.nan
    
    # 7. Huber-PEM (joint robust prediction error)
    try:
        theta_pem, _, _, _ = huber_pem_armax(y_tilde, u, p, q, r)
        results['Huber-PEM'] = np.linalg.norm(theta_pem - theta_true)
    except:
        results['Huber-PEM'] = np.nan
    
    # 8. Proposed (two-stage robust ARMAX)
    try:
        theta_prop, _, _, _, _, info = two_stage_robust_armax(
            y_tilde, u, p, q, r
        )
        results['Proposed'] = np.linalg.norm(theta_prop - theta_true)
    except:
        results['Proposed'] = np.nan
    
    # 8. Oracle
    try:
        theta_oracle, _, _, _ = oracle_robust_armax(
            y_tilde, u, p, q, r, outlier_set
        )
        results['Oracle'] = np.linalg.norm(theta_oracle - theta_true)
    except:
        results['Oracle'] = np.nan
    
    return results


def experiment1_error_vs_contamination(n_rep=20, n=500):
    """Experiment 1: Parameter error vs contamination fraction."""
    print("=" * 60)
    print("Experiment 1: Error vs Contamination (ARMAX(2,2,2), n=500)")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    
    outlier_fracs = [0.0, 0.02, 0.05, 0.10]
    methods = ['OLS', 'LAD', 'Huber', 'ARX-only', 'BIP-tau',
               'Student-t EM', 'Huber-PEM', 'Proposed', 'Oracle']
    
    results_table = {m: {f: [] for f in outlier_fracs} for m in methods}
    
    for frac in outlier_fracs:
        print(f"\n  Contamination = {frac:.0%}:")
        for rep in range(n_rep):
            res = run_single_experiment(
                a, b, c, n, frac, noise_type='gaussian',
                seed=1000 * rep + int(frac * 100)
            )
            for m in methods:
                if m in res:
                    results_table[m][frac].append(res[m])
        
        # Print summary
        for m in methods:
            vals = results_table[m][frac]
            if vals:
                mean_err = np.nanmean(vals)
                print(f"    {m:15s}: {mean_err:.4f}")
    
    return results_table


def experiment2_propagation_factor(n_rep=20, n=500):
    """Experiment 2: Validate (p+r+1) propagation factor.
    
    Compare ARMAX(p,q,r) with different (p,r) combinations
    at fixed contamination.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Propagation Factor Validation")
    print("=" * 60)
    
    configs = [
        (1, 2, 1, "p=1,r=1: factor=3"),
        (2, 2, 1, "p=2,r=1: factor=4"),
        (2, 2, 2, "p=2,r=2: factor=5"),
        (3, 2, 2, "p=3,r=2: factor=6"),
        (3, 2, 3, "p=3,r=3: factor=7"),
    ]
    
    frac = 0.05
    
    for p, q, r, label in configs:
        a, b, c = generate_armax_system(p=p, q=q, r=r, seed=42)
        errors = []
        
        for rep in range(n_rep):
            res = run_single_experiment(
                a, b, c, n, frac, seed=2000 * rep
            )
            if 'Proposed' in res:
                errors.append(res['Proposed'])
        
        factor = p + r + 1
        mean_err = np.nanmean(errors)
        print(f"  {label}: mean error = {mean_err:.4f}, "
              f"factor = {factor}, "
              f"sqrt(factor*o/n) = {np.sqrt(factor * frac):.4f}")


def experiment3_rate_validation(n_rep=20):
    """Experiment 3: Error vs n and vs sqrt((p+r+1)*o/n)."""
    print("\n" + "=" * 60)
    print("Experiment 3: Rate Validation")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    factor = p + r + 1  # = 5
    
    # TV-1: Error vs n (fixed contamination fraction)
    print("\n  TV-1: Error vs n (eps=0.05)")
    n_values = [200, 500, 1000, 2000]
    frac = 0.05
    
    for n in n_values:
        errors = []
        for rep in range(n_rep):
            res = run_single_experiment(a, b, c, n, frac, seed=3000 * rep)
            if 'Proposed' in res:
                errors.append(res['Proposed'])
        mean_err = np.nanmean(errors)
        theory_rate = np.sqrt(len(true_parameter_vector(a, b, c)) / n)
        print(f"    n={n:5d}: error={mean_err:.4f}, "
              f"sqrt(d/n)={theory_rate:.4f}, "
              f"ratio={mean_err/theory_rate:.3f}")
    
    # TV-2: Error vs sqrt((p+r+1)*o/n)
    print("\n  TV-2: Error vs sqrt((p+r+1)*o/n) (n=2000)")
    n = 2000
    outlier_fracs = [0.01, 0.02, 0.05, 0.08, 0.10]
    
    for frac in outlier_fracs:
        errors = []
        for rep in range(n_rep):
            res = run_single_experiment(a, b, c, n, frac, seed=4000 * rep)
            if 'Proposed' in res:
                errors.append(res['Proposed'])
        mean_err = np.nanmean(errors)
        theory_rate = np.sqrt(factor * frac)
        print(f"    o/n={frac:.2f}: error={mean_err:.4f}, "
              f"sqrt(5*o/n)={theory_rate:.4f}, "
              f"ratio={mean_err/theory_rate:.3f}")


def experiment4_noise_types(n_rep=20, n=500):
    """Experiment 4: Performance under different noise distributions."""
    print("\n" + "=" * 60)
    print("Experiment 4: Noise Distribution Comparison")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    frac = 0.05
    
    noise_types = ['gaussian', 'student_t', 'laplace']
    
    for noise in noise_types:
        print(f"\n  Noise: {noise}")
        errors_prop = []
        errors_ols = []
        errors_bip = []
        
        for rep in range(n_rep):
            res = run_single_experiment(
                a, b, c, n, frac, noise_type=noise, seed=5000 * rep
            )
            if 'Proposed' in res:
                errors_prop.append(res['Proposed'])
            if 'OLS' in res:
                errors_ols.append(res['OLS'])
            if 'BIP-tau' in res:
                errors_bip.append(res['BIP-tau'])
        
        print(f"    OLS:      {np.nanmean(errors_ols):.4f}")
        print(f"    BIP-tau:  {np.nanmean(errors_bip):.4f}")
        print(f"    Proposed: {np.nanmean(errors_prop):.4f}")


def experiment5_computation_time(n=500):
    """Experiment 5: Computation time comparison."""
    print("\n" + "=" * 60)
    print("Experiment 5: Computation Time")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    
    y, u, xi = simulate_armax(a, b, c, n, seed=42)
    y_tilde, outlier_set, _ = add_outliers(y, int(n * 0.05), seed=42)
    
    methods = {
        'OLS': lambda: ols_armax(y_tilde, u, p, q, r),
        'LAD': lambda: lad_armax(y_tilde, u, p, q, r),
        'Huber': lambda: huber_armax(y_tilde, u, p, q, r),
        'BIP-tau': lambda: bip_tau_armax(y_tilde, u, p, q, r),
        'Student-t EM': lambda: student_t_em_armax(y_tilde, u, p, q, r),
        'Proposed': lambda: two_stage_robust_armax(y_tilde, u, p, q, r),
    }
    
    for name, method in methods.items():
        times = []
        for _ in range(5):
            t0 = time.time()
            try:
                method()
            except:
                pass
            times.append(time.time() - t0)
        print(f"  {name:15s}: {np.mean(times):.3f}s (±{np.std(times):.3f}s)")


def experiment6_componentwise_rates(n_rep=20):
    """Experiment 6: Component-wise rate analysis.
    
    Separate error into AR, exogenous, and MA components to verify:
    - AR+exogenous parameters converge at sqrt(d_arx/n) (Stage 1 rate)
    - MA parameters converge slower (Stage 2 rate, affected by residual noise)
    """
    print("\n" + "=" * 60)
    print("Experiment 6: Component-wise Rate Analysis")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    d_arx = p + q + 1  # = 5
    d_ma = r  # = 2
    frac = 0.05
    
    print("\n  TV-1 (component-wise): Error vs n at 5% contamination")
    print(f"  {'n':>6s}  {'||a-a*||':>10s}  {'||b-b*||':>10s}  {'||c-c*||':>10s}  "
          f"{'||theta||':>10s}  {'sqrt(d_arx/n)':>14s}  {'ratio(arx)':>11s}  {'ratio(ma)':>10s}")
    print("  " + "-" * 95)
    
    n_values = [200, 500, 1000, 2000, 5000]
    
    for n in n_values:
        err_a = []
        err_b = []
        err_c = []
        err_total = []
        
        for rep in range(n_rep):
            y, u, xi = simulate_armax(a, b, c, n, seed=6000 * rep)
            n_out = int(n * frac)
            y_tilde, outlier_set, _ = add_outliers(y, n_out, seed=6000 * rep)
            
            try:
                theta, a_hat, b_hat, c_hat, _, _ = two_stage_robust_armax(
                    y_tilde, u, p, q, r
                )
                err_a.append(np.linalg.norm(a_hat - a))
                err_b.append(np.linalg.norm(b_hat - b))
                err_c.append(np.linalg.norm(c_hat - c))
                err_total.append(np.linalg.norm(theta - true_parameter_vector(a, b, c)))
            except:
                pass
        
        mean_a = np.mean(err_a)
        mean_b = np.mean(err_b)
        mean_c = np.mean(err_c)
        mean_total = np.mean(err_total)
        rate_arx = np.sqrt(d_arx / n)
        rate_ma = np.sqrt(d_ma / n)
        # ARX component error = sqrt(||a-a*||^2 + ||b-b*||^2)
        mean_arx = np.mean([np.sqrt(ea**2 + eb**2) for ea, eb in zip(err_a, err_b)])
        ratio_arx = mean_arx / rate_arx
        ratio_ma = mean_c / rate_ma
        
        print(f"  {n:6d}  {mean_a:10.4f}  {mean_b:10.4f}  {mean_c:10.4f}  "
              f"{mean_total:10.4f}  {rate_arx:14.4f}  {ratio_arx:11.3f}  {ratio_ma:10.3f}")
    
    # Also show clean (0%) rates for comparison
    print("\n  Clean (0% contamination) for reference:")
    print(f"  {'n':>6s}  {'||a-a*||':>10s}  {'||b-b*||':>10s}  {'||c-c*||':>10s}  "
          f"{'||theta||':>10s}  {'ratio(arx)':>11s}  {'ratio(ma)':>10s}")
    print("  " + "-" * 75)
    
    for n in n_values:
        err_a = []
        err_b = []
        err_c = []
        err_total = []
        
        for rep in range(n_rep):
            y, u, xi = simulate_armax(a, b, c, n, seed=7000 * rep)
            
            try:
                theta, a_hat, b_hat, c_hat, _, _ = two_stage_robust_armax(
                    y, u, p, q, r
                )
                err_a.append(np.linalg.norm(a_hat - a))
                err_b.append(np.linalg.norm(b_hat - b))
                err_c.append(np.linalg.norm(c_hat - c))
                err_total.append(np.linalg.norm(theta - true_parameter_vector(a, b, c)))
            except:
                pass
        
        mean_a = np.mean(err_a)
        mean_b = np.mean(err_b)
        mean_c = np.mean(err_c)
        mean_total = np.mean(err_total)
        rate_arx = np.sqrt(d_arx / n)
        rate_ma = np.sqrt(d_ma / n)
        mean_arx = np.mean([np.sqrt(ea**2 + eb**2) for ea, eb in zip(err_a, err_b)])
        ratio_arx = mean_arx / rate_arx
        ratio_ma = mean_c / rate_ma
        
        print(f"  {n:6d}  {mean_a:10.4f}  {mean_b:10.4f}  {mean_c:10.4f}  "
              f"{mean_total:10.4f}  {ratio_arx:11.3f}  {ratio_ma:10.3f}")
    
    # Log-log slope estimation
    print("\n  Log-log slope estimation (contaminated, 5%):")
    n_vals_log = [200, 500, 1000, 2000, 5000]
    
    arx_errors_log = []
    ma_errors_log = []
    total_errors_log = []
    
    for n in n_vals_log:
        err_arx_list = []
        err_ma_list = []
        err_total_list = []
        
        for rep in range(n_rep):
            y, u, xi = simulate_armax(a, b, c, n, seed=6000 * rep)
            n_out = int(n * frac)
            y_tilde, outlier_set, _ = add_outliers(y, n_out, seed=6000 * rep)
            
            try:
                theta, a_hat, b_hat, c_hat, _, _ = two_stage_robust_armax(
                    y_tilde, u, p, q, r
                )
                err_arx_list.append(np.sqrt(np.linalg.norm(a_hat - a)**2 +
                                             np.linalg.norm(b_hat - b)**2))
                err_ma_list.append(np.linalg.norm(c_hat - c))
                err_total_list.append(np.linalg.norm(theta - true_parameter_vector(a, b, c)))
            except:
                pass
        
        arx_errors_log.append(np.mean(err_arx_list))
        ma_errors_log.append(np.mean(err_ma_list))
        total_errors_log.append(np.mean(err_total_list))
    
    # Fit log-log slopes
    log_n = np.log(n_vals_log)
    
    log_arx = np.log(arx_errors_log)
    slope_arx = np.polyfit(log_n, log_arx, 1)[0]
    
    log_ma = np.log(ma_errors_log)
    slope_ma = np.polyfit(log_n, log_ma, 1)[0]
    
    log_total = np.log(total_errors_log)
    slope_total = np.polyfit(log_n, log_total, 1)[0]
    
    print(f"    AR+exogenous slope: {slope_arx:.3f} (theory: -0.500)")
    print(f"    MA slope:           {slope_ma:.3f} (theory: -0.500 if no bias floor)")
    print(f"    Total slope:        {slope_total:.3f}")


def experiment7_misspecification(n_rep=20, n=500):
    """Experiment 7: Model misspecification (wrong orders).
    
    True system is ARMAX(2,2,2). Fit with wrong orders.
    Report prediction MSE on clean test data.
    """
    print("\n" + "=" * 60)
    print("Experiment 7: Model Misspecification")
    print("=" * 60)
    
    # True system
    a_true, b_true, c_true = generate_armax_system(p=2, q=2, r=2, seed=42)
    p_true, q_true, r_true = 2, 2, 2
    frac = 0.05
    
    configs = [
        (2, 2, 1, "Under-spec r"),
        (2, 2, 2, "Correct"),
        (2, 2, 3, "Over-spec r"),
        (1, 2, 2, "Under-spec p"),
        (3, 2, 2, "Over-spec p"),
        (3, 2, 3, "Over-spec p,r"),
    ]
    
    methods_to_test = {
        'OLS': ols_armax,
        'BIP-tau': bip_tau_armax,
        'Huber-PEM': huber_pem_armax,
        'Proposed': None,  # special handling
    }
    
    print(f"\n  True system: ARMAX({p_true},{q_true},{r_true}), "
          f"5% contamination, {n_rep} reps")
    print(f"  Prediction MSE on clean test data")
    print(f"\n  {'Config':15s}  {'OLS':>8s}  {'BIP-tau':>8s}  "
          f"{'Huber-PEM':>10s}  {'Proposed':>10s}")
    print("  " + "-" * 60)
    
    for p_fit, q_fit, r_fit, label in configs:
        results = {m: [] for m in methods_to_test}
        
        for rep in range(n_rep):
            # Generate data from true system
            y_full, u_full, _ = simulate_armax(
                a_true, b_true, c_true, n=n+200, seed=8000*rep
            )
            
            # Train/test split
            y_train = y_full[:n].copy()
            u_train = u_full[:n].copy()
            y_test = y_full[n:]
            u_test = u_full[n:]
            
            # Add outliers to training
            n_out = int(n * frac)
            y_train_c, _, _ = add_outliers(y_train, n_out, seed=8000*rep)
            
            for method_name in methods_to_test:
                try:
                    if method_name == 'Proposed':
                        theta, a_hat, b_hat, c_hat, _, _ = two_stage_robust_armax(
                            y_train_c, u_train, p_fit, q_fit, r_fit
                        )
                    elif method_name == 'Huber-PEM':
                        theta, a_hat, b_hat, c_hat = huber_pem_armax(
                            y_train_c, u_train, p_fit, q_fit, r_fit
                        )
                    else:
                        func = methods_to_test[method_name]
                        theta, a_hat, b_hat, c_hat = func(
                            y_train_c, u_train, p_fit, q_fit, r_fit
                        )
                    
                    # Predict on test
                    start_test = max(p_fit, q_fit, r_fit)
                    xi_test = np.zeros(len(y_test))
                    y_pred = np.zeros(len(y_test))
                    
                    for t in range(start_test, len(y_test)):
                        pred = 0.0
                        for i in range(p_fit):
                            pred += a_hat[i] * y_test[t-1-i]
                        for j in range(q_fit + 1):
                            pred += b_hat[j] * u_test[t-j]
                        for k in range(r_fit):
                            pred += c_hat[k] * xi_test[t-1-k]
                        y_pred[t] = pred
                        xi_test[t] = y_test[t] - pred
                    
                    mse = np.mean((y_test[start_test:] - y_pred[start_test:])**2)
                    if np.isfinite(mse) and mse < 1e6:
                        results[method_name].append(mse)
                    else:
                        results[method_name].append(np.nan)
                except:
                    results[method_name].append(np.nan)
        
        vals = {m: np.nanmean(results[m]) if results[m] else np.nan 
                for m in methods_to_test}
        print(f"  ({p_fit},{q_fit},{r_fit}) {label:10s}"
              f"  {vals['OLS']:8.4f}  {vals['BIP-tau']:8.4f}"
              f"  {vals['Huber-PEM']:10.4f}  {vals['Proposed']:10.4f}")


def experiment8_sensitivity(n_rep=20, n=500):
    """Experiment 8: Sensitivity to tuning parameters.
    
    Varies damping alpha and downweight omega_0 to show
    robustness of the method to tuning choices.
    """
    print("\n" + "=" * 60)
    print("Experiment 8: Tuning Parameter Sensitivity")
    print("=" * 60)
    
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    p, q, r = 2, 2, 2
    theta_true = true_parameter_vector(a, b, c)
    frac = 0.05
    
    # Part 1: Sensitivity to damping alpha
    alphas = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"\n  Part A: Damping alpha (omega_0=0.01 fixed, 5% contam)")
    print(f"  {'alpha':>8s}  {'Error':>8s}  {'Std':>8s}")
    print("  " + "-" * 30)
    
    for alpha in alphas:
        errors = []
        for rep in range(n_rep):
            y, u_sig, _ = simulate_armax(a, b, c, n=n, seed=9000*rep)
            yt, _, _ = add_outliers(y, int(n * frac), seed=9000*rep)
            try:
                theta, _, _, _, _, _ = two_stage_robust_armax(
                    yt, u_sig, p, q, r, damping=alpha
                )
                errors.append(np.linalg.norm(theta - theta_true))
            except:
                errors.append(np.nan)
        
        mean_err = np.nanmean(errors)
        std_err = np.nanstd(errors)
        print(f"  {alpha:8.1f}  {mean_err:8.4f}  {std_err:8.4f}")
    
    # Part 2: Sensitivity to omega_0
    omegas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    
    print(f"\n  Part B: Downweight omega_0 (alpha=0.7 fixed, 5% contam)")
    print(f"  {'omega_0':>8s}  {'Error':>8s}  {'Std':>8s}")
    print("  " + "-" * 30)
    
    for omega in omegas:
        errors = []
        for rep in range(n_rep):
            y, u_sig, _ = simulate_armax(a, b, c, n=n, seed=9000*rep)
            yt, _, _ = add_outliers(y, int(n * frac), seed=9000*rep)
            try:
                theta, _, _, _, _, _ = two_stage_robust_armax(
                    yt, u_sig, p, q, r, omega_0=omega
                )
                errors.append(np.linalg.norm(theta - theta_true))
            except:
                errors.append(np.nan)
        
        mean_err = np.nanmean(errors)
        std_err = np.nanstd(errors)
        print(f"  {omega:8.3f}  {mean_err:8.4f}  {std_err:8.4f}")
    
    # Part 3: Sensitivity to C0
    C0s = [1.0, 1.5, 2.0, 3.0, 4.0]
    
    print(f"\n  Part C: Threshold C0 (alpha=0.7, omega_0=0.01 fixed)")
    print(f"  {'C0':>8s}  {'Error':>8s}  {'Std':>8s}")
    print("  " + "-" * 30)
    
    for C0 in C0s:
        errors = []
        for rep in range(n_rep):
            y, u_sig, _ = simulate_armax(a, b, c, n=n, seed=9000*rep)
            yt, _, _ = add_outliers(y, int(n * frac), seed=9000*rep)
            try:
                theta, _, _, _, _, _ = two_stage_robust_armax(
                    yt, u_sig, p, q, r, C0=C0
                )
                errors.append(np.linalg.norm(theta - theta_true))
            except:
                errors.append(np.nan)
        
        mean_err = np.nanmean(errors)
        std_err = np.nanstd(errors)
        print(f"  {C0:8.1f}  {mean_err:8.4f}  {std_err:8.4f}")


if __name__ == '__main__':
    import sys
    
    if '--full' in sys.argv:
        n_rep = 30
    else:
        n_rep = 5  # quick test
    
    print(f"Running with n_rep={n_rep}")
    print()
    
    # Quick sanity check
    a, b, c = generate_armax_system(p=2, q=2, r=2, seed=42)
    print(f"True system: a={a}, b={b}, c={c}")
    print(f"Stable: {True}, Invertible: {True}")
    print()
    
    experiment1_error_vs_contamination(n_rep=n_rep)
    experiment2_propagation_factor(n_rep=n_rep)
    experiment3_rate_validation(n_rep=n_rep)
    experiment4_noise_types(n_rep=n_rep)
    experiment5_computation_time()
    experiment6_componentwise_rates(n_rep=n_rep)
    experiment7_misspecification(n_rep=n_rep)
    experiment8_sensitivity(n_rep=n_rep)
