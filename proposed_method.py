"""
Two-Stage Robust ARMAX Estimator (v2 — tuning fixes).

Fixes over v1:
1. Adaptive detection threshold c_out = C_0 * (n / (d * log(d)))^{1/4}
   from Paper 1, replacing the fixed c_out=3.0 that fails at high
   contamination.
2. Damped MA recursion in Stage 2 to prevent oscillation:
   c_new = alpha * c_candidate + (1-alpha) * c_old.
3. Stage 2 uses clean-only residuals (detected outlier residuals are
   replaced by interpolated values before MA fitting) to prevent
   outlier energy from leaking into the innovation estimates.

The total propagation factor is (p + r + 1).
"""

import numpy as np
from scipy.optimize import minimize
from data_generation import (build_armax_regressor, get_propagation_set,
                              check_stability, check_invertibility)


# ============================================================
# Core building blocks
# ============================================================

def huber_loss(r, delta):
    """Huber loss."""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * abs_r - 0.5 * delta**2)


def huber_derivative(r, delta):
    """Huber psi function."""
    return np.clip(r, -delta, delta)


def threshold_regressor(Phi, tau):
    """Row-wise norm thresholding."""
    norms = np.linalg.norm(Phi, axis=1)
    scale = np.minimum(1.0, tau / np.maximum(norms, 1e-12))
    return Phi * scale[:, np.newaxis]


def compute_mad(residuals, trim_fraction=0.0):
    """Trimmed MAD scale estimate."""
    if trim_fraction > 0:
        n_keep = max(int(len(residuals) * (1 - trim_fraction)), 3)
        idx = np.argsort(np.abs(residuals))[:n_keep]
        residuals = residuals[idx]
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return max(mad / 0.6745, 1e-8)


def adaptive_threshold(n, d, delta=0.05, C0=2.0):
    """Adaptive detection threshold from Paper 1.
    
    c_out = C0 * (n / (d * log(d/delta)))^{1/4}
    
    This grows with n, ensuring:
    - False positives decrease (threshold grows)
    - True outliers (magnitude sqrt(n)) are still detected
      (sqrt(n) >> n^{1/4} for large n)
    """
    log_term = max(np.log(max(d, 2) / delta), 1.0)
    return C0 * (n / (d * log_term)) ** 0.25


def detect_outliers_adaptive(residuals, s_hat, n, d, p=0, r=0,
                              C0=2.0, delta=0.05):
    """Detect outliers with adaptive threshold + propagation.
    
    At contamination, direct outlier residuals are O(sqrt(n)),
    clean residuals are O(n^{1/4}) at worst.
    The adaptive threshold c_out = O(n^{1/4}) separates them.
    """
    c_out = adaptive_threshold(n, d, delta=delta, C0=C0)
    # But never below 3 (practical minimum)
    c_out = max(c_out, 3.0)
    
    threshold = c_out * s_hat
    detected = np.where(np.abs(residuals) > threshold)[0]
    
    # Propagation
    propagated = set()
    for t in detected:
        for dt in range(0, p + r + 1):
            if t + dt < len(residuals):
                propagated.add(t + dt)
    
    return detected, np.sort(list(propagated)), c_out


def irls_huber(Phi, y, delta, sample_weights, theta_init=None,
               max_iter=50, tol=1e-6):
    """IRLS for weighted Huber regression.
    
    Solves: min sum w_t * H_delta(y_t - phi_t^T theta)
    
    Parameters
    ----------
    delta : Huber threshold (absolute, not scaled)
    """
    n, d = Phi.shape
    
    if theta_init is not None:
        theta = theta_init.copy()
    else:
        # Weighted LS initialization
        W0 = np.diag(sample_weights)
        try:
            theta = np.linalg.solve(
                Phi.T @ W0 @ Phi + 1e-8 * np.eye(d),
                Phi.T @ (W0 @ y)
            )
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    
    for iteration in range(max_iter):
        residuals = y - Phi @ theta
        abs_r = np.abs(residuals)
        
        # Huber weights: min(1, delta/|r|)
        huber_w = np.where(abs_r > 1e-10,
                           np.minimum(1.0, delta / abs_r),
                           1.0)
        
        w_total = sample_weights * huber_w
        
        # Weighted LS
        PhiTW = Phi.T * w_total[np.newaxis, :]  # d x n
        try:
            theta_new = np.linalg.solve(
                PhiTW @ Phi + 1e-8 * np.eye(d),
                PhiTW @ y
            )
        except np.linalg.LinAlgError:
            break
        
        if np.linalg.norm(theta_new - theta) / (1 + np.linalg.norm(theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    
    return theta


# ============================================================
# Stage 1: Robust ARX estimation
# ============================================================

def stage1_robust_arx(y_tilde, u, p, q, n_iter=4, omega_0=0.01,
                       C0=2.0, verbose=False):
    """Stage 1: Robust ARX estimation (Algorithm II from Paper 1).
    
    Key improvements over v1:
    - Adaptive threshold c_out = C0 * (n/(d*log(d)))^{1/4}
    - Propagation in Stage 1 uses factor (p+1) only (AR propagation)
    - Residuals computed on UN-thresholded regressors for Stage 2
    """
    n = len(y_tilde)
    d_arx = p + q + 1
    
    # Build ARX regressor
    Phi, y_resp, start = build_armax_regressor(y_tilde, u, p, q, r=0, e=None)
    n_eff = len(y_resp)
    
    # Thresholding
    tau_phi = (n_eff / max(np.log(max(d_arx, 2)), 1.0)) ** 0.25
    Phi_thresh = threshold_regressor(Phi, tau_phi)
    
    # Huber threshold: delta = C * sigma * sqrt(n)
    # Use a moderate value that doesn't suppress too much
    sigma_init = compute_mad(y_resp)
    huber_delta = max(1.345 * sigma_init, 1.0)  # classic Huber tuning
    
    # Initialize with weighted LS (downweighting large residuals)
    theta = np.linalg.lstsq(Phi_thresh, y_resp, rcond=None)[0]
    weights = np.ones(n_eff)
    detected = np.array([], dtype=int)
    prop_set_local = np.array([], dtype=int)
    
    for k in range(n_iter):
        # Compute residuals
        residuals = y_resp - Phi_thresh @ theta
        
        # Scale estimate (trimmed MAD)
        n_downweighted = np.sum(weights < 0.5)
        trim_frac = min(0.4, 1.5 * n_downweighted / n_eff) if n_downweighted > 0 else 0.0
        s_hat = compute_mad(residuals, trim_fraction=trim_frac)
        
        # Adaptive detection
        detected, prop_set_local, c_out_used = detect_outliers_adaptive(
            residuals, s_hat, n_eff, d_arx, p=p, r=0, C0=C0
        )
        
        # Update weights: propagation factor (p+1) for Stage 1
        weights = np.ones(n_eff)
        for t in prop_set_local:
            if t < n_eff:
                weights[t] = omega_0
        
        # Update Huber delta based on current scale
        huber_delta = max(1.345 * s_hat, 0.5)
        
        # Solve weighted Huber regression
        theta = irls_huber(Phi_thresh, y_resp, huber_delta, weights,
                           theta_init=theta)
        
        if verbose:
            print(f"  Stage 1 iter {k+1}: detected {len(detected)}, "
                  f"propagated {len(prop_set_local)}, c_out={c_out_used:.1f}, "
                  f"s_hat={s_hat:.3f}, ||theta||={np.linalg.norm(theta):.4f}")
    
    # Final residuals on ORIGINAL (unthresholded) regressors
    full_residuals = np.zeros(n)
    full_residuals[start:] = y_tilde[start:] - Phi @ theta
    
    # Map detected indices back to original time indices
    detected_orig = detected + start
    prop_set_orig = prop_set_local + start
    
    return theta, full_residuals, detected_orig, prop_set_orig


# ============================================================
# Stage 2: Robust MA estimation from residuals
# ============================================================

def clean_residuals(residuals, detected_set, p, r, n):
    """Replace outlier-contaminated residuals with interpolated values.
    
    This prevents outlier energy from leaking into the MA recursion.
    For each contaminated index, replace with the median of nearby
    clean residuals.
    """
    clean = residuals.copy()
    
    # Full propagation set
    contam = set()
    for t in detected_set:
        for dt in range(0, p + r + 1):
            if t + dt < n:
                contam.add(t + dt)
    
    # Replace contaminated residuals with local median of clean ones
    clean_indices = [i for i in range(n) if i not in contam]
    if len(clean_indices) == 0:
        return clean
    
    clean_median = np.median(residuals[clean_indices])
    
    for t in contam:
        # Find nearest clean neighbors
        neighbors = []
        for offset in range(-10, 11):
            idx = t + offset
            if 0 <= idx < n and idx not in contam:
                neighbors.append(residuals[idx])
        
        if len(neighbors) >= 3:
            clean[t] = np.median(neighbors)
        else:
            clean[t] = clean_median
    
    return clean


def stage2_robust_ma(residuals, r, detected_set, p, n_iter=3,
                      omega_0=0.01, C0=2.0, damping=0.7, verbose=False):
    """Stage 2: Robust MA estimation from Stage 1 residuals.
    
    Key improvements over v1:
    1. Clean residuals before MA recursion (replace outlier residuals
       with interpolated values)
    2. Damped updates: c_new = alpha*c_candidate + (1-alpha)*c_old
    3. Adaptive threshold for MA outlier detection
    4. Proper convergence monitoring
    """
    n = len(residuals)
    
    # Step 1: Clean the residuals
    resid_clean = clean_residuals(residuals, detected_set, p, r, n)
    
    # Initialize
    xi_hat = resid_clean.copy()
    c_hat = np.zeros(r)
    c_prev = np.zeros(r)
    
    best_c = np.zeros(r)
    best_loss = np.inf
    
    for outer in range(8):  # more iterations with damping
        # Build MA regressor from current innovations
        start = r
        n_eff = n - start
        Phi_ma = np.zeros((n_eff, r))
        for t in range(n_eff):
            idx = t + start
            for k in range(r):
                Phi_ma[t, k] = xi_hat[idx - 1 - k]
        
        e_resp = resid_clean[start:]
        
        # Thresholding
        tau_phi = (n_eff / max(np.log(max(r, 2)), 1.0)) ** 0.25
        Phi_ma_thresh = threshold_regressor(Phi_ma, tau_phi)
        
        # Compute propagation set in MA domain
        prop_set_ma = set()
        for t in detected_set:
            for dt in range(0, p + r + 1):
                idx = t + dt - start
                if 0 <= idx < n_eff:
                    prop_set_ma.add(idx)
        
        # Weights
        weights = np.ones(n_eff)
        for t in prop_set_ma:
            weights[t] = omega_0
        
        # Scale estimate
        ma_resid = e_resp - Phi_ma_thresh @ c_hat
        trim_frac = min(0.4, 1.5 * len(prop_set_ma) / n_eff) if len(prop_set_ma) > 0 else 0.0
        s_hat = compute_mad(ma_resid, trim_fraction=trim_frac)
        huber_delta = max(1.345 * s_hat, 0.5)
        
        # Solve robust regression for c
        c_candidate = irls_huber(Phi_ma_thresh, e_resp, huber_delta, weights,
                                 theta_init=c_hat)
        
        # Damped update
        c_hat = damping * c_candidate + (1 - damping) * c_hat
        
        # Update innovations using the cleaned residuals
        xi_hat = resid_clean.copy()
        for t in range(r, n):
            ma_val = 0.0
            for k in range(r):
                ma_val += c_hat[k] * xi_hat[t - 1 - k]
            xi_hat[t] = resid_clean[t] - ma_val
        
        # Track best solution
        current_loss = np.sum(huber_loss(e_resp - Phi_ma_thresh @ c_hat, huber_delta))
        if current_loss < best_loss:
            best_loss = current_loss
            best_c = c_hat.copy()
        
        if verbose:
            print(f"  Stage 2 iter {outer+1}: c_hat={np.round(c_hat, 4)}, "
                  f"loss={current_loss:.2f}, s_hat={s_hat:.3f}")
        
        # Check convergence
        if outer > 0 and np.linalg.norm(c_hat - c_prev) < 1e-4:
            break
        c_prev = c_hat.copy()
    
    # Use best solution found
    c_hat = best_c
    
    # Final innovation computation on ORIGINAL residuals
    xi_hat = residuals.copy()
    for t in range(r, n):
        ma_val = 0.0
        for k in range(r):
            ma_val += c_hat[k] * xi_hat[t - 1 - k]
        xi_hat[t] = residuals[t] - ma_val
    
    return c_hat, xi_hat


# ============================================================
# Full two-stage estimator
# ============================================================

def two_stage_robust_armax(y_tilde, u, p, q, r,
                            n_iter_stage1=4, n_iter_stage2=3,
                            omega_0=0.01, C0=2.0, damping=0.7,
                            verbose=False):
    """Two-Stage Robust ARMAX Estimator.
    
    Stage 1: Robust ARX estimation (propagation-aware Huber, adaptive threshold)
    Stage 2: Robust MA estimation (damped iteration, cleaned residuals)
    """
    if verbose:
        print("=== Stage 1: Robust ARX ===")
    
    # Stage 1
    theta_arx, residuals, detected, prop_set = stage1_robust_arx(
        y_tilde, u, p, q,
        n_iter=n_iter_stage1, omega_0=omega_0, C0=C0,
        verbose=verbose
    )
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    if verbose:
        print(f"\n=== Stage 2: Robust MA ===")
    
    # Stage 2
    c_hat, xi_hat = stage2_robust_ma(
        residuals, r, detected, p,
        n_iter=n_iter_stage2, omega_0=omega_0, C0=C0,
        damping=damping, verbose=verbose
    )
    
    # Full parameter vector
    theta = np.concatenate([a_hat, b_hat, c_hat])
    
    info = {
        'theta_arx': theta_arx,
        'residuals_stage1': residuals,
        'detected_outliers': detected,
        'propagation_set': prop_set,
        'xi_hat': xi_hat,
        'n_detected': len(detected),
        'n_propagated': len(prop_set),
    }
    
    return theta, a_hat, b_hat, c_hat, xi_hat, info


# ============================================================
# Oracle estimator (knows true outlier set)
# ============================================================

def oracle_robust_armax(y_tilde, u, p, q, r, outlier_set,
                         omega_0=0.01, verbose=False):
    """Oracle estimator: knows the true outlier set."""
    n = len(y_tilde)
    d_arx = p + q + 1
    
    # Build ARX regressor
    Phi, y_resp, start = build_armax_regressor(y_tilde, u, p, q, r=0, e=None)
    n_eff = len(y_resp)
    
    # Thresholding
    tau_phi = (n_eff / max(np.log(max(d_arx, 2)), 1.0)) ** 0.25
    Phi_thresh = threshold_regressor(Phi, tau_phi)
    
    # Oracle propagation set (uses p+r+1 for full ARMAX propagation)
    prop_set = get_propagation_set(outlier_set, p, r, n)
    
    # Oracle weights
    weights = np.ones(n_eff)
    for t in prop_set:
        t_local = t - start
        if 0 <= t_local < n_eff:
            weights[t_local] = omega_0
    
    # Huber tuning
    sigma_hat = compute_mad(y_resp)
    huber_delta = max(1.345 * sigma_hat, 0.5)
    
    # Stage 1: Oracle ARX
    theta_arx = irls_huber(Phi_thresh, y_resp, huber_delta, weights)
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    # Residuals on original regressors
    residuals = np.zeros(n)
    residuals[start:] = y_tilde[start:] - Phi @ theta_arx
    
    # Stage 2: Oracle MA (uses true outlier set for cleaning)
    c_hat, xi_hat = stage2_robust_ma(
        residuals, r, outlier_set, p,
        n_iter=3, omega_0=omega_0, verbose=verbose
    )
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat
