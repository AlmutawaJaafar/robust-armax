"""
Baseline methods for ARMAX estimation under outlier contamination.

Baselines:
1. OLS (ordinary least squares, two-stage)
2. LAD (least absolute deviations)
3. Huber regression (non-propagation-aware)
4. ARX-only (Paper 1 method, ignoring MA)
5. BIP-ARMA (Muler/Peña/Yohai style, bounded influence propagation)
6. Student-t EM (expectation-maximization with t-distribution)
7. Huber-PEM (joint robust prediction error minimization)
"""

import numpy as np
from scipy.optimize import minimize
from data_generation import build_armax_regressor


def ols_armax(y, u, p, q, r):
    """Two-stage OLS ARMAX estimation.
    
    Stage 1: OLS on ARX submodel
    Stage 2: OLS on MA from residuals (iterative)
    """
    n = len(y)
    
    # Stage 1: ARX
    Phi, y_resp, start = build_armax_regressor(y, u, p, q, r=0, e=None)
    theta_arx = np.linalg.lstsq(Phi, y_resp, rcond=None)[0]
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    # Residuals
    residuals = np.zeros(n)
    residuals[start:] = y[start:] - Phi @ theta_arx
    
    # Stage 2: MA from residuals (iterative)
    xi_hat = residuals.copy()
    c_hat = np.zeros(r)
    
    for _ in range(5):
        # Build MA regressor
        Phi_ma = np.zeros((n - r, r))
        for t in range(n - r):
            idx = t + r
            for k in range(r):
                Phi_ma[t, k] = xi_hat[idx - 1 - k]
        
        e_resp = residuals[r:]
        c_hat = np.linalg.lstsq(Phi_ma, e_resp, rcond=None)[0]
        
        # Update innovations
        xi_hat = residuals.copy()
        for t in range(r, n):
            xi_hat[t] = residuals[t] - np.dot(c_hat, np.array([xi_hat[t-1-k] for k in range(r)]))
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat


def lad_armax(y, u, p, q, r):
    """Two-stage LAD (L1) ARMAX estimation."""
    n = len(y)
    
    # Stage 1: LAD on ARX
    Phi, y_resp, start = build_armax_regressor(y, u, p, q, r=0, e=None)
    n_eff, d = Phi.shape
    
    # Solve LAD via linear programming
    from scipy.optimize import linprog
    
    # LAD: min sum |y - Phi*theta| = min sum u_i + v_i
    # s.t. Phi*theta + u - v = y, u,v >= 0
    c_lp = np.ones(2 * n_eff)  # minimize sum of u + v
    A_eq = np.hstack([Phi, np.eye(n_eff), -np.eye(n_eff)])
    b_eq = y_resp
    bounds = [(None, None)] * d + [(0, None)] * (2 * n_eff)
    
    try:
        result = linprog(c_lp, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                        method='highs')
        theta_arx = result.x[:d]
    except:
        # Fallback to scipy minimize
        def l1_loss(theta):
            return np.sum(np.abs(y_resp - Phi @ theta))
        result = minimize(l1_loss, np.zeros(d), method='Nelder-Mead',
                         options={'maxiter': 5000})
        theta_arx = result.x
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    # Residuals
    residuals = np.zeros(n)
    residuals[start:] = y[start:] - Phi @ theta_arx
    
    # Stage 2: LAD on MA
    xi_hat = residuals.copy()
    c_hat = np.zeros(r)
    
    for _ in range(5):
        Phi_ma = np.zeros((n - r, r))
        for t in range(n - r):
            idx = t + r
            for k in range(r):
                Phi_ma[t, k] = xi_hat[idx - 1 - k]
        
        e_resp = residuals[r:]
        
        def l1_ma(c):
            return np.sum(np.abs(e_resp - Phi_ma @ c))
        
        result = minimize(l1_ma, c_hat, method='Nelder-Mead',
                         options={'maxiter': 3000})
        c_hat = result.x
        
        xi_hat = residuals.copy()
        for t in range(r, n):
            xi_hat[t] = residuals[t] - np.dot(c_hat, np.array([xi_hat[t-1-k] for k in range(r)]))
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat


def huber_armax(y, u, p, q, r, delta=1.345):
    """Two-stage Huber M-estimation for ARMAX (non-propagation-aware)."""
    n = len(y)
    
    def huber_loss_fn(residuals, delta):
        abs_r = np.abs(residuals)
        return np.sum(np.where(abs_r <= delta, 0.5 * residuals**2,
                               delta * abs_r - 0.5 * delta**2))
    
    # Stage 1: Huber on ARX
    Phi, y_resp, start = build_armax_regressor(y, u, p, q, r=0, e=None)
    n_eff, d = Phi.shape
    
    def loss_arx(theta):
        return huber_loss_fn(y_resp - Phi @ theta, delta) / n_eff
    
    theta_init = np.linalg.lstsq(Phi, y_resp, rcond=None)[0]
    result = minimize(loss_arx, theta_init, method='L-BFGS-B',
                     options={'maxiter': 1000})
    theta_arx = result.x
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    # Residuals
    residuals = np.zeros(n)
    residuals[start:] = y[start:] - Phi @ theta_arx
    
    # Stage 2: Huber on MA
    xi_hat = residuals.copy()
    c_hat = np.zeros(r)
    
    for _ in range(5):
        Phi_ma = np.zeros((n - r, r))
        for t in range(n - r):
            idx = t + r
            for k in range(r):
                Phi_ma[t, k] = xi_hat[idx - 1 - k]
        
        e_resp = residuals[r:]
        
        def loss_ma(c):
            return huber_loss_fn(e_resp - Phi_ma @ c, delta) / len(e_resp)
        
        result = minimize(loss_ma, c_hat, method='L-BFGS-B',
                         options={'maxiter': 500})
        c_hat = result.x
        
        xi_hat = residuals.copy()
        for t in range(r, n):
            xi_hat[t] = residuals[t] - np.dot(c_hat, np.array([xi_hat[t-1-k] for k in range(r)]))
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat


def arx_only(y, u, p, q, r):
    """Paper 1 ARX-only estimator (ignores MA component).
    
    Estimates (a, b) robustly but sets c = 0.
    """
    from proposed_method import stage1_robust_arx
    
    theta_arx, residuals, detected, prop_set = stage1_robust_arx(
        y, u, p, q, n_iter=4, omega_0=0.01, C0=2.0
    )
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    c_hat = np.zeros(r)
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat


def bip_tau_armax(y, u, p, q, r, n_iter=50):
    """BIP-ARMA estimation (Muler/Peña/Yohai 2009, Muma & Zoubir 2016).
    
    Bounded Influence Propagation: replaces innovations with
    bounded versions a_t = psi_k(xi_t / s) * s in both AR and MA
    recursions, preventing outlier propagation through the model.
    Uses tau-estimate of scale for robustness and efficiency.
    
    Key difference from naive Huber-IRLS: the BIP filter truncates
    the INNOVATIONS (not residuals), which prevents cascading
    contamination through the AR/MA recursion.
    """
    n = len(y)
    
    # Initialize with OLS two-stage
    from baseline_methods import ols_armax
    try:
        _, a_init, b_init, c_init = ols_armax(y, u, p, q, r)
    except:
        a_init = np.zeros(p)
        b_init = np.zeros(q + 1)
        c_init = np.zeros(r)
    
    a_hat = a_init.copy()
    b_hat = b_init.copy()
    c_hat = c_init.copy()
    
    start = max(p, q, r)
    huber_k = 1.345  # Huber psi tuning constant
    
    for iteration in range(n_iter):
        # Compute BIP innovations: apply bounded-influence filter
        # e_t = y_t - sum a_i y_{t-i} - sum b_j u_{t-j}
        # xi_t^BIP = psi_k(e_t - sum c_k xi_{t-k}^BIP) / s) * s
        xi_bip = np.zeros(n)
        e_raw = np.zeros(n)
        
        for t in range(start, n):
            # AR + exogenous prediction
            pred = 0.0
            for i in range(p):
                pred += a_hat[i] * y[t - 1 - i]
            for j in range(q + 1):
                pred += b_hat[j] * u[t - j]
            # MA prediction from BIP innovations
            for k in range(r):
                if t - 1 - k >= 0:
                    pred += c_hat[k] * xi_bip[t - 1 - k]
            
            # Overflow protection
            if not np.isfinite(pred):
                pred = 0.0
            
            e_raw[t] = y[t] - pred
            xi_bip[t] = e_raw[t]
        
        # Tau scale of BIP innovations
        s_hat = compute_tau_scale(xi_bip[start:])
        if s_hat < 1e-10:
            s_hat = 1.0
        
        # Apply bounded-influence psi to innovations
        for t in range(start, n):
            pred = 0.0
            for i in range(p):
                pred += a_hat[i] * y[t - 1 - i]
            for j in range(q + 1):
                pred += b_hat[j] * u[t - j]
            for k in range(r):
                if t - 1 - k >= 0:
                    pred += c_hat[k] * xi_bip[t - 1 - k]
            
            raw_innov = y[t] - pred
            # Huber psi: clip to [-k*s, k*s]
            xi_bip[t] = np.clip(raw_innov, -huber_k * s_hat, huber_k * s_hat)
        
        # Now estimate parameters using the BIP innovations
        # Build regressor with BIP innovations for MA part
        n_eff = n - start
        d_total = p + q + 1 + r
        Phi_bip = np.zeros((n_eff, d_total))
        y_bip = np.zeros(n_eff)
        
        for t in range(n_eff):
            idx = t + start
            # AR lags
            for i in range(p):
                Phi_bip[t, i] = y[idx - 1 - i]
            # Exogenous lags
            for j in range(q + 1):
                Phi_bip[t, p + j] = u[idx - j]
            # MA lags (BIP innovations)
            for k in range(r):
                if idx - 1 - k >= 0:
                    Phi_bip[t, p + q + 1 + k] = xi_bip[idx - 1 - k]
            
            y_bip[t] = y[idx]
        
        # Weighted least squares with Tukey biweight on residuals
        residuals_bip = y_bip - Phi_bip @ np.concatenate([a_hat, b_hat, c_hat])
        s_res = compute_tau_scale(residuals_bip)
        if s_res < 1e-10:
            s_res = s_hat
        
        u_vals = residuals_bip / s_res
        weights = tukey_biweight(u_vals, c=4.685)
        
        W = np.diag(weights)
        try:
            theta_new = np.linalg.solve(
                Phi_bip.T @ W @ Phi_bip + 1e-6 * np.eye(d_total),
                Phi_bip.T @ (W @ y_bip)
            )
        except np.linalg.LinAlgError:
            break
        
        a_new = theta_new[:p]
        b_new = theta_new[p:p+q+1]
        c_new = theta_new[p+q+1:]
        
        # Stability check: reject update if parameters explode
        if (not np.all(np.isfinite(theta_new)) or
            np.linalg.norm(theta_new) > 100):
            break
        
        # Check convergence
        change = (np.linalg.norm(a_new - a_hat) +
                  np.linalg.norm(b_new - b_hat) +
                  np.linalg.norm(c_new - c_hat))
        
        a_hat, b_hat, c_hat = a_new, b_new, c_new
        
        if change < 1e-6:
            break
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    
    # Check for divergence
    if not np.all(np.isfinite(theta)) or np.linalg.norm(theta) > 50:
        # BIP diverged — return NaN
        return (np.full(p+q+1+r, np.nan), 
                np.full(p, np.nan), np.full(q+1, np.nan), np.full(r, np.nan))
    
    return theta, a_hat, b_hat, c_hat


def student_t_em_armax(y, u, p, q, r, df_init=5, max_iter=100, tol=1e-5):
    """Student-t EM for ARMAX (two-stage).
    
    Models noise as Student-t distribution and uses EM to estimate
    parameters and detect outliers implicitly.
    """
    n = len(y)
    
    # Stage 1: Student-t EM for ARX
    Phi, y_resp, start = build_armax_regressor(y, u, p, q, r=0, e=None)
    n_eff, d = Phi.shape
    
    # Initialize
    theta_arx = np.linalg.lstsq(Phi, y_resp, rcond=None)[0]
    residuals = y_resp - Phi @ theta_arx
    sigma2 = np.var(residuals)
    nu = float(df_init)
    
    for iteration in range(max_iter):
        # E-step: compute weights
        residuals = y_resp - Phi @ theta_arx
        tau = (nu + 1) / (nu + residuals**2 / sigma2)  # posterior weights
        
        # M-step: weighted least squares
        W = np.diag(tau)
        try:
            theta_new = np.linalg.solve(Phi.T @ W @ Phi + 1e-8 * np.eye(d),
                                         Phi.T @ W @ y_resp)
        except np.linalg.LinAlgError:
            break
        
        # Update scale
        residuals_new = y_resp - Phi @ theta_new
        sigma2_new = np.mean(tau * residuals_new**2)
        
        # Check convergence
        if (np.linalg.norm(theta_new - theta_arx) < tol and
            abs(sigma2_new - sigma2) < tol):
            theta_arx = theta_new
            sigma2 = sigma2_new
            break
        
        theta_arx = theta_new
        sigma2 = max(sigma2_new, 1e-10)
    
    a_hat = theta_arx[:p]
    b_hat = theta_arx[p:p+q+1]
    
    # Residuals
    full_residuals = np.zeros(n)
    full_residuals[start:] = y[start:] - Phi @ theta_arx
    
    # Stage 2: Student-t EM for MA
    xi_hat = full_residuals.copy()
    c_hat = np.zeros(r)
    
    for outer in range(5):
        Phi_ma = np.zeros((n - r, r))
        for t in range(n - r):
            idx = t + r
            for k in range(r):
                Phi_ma[t, k] = xi_hat[idx - 1 - k]
        
        e_resp = full_residuals[r:]
        sigma2_ma = np.var(e_resp)
        
        for iteration in range(max_iter):
            residuals_ma = e_resp - Phi_ma @ c_hat
            tau = (nu + 1) / (nu + residuals_ma**2 / sigma2_ma)
            
            W = np.diag(tau)
            try:
                c_new = np.linalg.solve(Phi_ma.T @ W @ Phi_ma + 1e-8 * np.eye(r),
                                         Phi_ma.T @ W @ e_resp)
            except np.linalg.LinAlgError:
                break
            
            residuals_new = e_resp - Phi_ma @ c_new
            sigma2_ma = max(np.mean(tau * residuals_new**2), 1e-10)
            
            if np.linalg.norm(c_new - c_hat) < tol:
                c_hat = c_new
                break
            c_hat = c_new
        
        xi_hat = full_residuals.copy()
        for t in range(r, n):
            xi_hat[t] = full_residuals[t] - np.dot(c_hat, np.array([xi_hat[t-1-k] for k in range(r)]))
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat


# ============================================================
# Helper functions
# ============================================================

def compute_tau_scale(residuals):
    """Tau scale estimate (Maronna & Zamar)."""
    n = len(residuals)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) / 0.6745
    if mad < 1e-10:
        return 1e-10
    
    u = (residuals - med) / (mad * 4.685)
    w = np.where(np.abs(u) <= 1, (1 - u**2)**2, 0)
    
    s2 = mad**2 * np.sum(w * ((residuals - med) / mad)**2) / np.sum(w)
    return np.sqrt(max(s2, 1e-20))


def tukey_biweight(u, c=4.685):
    """Tukey's biweight weight function."""
    return np.where(np.abs(u) <= c, (1 - (u / c)**2)**2, 0)


def huber_pem_armax(y, u, p, q, r, delta=1.345, max_iter=200):
    """Joint Robust PEM: minimize sum H_delta(e_t(theta)) over all
    ARMAX parameters theta = (a, b, c) simultaneously.
    
    This is the natural non-two-stage robust competitor.
    PEM computes innovations recursively:
        e_t(theta) = y_t - sum a_i y_{t-i} - sum b_j u_{t-j}
                     - sum c_k e_{t-k}(theta)
    and minimizes sum H_delta(e_t(theta)).
    
    Key difference from two-stage: no bias floor from separate
    estimation, but no propagation awareness either (outliers
    propagate freely through the e_t recursion).
    """
    n = len(y)
    start = max(p, q, r)
    
    def compute_innovations(theta, y, u, p, q, r):
        """Recursively compute innovations e_t(theta)."""
        a = theta[:p]
        b = theta[p:p+q+1]
        c = theta[p+q+1:p+q+1+r]
        
        e = np.zeros(n)
        for t in range(start, n):
            pred = 0.0
            for i in range(p):
                pred += a[i] * y[t - 1 - i]
            for j in range(q + 1):
                pred += b[j] * u[t - j]
            for k in range(r):
                if t - 1 - k >= start:
                    pred += c[k] * e[t - 1 - k]
            if not np.isfinite(pred):
                pred = 0.0
            e[t] = np.clip(y[t] - pred, -1e10, 1e10)
        return e[start:]
    
    def huber_pem_loss(theta):
        """Huber PEM objective."""
        e = compute_innovations(theta, y, u, p, q, r)
        abs_e = np.abs(e)
        loss = np.sum(np.where(abs_e <= delta,
                               0.5 * e**2,
                               delta * abs_e - 0.5 * delta**2))
        return loss / len(e)
    
    def huber_pem_grad(theta):
        """Numerical gradient of Huber PEM (finite differences)."""
        d = len(theta)
        grad = np.zeros(d)
        eps = 1e-6
        f0 = huber_pem_loss(theta)
        for i in range(d):
            theta_p = theta.copy()
            theta_p[i] += eps
            grad[i] = (huber_pem_loss(theta_p) - f0) / eps
        return grad
    
    # Initialize with OLS two-stage
    try:
        _, a_init, b_init, c_init = ols_armax(y, u, p, q, r)
        theta_init = np.concatenate([a_init, b_init, c_init])
    except:
        theta_init = np.zeros(p + q + 1 + r)
    
    # Also try standard PEM (quadratic) initialization
    def pem_loss(theta):
        e = compute_innovations(theta, y, u, p, q, r)
        return np.mean(e**2)
    
    try:
        result_pem = minimize(pem_loss, theta_init, method='L-BFGS-B',
                              options={'maxiter': 100})
        if result_pem.fun < pem_loss(theta_init):
            theta_init = result_pem.x
    except:
        pass
    
    # Optimize Huber PEM
    try:
        result = minimize(huber_pem_loss, theta_init, 
                         jac=huber_pem_grad,
                         method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-10})
        theta_hat = result.x
    except:
        # Fallback to Nelder-Mead
        try:
            result = minimize(huber_pem_loss, theta_init,
                             method='Nelder-Mead',
                             options={'maxiter': max_iter * 10})
            theta_hat = result.x
        except:
            theta_hat = theta_init
    
    a_hat = theta_hat[:p]
    b_hat = theta_hat[p:p+q+1]
    c_hat = theta_hat[p+q+1:p+q+1+r]
    
    theta = np.concatenate([a_hat, b_hat, c_hat])
    return theta, a_hat, b_hat, c_hat
