"""
Data generation for robust ARMAX identification experiments.
ARMAX model: y_t = sum_i a_i y_{t-i} + sum_j b_j u_{t-j} + sum_k c_k xi_{t-k} + xi_t
with adversarial output outliers: ytilde_t = y_t + sqrt(n) * o_t for t in O.
"""

import numpy as np
from scipy.signal import lfilter


def generate_armax_system(p=2, q=2, r=2, seed=None):
    """Generate a stable ARMAX(p,q,r) system.
    
    Returns
    -------
    a : array (p,) — AR coefficients
    b : array (q+1,) — exogenous coefficients (includes b_0)
    c : array (r,) — MA coefficients (c_1,...,c_r; c_0=1 implicit)
    """
    rng = np.random.RandomState(seed)
    
    # Generate stable AR polynomial: place roots outside unit circle
    if p == 2:
        a = np.array([0.5, -0.3])  # roots at ~0.7 magnitude
    elif p == 3:
        a = np.array([0.6, -0.3, 0.1])
    elif p == 1:
        a = np.array([0.7])
    else:
        # Random stable AR
        roots = (1.2 + 0.5 * rng.rand(p)) * np.exp(1j * 2 * np.pi * rng.rand(p))
        roots = np.concatenate([roots[:p//2], np.conj(roots[:p//2])])
        if p % 2 == 1:
            roots = np.append(roots, 1.2 + 0.5 * rng.rand())
        poly = np.real(np.poly(roots[:p]))
        a = -poly[1:p+1]
    
    # Exogenous coefficients
    if q == 2:
        b = np.array([1.0, 0.5, -0.3])
    elif q == 1:
        b = np.array([1.0, 0.4])
    elif q == 0:
        b = np.array([1.0])
    else:
        b = rng.randn(q + 1) * 0.5
        b[0] = 1.0
    
    # MA coefficients (invertible: roots of 1 + c_1 z + ... outside unit circle)
    if r == 2:
        c = np.array([0.4, -0.2])
    elif r == 1:
        c = np.array([0.3])
    elif r == 3:
        c = np.array([0.4, -0.2, 0.1])
    else:
        c = rng.randn(r) * 0.3
    
    return a, b, c


def check_stability(a):
    """Check AR polynomial stability (all roots outside unit circle).
    A(z) = 1 - a_1 z^{-1} - ... - a_p z^{-p}. Roots of z^p - a_1 z^{p-1} - ... - a_p.
    """
    poly = np.concatenate(([1.0], -a))
    roots = np.roots(poly)
    return np.all(np.abs(roots) < 1.0)  # AR roots inside unit circle = stable


def check_invertibility(c):
    """Check MA polynomial invertibility (all roots outside unit circle).
    C(z) = 1 + c_1 z^{-1} + ... + c_r z^{-r}. Roots of z^r + c_1 z^{r-1} + ... + c_r.
    """
    poly = np.concatenate(([1.0], c))
    roots = np.roots(poly)
    return np.all(np.abs(roots) < 1.0)  # MA roots inside unit circle = invertible


def generate_noise(n, noise_type='gaussian', df=5, seed=None):
    """Generate innovation noise xi_t.
    
    Parameters
    ----------
    noise_type : 'gaussian', 'student_t', 'laplace', 'contaminated_gaussian'
    df : degrees of freedom (for student_t)
    """
    rng = np.random.RandomState(seed)
    
    if noise_type == 'gaussian':
        xi = rng.randn(n)
    elif noise_type == 'student_t':
        xi = rng.standard_t(df, size=n)
        xi = xi / np.sqrt(df / (df - 2))  # normalize to unit variance
    elif noise_type == 'laplace':
        xi = rng.laplace(0, 1.0 / np.sqrt(2), size=n)
    elif noise_type == 'contaminated_gaussian':
        xi = rng.randn(n)
        contam = rng.rand(n) < 0.05
        xi[contam] = xi[contam] * 5
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return xi


def simulate_armax(a, b, c, n, u=None, noise_type='gaussian', df=5,
                   sigma=1.0, seed=None, burn_in=200):
    """Simulate ARMAX(p,q,r) process.
    
    y_t = sum_{i=1}^p a_i y_{t-i} + sum_{j=0}^q b_j u_{t-j}
          + sum_{k=1}^r c_k xi_{t-k} + xi_t
    
    Parameters
    ----------
    a : AR coefficients (p,)
    b : exogenous coefficients (q+1,) including b_0
    c : MA coefficients (r,)
    n : sample size
    u : input signal; if None, PRBS is generated
    sigma : noise standard deviation
    
    Returns
    -------
    y : output (n,)
    u : input (n,)
    xi : innovations (n,)
    """
    rng = np.random.RandomState(seed)
    p = len(a)
    q = len(b) - 1
    r = len(c)
    n_total = n + burn_in
    
    # Generate input
    if u is None:
        u_total = rng.choice([-1.0, 1.0], size=n_total)
    else:
        u_total = np.concatenate([rng.choice([-1.0, 1.0], size=burn_in), u])
    
    # Generate noise
    xi = generate_noise(n_total, noise_type, df, seed=seed)
    xi = sigma * xi
    
    # Simulate using direct recursion
    y = np.zeros(n_total)
    for t in range(max(p, q, r), n_total):
        # AR part
        ar_part = 0.0
        for i in range(p):
            ar_part += a[i] * y[t - 1 - i]
        # Exogenous part
        ex_part = 0.0
        for j in range(q + 1):
            ex_part += b[j] * u_total[t - j]
        # MA part
        ma_part = 0.0
        for k in range(r):
            ma_part += c[k] * xi[t - 1 - k]
        # Innovation
        y[t] = ar_part + ex_part + ma_part + xi[t]
    
    # Remove burn-in
    y = y[burn_in:]
    u_out = u_total[burn_in:]
    xi_out = xi[burn_in:]
    
    return y, u_out, xi_out


def add_outliers(y, n_outliers, outlier_magnitude=5.0, seed=None):
    """Add adversarial output outliers: ytilde_t = y_t + sqrt(n) * o_t.
    
    Parameters
    ----------
    y : clean output (n,)
    n_outliers : number of outliers (o = |O|)
    outlier_magnitude : |o_t| >= delta_min, |o_t| <= M
    
    Returns
    -------
    y_tilde : contaminated output
    outlier_set : indices of contaminated samples (O)
    outlier_values : o_t values
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    sqrt_n = np.sqrt(n)
    
    # Select outlier locations (adversarial: spread out)
    # Ensure outliers are at least p+r+1 apart for worst-case propagation
    min_gap = 5  # minimum gap between outliers
    candidates = np.arange(10, n - 10)  # avoid edges
    
    outlier_set = []
    available = set(candidates)
    for _ in range(n_outliers):
        if not available:
            break
        t = rng.choice(list(available))
        outlier_set.append(t)
        # Remove nearby indices
        for dt in range(-min_gap, min_gap + 1):
            available.discard(t + dt)
    
    outlier_set = np.sort(outlier_set)
    
    # Generate outlier values: random sign, magnitude in [delta_min, M]
    signs = rng.choice([-1.0, 1.0], size=len(outlier_set))
    magnitudes = outlier_magnitude * np.ones(len(outlier_set))
    outlier_values = signs * magnitudes
    
    # Add outliers
    y_tilde = y.copy()
    y_tilde[outlier_set] += sqrt_n * outlier_values
    
    return y_tilde, outlier_set, outlier_values


def build_armax_regressor(y, u, p, q, r, e=None):
    """Build the ARMAX regression matrix.
    
    For extended least squares:
    phi_t = [y_{t-1},...,y_{t-p}, u_t,...,u_{t-q}, e_{t-1},...,e_{t-r}]
    
    If e is None (Stage 1), build ARX regressor only:
    phi_t = [y_{t-1},...,y_{t-p}, u_t,...,u_{t-q}]
    
    Parameters
    ----------
    y : output (n,)
    u : input (n,)
    p, q, r : model orders
    e : residuals for MA part (n,); None for ARX-only
    
    Returns
    -------
    Phi : regressor matrix (n_eff, d)
    y_out : response vector (n_eff,)
    start_idx : starting index in original arrays
    """
    n = len(y)
    start = max(p, q, r)
    n_eff = n - start
    
    if e is None:
        # ARX regressor only
        d = p + q + 1
        Phi = np.zeros((n_eff, d))
        for t in range(n_eff):
            idx = t + start
            # AR part
            for i in range(p):
                Phi[t, i] = y[idx - 1 - i]
            # Exogenous part
            for j in range(q + 1):
                Phi[t, p + j] = u[idx - j]
    else:
        # Full ARMAX regressor
        d = p + q + 1 + r
        Phi = np.zeros((n_eff, d))
        for t in range(n_eff):
            idx = t + start
            # AR part
            for i in range(p):
                Phi[t, i] = y[idx - 1 - i]
            # Exogenous part
            for j in range(q + 1):
                Phi[t, p + j] = u[idx - j]
            # MA part
            for k in range(r):
                Phi[t, p + q + 1 + k] = e[idx - 1 - k]
    
    y_out = y[start:]
    return Phi, y_out, start


def get_propagation_set(outlier_set, p, r, n):
    """Compute the propagation set for ARMAX.
    
    In ARMAX, an outlier at time t contaminates:
    - AR lags: regressor rows t+1, ..., t+p (via y_{t-i} in phi)
    - MA lags: residual rows t+1, ..., t+r (via e_{t-k} in phi)
    Total propagation: rows t, t+1, ..., t+max(p,r)
    
    But the AR contamination also affects residuals, which propagate
    through the MA part. The effective propagation is t, t+1, ..., t+p+r.
    
    Parameters
    ----------
    outlier_set : direct outlier indices O
    p : AR order
    r : MA order
    n : total sample size
    
    Returns
    -------
    prop_set : full propagation set (indices of all contaminated rows)
    """
    prop_set = set()
    for t in outlier_set:
        # Direct outlier + AR propagation + MA propagation
        for dt in range(0, p + r + 1):
            if t + dt < n:
                prop_set.add(t + dt)
    return np.sort(list(prop_set))


def true_parameter_vector(a, b, c):
    """Concatenate true parameters into a single vector theta*."""
    return np.concatenate([a, b, c])
