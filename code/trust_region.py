"""
Trust Region management and EHVI acquisition for TR-MOBO optimization.

Implements:
    - TrustRegion:               single trust region with adaptive sizing
    - MultiTrustRegionManager:   coordination of 4 TRs
    - compute_hypervolume:       Pareto front hypervolume indicator
    - compute_ehvi:              Expected Hypervolume Improvement (MC)
    - maximize_ehvi_in_tr:       candidate selection within TR
    - compute_improvement_ratio: TR update ratio rho
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube

from config import (
    TR_N_REGIONS, TR_LENGTH_INIT, TR_LENGTH_MIN, TR_LENGTH_MAX,
    TR_EXPAND_FACTOR, TR_SHRINK_FACTOR,
    TR_EXPAND_THRESHOLD, TR_SHRINK_THRESHOLD,
    EHVI_MC_SAMPLES, EHVI_REFERENCE_POINT, EHVI_CANDIDATES_PER_TR,
    THETA_LOWER, THETA_UPPER, N_THETA,
)


# ================================================================== #
#  1. Single Trust Region                                              #
# ================================================================== #

class TrustRegion:
    """A single trust region in normalized parameter space."""

    def __init__(self, center, length, bounds_lower, bounds_upper,
                 length_min=TR_LENGTH_MIN, length_max=TR_LENGTH_MAX,
                 expand_threshold=TR_EXPAND_THRESHOLD,
                 shrink_threshold=TR_SHRINK_THRESHOLD,
                 expand_factor=TR_EXPAND_FACTOR,
                 shrink_factor=TR_SHRINK_FACTOR):
        """
        Parameters
        ----------
        center : ndarray (17,)
            Center of the trust region in original parameter space.
        length : float
            Normalized TR half-length (fraction of parameter range).
        bounds_lower, bounds_upper : ndarray (17,)
            Global parameter bounds.
        """
        self.center = np.array(center, dtype=np.float64)
        self.length = float(length)
        self.bounds_lower = np.array(bounds_lower, dtype=np.float64)
        self.bounds_upper = np.array(bounds_upper, dtype=np.float64)
        self.length_min = length_min
        self.length_max = length_max
        self.expand_threshold = expand_threshold
        self.shrink_threshold = shrink_threshold
        self.expand_factor = expand_factor
        self.shrink_factor = shrink_factor

        # History
        self.history = []

    def get_bounds(self):
        """Get current TR bounds clipped to global bounds.

        Returns
        -------
        lower : ndarray (17,)
        upper : ndarray (17,)
        """
        param_range = self.bounds_upper - self.bounds_lower
        half_width = self.length * param_range / 2.0
        lower = np.maximum(self.center - half_width, self.bounds_lower)
        upper = np.minimum(self.center + half_width, self.bounds_upper)
        return lower, upper

    def update(self, rho):
        """Update TR size based on improvement ratio rho.

        Parameters
        ----------
        rho : float
            Improvement ratio (actual / predicted).
        """
        old_length = self.length
        if rho > self.expand_threshold:
            self.length = min(self.length * self.expand_factor, self.length_max)
        elif rho < self.shrink_threshold:
            self.length = max(self.length * self.shrink_factor, self.length_min)
        self.history.append({
            'rho': rho,
            'length_before': old_length,
            'length_after': self.length,
        })

    def contains(self, theta):
        """Check if theta is within the trust region.

        Parameters
        ----------
        theta : ndarray (17,)

        Returns
        -------
        bool
        """
        lower, upper = self.get_bounds()
        return bool(np.all(theta >= lower) and np.all(theta <= upper))

    @property
    def volume(self):
        """Normalized volume of the trust region (fraction of total space)."""
        return self.length ** N_THETA


# ================================================================== #
#  2. Multi Trust Region Manager                                       #
# ================================================================== #

class MultiTrustRegionManager:
    """Manages 4 trust regions for TR-MOBO."""

    def __init__(self, n_regions=TR_N_REGIONS,
                 bounds_lower=THETA_LOWER, bounds_upper=THETA_UPPER,
                 length_init=TR_LENGTH_INIT, **kwargs):
        self.n_regions = n_regions
        self.bounds_lower = np.array(bounds_lower, dtype=np.float64)
        self.bounds_upper = np.array(bounds_upper, dtype=np.float64)
        self.length_init = length_init
        self.tr_kwargs = kwargs
        self.regions = []

    def initialize_from_data(self, X, F):
        """Initialize TR centers from initial ABM data.

        Uses the best solution for each objective as a TR center.

        Parameters
        ----------
        X : ndarray (n, 17)
            Evaluated parameter vectors.
        F : ndarray (n, 4)
            Corresponding objective values (minimization form).
        """
        self.regions = []
        for obj_idx in range(min(self.n_regions, F.shape[1])):
            best_idx = np.argmin(F[:, obj_idx])
            center = X[best_idx].copy()
            tr = TrustRegion(
                center=center,
                length=self.length_init,
                bounds_lower=self.bounds_lower,
                bounds_upper=self.bounds_upper,
                **self.tr_kwargs,
            )
            self.regions.append(tr)

        # If fewer than n_regions objectives, add TRs at random best points
        while len(self.regions) < self.n_regions:
            idx = np.random.randint(X.shape[0])
            tr = TrustRegion(
                center=X[idx].copy(),
                length=self.length_init,
                bounds_lower=self.bounds_lower,
                bounds_upper=self.bounds_upper,
                **self.tr_kwargs,
            )
            self.regions.append(tr)

    def get_all_bounds(self):
        """Get bounds for all trust regions.

        Returns
        -------
        list of tuple(ndarray[17], ndarray[17])
        """
        return [tr.get_bounds() for tr in self.regions]

    def update_regions(self, region_idx, rho, new_center=None):
        """Update a specific trust region.

        Parameters
        ----------
        region_idx : int
        rho : float
            Improvement ratio.
        new_center : ndarray (17,), optional
            New center if improvement was positive.
        """
        self.regions[region_idx].update(rho)
        if new_center is not None and rho > 0:
            self.regions[region_idx].center = np.array(new_center, dtype=np.float64)

    def get_status(self):
        """Get status of all trust regions.

        Returns
        -------
        dict with lists of centers, lengths, histories.
        """
        return {
            'centers': [tr.center.copy() for tr in self.regions],
            'lengths': [tr.length for tr in self.regions],
            'histories': [tr.history.copy() for tr in self.regions],
            'volumes': [tr.volume for tr in self.regions],
        }

    def restart_region(self, region_idx):
        """Restart a trust region that has shrunk to minimum size.

        Parameters
        ----------
        region_idx : int
        """
        tr = self.regions[region_idx]
        # Re-center at a random point within global bounds
        param_range = self.bounds_upper - self.bounds_lower
        new_center = self.bounds_lower + np.random.random(N_THETA) * param_range
        tr.center = new_center
        tr.length = self.length_init
        tr.history.append({'restart': True})


# ================================================================== #
#  3. Hypervolume Computation                                          #
# ================================================================== #

def compute_hypervolume(F, reference_point):
    """Compute hypervolume indicator of a Pareto front.

    Parameters
    ----------
    F : ndarray (n, 4)
        Objective values (minimization form).
    reference_point : ndarray (4,)

    Returns
    -------
    float
        Hypervolume value.
    """
    if F.shape[0] == 0:
        return 0.0

    # Try pymoo's hypervolume first
    try:
        from pymoo.indicators.hv import HV
        indicator = HV(ref_point=np.array(reference_point, dtype=np.float64))
        return float(indicator(np.array(F, dtype=np.float64)))
    except (ImportError, Exception):
        pass

    # Fallback: simple 2D hypervolume for first 2 objectives, or MC estimation
    return _mc_hypervolume(F, reference_point, n_samples=10000)


def _mc_hypervolume(F, reference_point, n_samples=10000):
    """Monte Carlo estimate of hypervolume (fallback).

    Parameters
    ----------
    F : ndarray (n, d)
    reference_point : ndarray (d,)
    n_samples : int

    Returns
    -------
    float
    """
    F = np.array(F, dtype=np.float64)
    ref = np.array(reference_point, dtype=np.float64)
    d = F.shape[1]

    # Find ideal point
    ideal = F.min(axis=0)

    # Volume of bounding box
    box_vol = np.prod(ref - ideal)
    if box_vol <= 0:
        return 0.0

    # Sample random points in bounding box
    samples = np.random.uniform(ideal, ref, size=(n_samples, d))

    # Count points dominated by at least one Pareto point
    dominated = np.zeros(n_samples, dtype=bool)
    for i in range(F.shape[0]):
        dom_by_i = np.all(samples >= F[i], axis=1)
        dominated |= dom_by_i

    return box_vol * dominated.mean()


# ================================================================== #
#  4. Expected Hypervolume Improvement                                 #
# ================================================================== #

def compute_ehvi(mu, sigma, pareto_F, reference_point,
                 n_samples=EHVI_MC_SAMPLES):
    """Compute Expected Hypervolume Improvement via Monte Carlo.

    Parameters
    ----------
    mu : ndarray (4,)
        ANP predicted mean.
    sigma : ndarray (4,)
        ANP predicted standard deviation.
    pareto_F : ndarray (m, 4)
        Current Pareto front.
    reference_point : ndarray (4,)
    n_samples : int

    Returns
    -------
    float
        Expected hypervolume improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)

    # Current hypervolume
    if pareto_F.shape[0] > 0:
        hv_current = compute_hypervolume(pareto_F, ref)
    else:
        hv_current = 0.0

    # Monte Carlo sampling
    # f_samples ~ N(mu, sigma^2), shape (n_samples, 4)
    eps = np.random.randn(n_samples, len(mu))
    f_samples = mu[np.newaxis, :] + sigma[np.newaxis, :] * eps

    hv_improvements = np.zeros(n_samples)
    for i in range(n_samples):
        # Add sampled point to Pareto front
        if pareto_F.shape[0] > 0:
            augmented_F = np.vstack([pareto_F, f_samples[i:i+1]])
        else:
            augmented_F = f_samples[i:i+1]

        # Filter to non-dominated only (for efficiency)
        nd_mask = _is_non_dominated(augmented_F)
        nd_F = augmented_F[nd_mask]

        hv_new = compute_hypervolume(nd_F, ref)
        hv_improvements[i] = max(0.0, hv_new - hv_current)

    return float(hv_improvements.mean())


def _is_non_dominated(F):
    """Find non-dominated points.

    Parameters
    ----------
    F : ndarray (n, d)

    Returns
    -------
    mask : ndarray (n,) bool
    """
    n = F.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if i == j or not is_nd[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_nd[i] = False
                break
    return is_nd


# ================================================================== #
#  5. EHVI Maximization within Trust Region                            #
# ================================================================== #

def maximize_ehvi_in_tr(predictor, tr_bounds, pareto_F, reference_point,
                         n_candidates=EHVI_CANDIDATES_PER_TR, n_best=1):
    """Select next evaluation point by maximizing EHVI within a trust region.

    Uses LHS sampling + EHVI evaluation strategy.

    Parameters
    ----------
    predictor : ANPPredictor
    tr_bounds : tuple (lower, upper), each ndarray (17,)
    pareto_F : ndarray (m, 4)
    reference_point : ndarray (4,)
    n_candidates : int
    n_best : int

    Returns
    -------
    best_theta : ndarray (n_best, 17)
    """
    lower, upper = tr_bounds
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    # Ensure bounds are valid
    range_vec = upper - lower
    valid_dims = range_vec > 1e-12
    if not np.any(valid_dims):
        # Degenerate TR: return center
        return ((lower + upper) / 2.0).reshape(1, -1)

    # LHS sampling within TR bounds
    sampler = LatinHypercube(d=N_THETA)
    X_unit = sampler.random(n=n_candidates)
    X_candidates = lower + X_unit * range_vec  # (n_candidates, 17)

    # Evaluate EHVI for each candidate (objectives only, 4-dim)
    ehvi_values = np.zeros(n_candidates)
    if hasattr(predictor, 'predict_objectives'):
        mu_all, sigma_all = predictor.predict_objectives(X_candidates)
    else:
        mu_all, sigma_all = predictor.predict(X_candidates)

    for i in range(n_candidates):
        ehvi_values[i] = compute_ehvi(
            mu_all[i], sigma_all[i], pareto_F, reference_point)

    # Select top-n_best
    top_idx = np.argsort(ehvi_values)[-n_best:][::-1]
    return X_candidates[top_idx]


# ================================================================== #
#  6. Improvement Ratio                                                #
# ================================================================== #

def compute_improvement_ratio(predicted_hv_gain, actual_hv_gain):
    """Compute trust region improvement ratio rho.

    Parameters
    ----------
    predicted_hv_gain : float
        EHVI (expected gain from surrogate).
    actual_hv_gain : float
        Actual HV gain from ABM evaluation.

    Returns
    -------
    float
        rho = actual / predicted. Returns 0 if predicted <= 0.
    """
    if predicted_hv_gain <= 1e-12:
        return 0.0 if actual_hv_gain <= 1e-12 else 1.0
    return actual_hv_gain / predicted_hv_gain
