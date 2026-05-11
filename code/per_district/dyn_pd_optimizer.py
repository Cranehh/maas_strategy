"""Dynamic TR-MOBO for PD-5zones (Phase B for 360d θ).

Implements:
  - PhaseA 50 LHS init (independent of training data, fresh seeds)
  - TrustRegion class adapted to 360d (zone × stage × 12 dim)
  - 3 TR with fixed Chebyshev weights (TR_adoption, TR_revenue, TR_carbon)
  - Pareto manager (3-obj non-dominated, append-only)
  - Pure MIP-in-TR acquisition (no LHS in inner loop; D2 wires this in)
  - Continual learning: DL Model B partial_refit every K rounds, GP refit every round

Architecture matches plan file: /home/cranehh/.claude/plans/dl-mip-tr-lhs-mip-lhs-gp-rustling-penguin.md

D1 deliverable: framework runs Phase A LHS → empty Phase B loop (no MIP); next
step (D2) plugs in MIP acquisition.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.stats import qmc

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'per_district'))

import data_loader
import ch1_model
import ch2_model
import ch3_model
import torch
from ccnn_pipeline import (
    DYN_LOWER, DYN_UPPER, STAGE_WEEKS,
    TAU_HIGH_FIXED, TAU_LOW_FIXED, FREQ_ADJ_FIXED,
)
from per_district.abm_pd import PerDistrictABMSimulation
from per_district.labels_pd import build_full_theta_pd, build_stagewise_label_pd
from per_district.zone_aggregator import aggregate_per_district_to_zone
from per_district.regen_zone_data_v2 import aggregate_pd_obj_3obj
from per_district.analytical_pd import PerDistrictAnalyticalSurrogate
from per_district.ccnn_pd_v3_attn_model import ContextConditionedCNNPDv3Attn
from per_district.ccnn_pd_v3_model import CCNNPDv3Normalizer
from mip_encoder import MIPEncoder
from mip_encoder_grb import MIPEncoderGRB
import gurobipy as gp
from gurobipy import GRB


# ====================================================================
# Constants
# ====================================================================
K_STAGES = len(STAGE_WEEKS)        # 6
Z_ZONES = 5
DYN_DIM = 12
N_OBJ = 3                            # adoption / revenue / carbon

# Chebyshev fixed weights (ε avoids weak Pareto)
EPS_CHEB = 0.01
CHEB_WEIGHTS = np.array([
    [1.0 - 2 * EPS_CHEB, EPS_CHEB, EPS_CHEB],  # TR_adoption
    [EPS_CHEB, 1.0 - 2 * EPS_CHEB, EPS_CHEB],  # TR_revenue
    [EPS_CHEB, EPS_CHEB, 1.0 - 2 * EPS_CHEB],  # TR_carbon
])
N_TR = CHEB_WEIGHTS.shape[0]         # 3

# TR adaptation params: TR=0.2 to test MIP speed bottleneck (2026-05-11 user request)
# Smaller TR → tighter sample bounds → fewer mixed binaries → faster MIP.
TR_LENGTH_INIT = 0.2
TR_LENGTH_MIN = 0.05
TR_LENGTH_MAX = 0.4
TR_EXPAND_FACTOR = 1.5
TR_SHRINK_FACTOR = 0.5
TR_EXPAND_THRESHOLD = 0.75
TR_SHRINK_THRESHOLD = 0.25


# ====================================================================
# Trust Region (360d)
# ====================================================================
class TrustRegion360d:
    """Trust region in zone-level θ space (K=6, Z=5, 12) = 360d."""

    def __init__(self, center_zone_360: np.ndarray,
                 theta_lower_12: np.ndarray,
                 theta_upper_12: np.ndarray,
                 length: float = TR_LENGTH_INIT,
                 tr_id: int = 0):
        """
        center_zone_360 : (K, Z, 12) θ in PHYSICAL units
        theta_lower_12, theta_upper_12 : (12,) global bounds (per-dim, shared across stages/zones)
        length : normalized half-length in [0, 1]
        tr_id : 0=adoption / 1=revenue / 2=carbon (for Chebyshev weight)
        """
        self.center = np.asarray(center_zone_360, dtype=np.float64).copy()
        assert self.center.shape == (K_STAGES, Z_ZONES, DYN_DIM)
        self.tl = np.asarray(theta_lower_12, dtype=np.float64)
        self.tu = np.asarray(theta_upper_12, dtype=np.float64)
        self.length = float(length)
        self.tr_id = int(tr_id)
        self.history = []

    def get_bounds_zone(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) TR bounds in (K, Z, 12) shape, clipped to global."""
        param_range_12 = self.tu - self.tl
        half_width_12 = self.length * param_range_12 / 2.0
        # Broadcast to (K, Z, 12)
        half_width = np.broadcast_to(half_width_12, (K_STAGES, Z_ZONES, DYN_DIM)).copy()
        lo = np.maximum(self.center - half_width,
                          np.broadcast_to(self.tl, (K_STAGES, Z_ZONES, DYN_DIM)))
        hi = np.minimum(self.center + half_width,
                          np.broadcast_to(self.tu, (K_STAGES, Z_ZONES, DYN_DIM)))
        return lo, hi

    def update(self, rho: float, new_center: Optional[np.ndarray] = None):
        """Standard TR update based on improvement ratio rho.

        rho = (w·y_actual - best_so_far_under_w) /
              (w·μ_MIP_optimal - best_so_far_under_w)

        rho > 0.75 → expand;  rho < 0.25 → shrink.
        Move center to new_center if rho > 0 (improvement).
        """
        old_len = self.length
        if rho > TR_EXPAND_THRESHOLD:
            self.length = min(self.length * TR_EXPAND_FACTOR, TR_LENGTH_MAX)
        elif rho < TR_SHRINK_THRESHOLD:
            self.length = max(self.length * TR_SHRINK_FACTOR, TR_LENGTH_MIN)
        if new_center is not None and rho > 0:
            self.center = np.asarray(new_center, dtype=np.float64).copy()
        self.history.append({'rho': float(rho), 'length_before': old_len,
                              'length_after': self.length})


# ====================================================================
# Pareto manager (3-obj)
# ====================================================================
class ParetoManager3obj:
    """Append-only Pareto front manager for 3 objectives.

    Convention: ABM `cum_obj` output is ALREADY in min-form:
      [-adoption, -revenue, -carbon_reduction]
    (all 3 are negative quantities; ABM negates so optimizer can min uniformly.)
    Pareto stores min-form directly. Chebyshev minimization aligns with this.
    """

    def __init__(self):
        self.X: List[np.ndarray] = []     # each (K, Z, 12)
        self.Y_min: List[np.ndarray] = [] # each (3,) — same as input city_obj

    def add(self, theta_zone: np.ndarray, y_min: np.ndarray):
        self.X.append(np.asarray(theta_zone, dtype=np.float64).copy())
        self.Y_min.append(np.asarray(y_min, dtype=np.float64).copy())

    def get_nondominated_indices(self) -> np.ndarray:
        """Return indices of currently non-dominated points (min-form)."""
        if not self.Y_min:
            return np.array([], dtype=int)
        Y = np.stack(self.Y_min, axis=0)  # (N, 3)
        N = Y.shape[0]
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            for j in range(N):
                if i == j or not is_nd[j]:
                    continue
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    is_nd[i] = False
                    break
        return np.where(is_nd)[0]

    def hypervolume(self, reference_point: np.ndarray) -> float:
        """Pareto hypervolume in min-form."""
        nd = self.get_nondominated_indices()
        if len(nd) == 0:
            return 0.0
        Y = np.stack([self.Y_min[i] for i in nd], axis=0)
        try:
            from pymoo.indicators.hv import HV
            ind = HV(ref_point=np.asarray(reference_point, dtype=np.float64))
            return float(ind(Y))
        except (ImportError, Exception):
            # MC fallback
            return _mc_hypervolume_3obj(Y, reference_point, n_samples=10000)

    def __len__(self):
        return len(self.X)


def _mc_hypervolume_3obj(F: np.ndarray, ref: np.ndarray, n_samples: int = 10000) -> float:
    """MC estimate of 3-obj HV."""
    if F.shape[0] == 0:
        return 0.0
    ref = np.asarray(ref, dtype=np.float64)
    ideal = F.min(axis=0)
    box_vol = float(np.prod(ref - ideal))
    if box_vol <= 0:
        return 0.0
    samples = np.random.RandomState(42).uniform(low=ideal, high=ref, size=(n_samples, 3))
    dominated = np.zeros(n_samples, dtype=bool)
    for i in range(F.shape[0]):
        dominated |= np.all(samples >= F[i:i+1], axis=1)
    return box_vol * dominated.mean()


# ====================================================================
# ABM evaluation pipeline (zone θ → district → ABM → zone 3-obj)
# ====================================================================
def zone_to_district_theta(theta_zone: np.ndarray, zone_id: np.ndarray) -> np.ndarray:
    """(K, Z, 12) → (K, D=17, 12) via zone_id broadcasting."""
    K, Z, C = theta_zone.shape
    D = len(zone_id)
    theta_pd = np.zeros((K, D, C), dtype=np.float64)
    for d in range(D):
        theta_pd[:, d, :] = theta_zone[:, zone_id[d], :]
    return theta_pd


def evaluate_zone_theta_via_abm(theta_zone: np.ndarray,
                                  abm_pd: PerDistrictABMSimulation,
                                  zone_id: np.ndarray,
                                  district_pop: np.ndarray,
                                  zone_pop: np.ndarray,
                                  abm_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full evaluation: zone θ → district → ABM → (zone 3-obj, city 3-obj, stage_cum_zone).

    Returns
    -------
    zone_obj : (Z, 3)         per-zone terminal [adoption, revenue, carbon] (min-form)
    city_obj : (3,)           city-aggregate (min-form):
                               adoption pop-mean, revenue/carbon raw sum
    stage_cum_zone : (Z, K, 3) per-zone per-stage cumulative obj (min-form)
    """
    theta_pd = zone_to_district_theta(theta_zone, zone_id)  # (K, D, 12)
    schedule = [build_full_theta_pd(theta_pd[k]) for k in range(theta_pd.shape[0])]
    cum_obj_pd, _, _, stage_deltas_pd = abm_pd.run_dynamic_pd(
        schedule, list(STAGE_WEEKS), seed=abm_seed, softmax_revenue=True)
    # cum_obj_pd: (D, 4) — terminal min-form [-adopt, -rev, gini, -carbon]
    # stage_deltas_pd: (K, D, 4) — per-stage delta of cum_obj
    zone_3obj = aggregate_pd_obj_3obj(cum_obj_pd[None, :, :], zone_id, district_pop)[0]  # (Z, 3)

    # Per-stage cumulative obj at zone level (min-form 3-obj)
    K_local = stage_deltas_pd.shape[0]
    Z_local = int(zone_id.max() + 1)
    stage_cum_pd = np.cumsum(stage_deltas_pd, axis=0)            # (K, D, 4)
    stage_cum_zone = np.zeros((Z_local, K_local, N_OBJ), dtype=np.float64)
    for k in range(K_local):
        stage_cum_zone[:, k, :] = aggregate_pd_obj_3obj(
            stage_cum_pd[k:k+1], zone_id, district_pop)[0]       # (Z, 3)

    # City-aggregate (min-form, all 3 to be minimized):
    total_pop = float(zone_pop.sum())
    city_adopt = float(np.sum(zone_3obj[:, 0] * zone_pop) / total_pop)
    city_revenue = float(np.sum(zone_3obj[:, 1]))
    city_carbon = float(np.sum(zone_3obj[:, 2]))
    city_obj = np.array([city_adopt, city_revenue, city_carbon], dtype=np.float64)
    return zone_3obj, city_obj, stage_cum_zone


# ====================================================================
# Phase A: 50 LHS Init
# ====================================================================
def phase_a_lhs_init(abm_pd: PerDistrictABMSimulation,
                      zone_id: np.ndarray,
                      district_pop: np.ndarray,
                      zone_pop: np.ndarray,
                      theta_lower_12: np.ndarray,
                      theta_upper_12: np.ndarray,
                      n_samples: int = 50,
                      seed_lhs: int = 555,
                      seed_abm_base: int = 1000,
                      verbose: bool = True) -> Dict:
    """LHS sample 50 zone θ, evaluate via ABM, build initial Pareto + context.

    Returns dict with:
      X_zone : (N, K, Z, 12) sampled θ
      Y_zone : (N, Z, 3)     per-zone obj
      Y_city : (N, 3)        city-aggregate obj
      pareto : ParetoManager3obj
    """
    K, Z, C = K_STAGES, Z_ZONES, DYN_DIM
    if verbose:
        print(f"[Phase A] LHS sample {n_samples} × (K={K}, Z={Z}, 12) = {n_samples}×360d")

    # Latin hypercube in flat 360 dim, then reshape
    sampler = qmc.LatinHypercube(d=K * Z * C, seed=seed_lhs)
    U = sampler.random(n_samples)                              # (N, 360) in [0, 1]
    # Scale per-dim (broadcast 12d bounds across K×Z=30 stages-zones)
    bounds_lo = np.broadcast_to(theta_lower_12, (K, Z, C)).reshape(-1)
    bounds_hi = np.broadcast_to(theta_upper_12, (K, Z, C)).reshape(-1)
    samples_flat = bounds_lo + U * (bounds_hi - bounds_lo)
    X_zone = samples_flat.reshape(n_samples, K, Z, C)          # (N, K, Z, 12)

    Y_zone = np.zeros((n_samples, Z, N_OBJ), dtype=np.float64)
    Y_city = np.zeros((n_samples, N_OBJ), dtype=np.float64)
    pareto = ParetoManager3obj()

    # Also capture stage_cum_zone (K, Z, 3) per sample for Model B context
    Y_stage_cum = np.zeros((n_samples, Z, K, N_OBJ), dtype=np.float64)

    t0 = time.time()
    for i in range(n_samples):
        zone_obj_i, city_obj_i, stage_cum_zone_i = evaluate_zone_theta_via_abm(
            X_zone[i], abm_pd, zone_id, district_pop, zone_pop,
            abm_seed=seed_abm_base + i)
        Y_zone[i] = zone_obj_i
        Y_city[i] = city_obj_i
        Y_stage_cum[i] = stage_cum_zone_i
        pareto.add(X_zone[i], city_obj_i)
        if verbose and ((i + 1) % 10 == 0 or i == 0):
            elapsed = time.time() - t0
            print(f"  [Phase A] {i+1}/{n_samples}  "
                  f"adopt={city_obj_i[0]:.4f} rev={city_obj_i[1]:.2e} "
                  f"carbon={city_obj_i[2]:.2e}  ({elapsed:.1f}s)")
    if verbose:
        print(f"[Phase A] done in {time.time()-t0:.1f}s; |Pareto|={len(pareto.get_nondominated_indices())}")

    return {
        'X_zone': X_zone,
        'Y_zone': Y_zone,             # (N, Z, 3) terminal
        'Y_city': Y_city,             # (N, 3)
        'Y_stage_cum': Y_stage_cum,   # (N, Z, K, 3)
        'pareto': pareto,
    }


# ====================================================================
# TR initialization from Phase A
# ====================================================================
def init_trs_from_phase_a(phase_a_result: Dict,
                           theta_lower_12: np.ndarray,
                           theta_upper_12: np.ndarray) -> List[TrustRegion360d]:
    """Place each of 3 TRs at the Phase A best point for that objective.

    All 3 objectives are MINIMIZED in ABM min-form, so argmin uniformly.
      TR_0 (adoption-biased): argmin Y_city[:, 0]  (most negative = best adoption)
      TR_1 (revenue-biased):  argmin Y_city[:, 1]
      TR_2 (carbon-biased):   argmin Y_city[:, 2]
    """
    Y_city = phase_a_result['Y_city']  # (N, 3) min-form
    X_zone = phase_a_result['X_zone']  # (N, K, Z, 12)
    trs = []
    for tr_id in range(N_TR):
        idx = int(np.argmin(Y_city[:, tr_id]))
        trs.append(TrustRegion360d(X_zone[idx], theta_lower_12, theta_upper_12,
                                    length=TR_LENGTH_INIT, tr_id=tr_id))
    return trs


# ====================================================================
# Chebyshev scalarization helper
# ====================================================================
def chebyshev_value(y_min: np.ndarray, weights: np.ndarray) -> float:
    """Weighted-sum scalarization on min-form objectives.

    All 3 objs are minimized in ABM's min-form. MIP MINIMIZES this scalar.

    Args
    ----
    y_min: (3,) ABM cum_obj in min-form [-adopt, -revenue, -carbon_reduction]
    weights: (3,) one of CHEB_WEIGHTS rows (positive)

    Returns
    -------
    weighted-sum value (lower = better)
    """
    return float(np.dot(weights, y_min))


# ====================================================================
# Model B loader + context_y_zone constructor
# ====================================================================
def load_model_b(ckpt_path: str, device: str = 'cpu'):
    """Load Model B (ContextConditionedCNNPDv3Attn) + normalizer from ckpt."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state']
    D, y_dim = ckpt['D'], ckpt['y_dim']
    model = ContextConditionedCNNPDv3Attn(D=D, y_dim=y_dim, K=K_STAGES, decoder_dropout=0.05)
    model.load_state_dict(sd); model.eval()
    norm = CCNNPDv3Normalizer(ckpt['theta_lower'], ckpt['theta_upper'],
                                D=D, y_dim=y_dim, idx_obj_start=ckpt['idx_obj_start'])
    norm.y_mean = ckpt['y_mean']
    norm.y_std = ckpt['y_std']
    norm.fitted = True
    return model, sd, norm, ckpt


def build_context_y_zone(theta_zone_history: np.ndarray,
                          stage_cum_history: np.ndarray,
                          surr_pd: PerDistrictAnalyticalSurrogate,
                          zone_id: np.ndarray,
                          district_pop: np.ndarray) -> np.ndarray:
    """Build (M, Z, 27) context y for Model B from history.

    y27 layout: [0:9] = 9 analytical inter (zone-pop-weighted mean)
                [9:27] = 6 stages × 3 obj (from ABM stage_cum_history)

    Args
    ----
    theta_zone_history : (M, K, Z, 12)
    stage_cum_history : (M, Z, K, 3) — from ABM
    """
    M = theta_zone_history.shape[0]
    Z = Z_ZONES
    y27 = np.zeros((M, Z, 27), dtype=np.float64)
    for i in range(M):
        # Analytical for 9 inter (use existing helper from ccnn_zone_v2_attn_pipeline)
        from per_district.ccnn_zone_v2_attn_pipeline import (
            build_stagewise_label_zone_3obj as _build_zone_label
        )
        y27_anal = _build_zone_label(surr_pd, theta_zone_history[i],
                                       zone_id, district_pop, softmax_revenue=True)
        # y27_anal: (Z, 27) — analytical for all 27 dims
        # Override [9:27] with ABM stage cumulative obj
        y27_anal[:, 9:27] = stage_cum_history[i].reshape(Z, K_STAGES * N_OBJ)
        y27[i] = y27_anal
    return y27


# ====================================================================
# MIP acquisition (DL Model B)
# ====================================================================
def mip_dl_acquisition(state_dict, normalizer, surr_pd,
                        condition_global: np.ndarray,
                        context_theta_hist: np.ndarray,   # (M, K, Z, 12)
                        context_y27_hist: np.ndarray,     # (M, Z, 27)
                        tr: TrustRegion360d,
                        zone_pop: np.ndarray,
                        theta_lower_12: np.ndarray,
                        theta_upper_12: np.ndarray,
                        time_limit: int = 1800,
                        mip_gap: float = 1e-4,
                        relu_mode: str = 'bigm',
                        sound_bounds_only: bool = True,
                        pbt_budget_per_layer: int = 256,
                        n_sample_bounds: int = 300,
                        verbose: bool = False) -> Tuple[np.ndarray, float, float, int]:
    """Build + solve MIP for DL Model B with Chebyshev scalarization.

    Per the 2026-05-11 plan: strict exact solve via Tjeng/Xiao/Tedrake
    (ICLR 2019 MIPVerify) approach — big-M ReLU + sound IBP + LP-PBT. Raises
    RuntimeError if Gurobi cannot prove optimality within `time_limit`
    (Goal 4: ≤ 1800 s).

    Returns
    -------
    theta_star : (K, Z, 12) physical
    mu_predicted : (3,) predicted min-form city obj at θ*
    solve_time : float seconds
    binaries : int  (after PBT)
    """
    encoder = MIPEncoderGRB()
    tr_lo, tr_hi = tr.get_bounds_zone()       # (K, Z, 12)

    # Build MIP at TR center with big-M ReLU + sound IBP (+ optional LP-PBT)
    m, info = encoder.encode_pd_v3_attn(
        state_dict,
        theta_lower_12, theta_upper_12,
        condition_global,
        context_theta_hist,
        context_y27_hist,
        tr_center_pd=tr.center,
        normalizer=normalizer,
        tr_lower_pd=tr_lo, tr_upper_pd=tr_hi,
        n_sample_bounds=n_sample_bounds,
        relu_mode=relu_mode,
        sound_bounds_only=sound_bounds_only,
        pbt_budget_per_layer=pbt_budget_per_layer,
        verbose=verbose)

    # Tighten θ_n bounds to TR (gurobipy uses .LB/.UB attrs)
    K, Z, _ = tr.center.shape
    for k in range(K):
        for d in range(Z):
            for c in range(12):
                rng = theta_upper_12[c] - theta_lower_12[c]
                if rng < 1e-12:
                    continue
                lo_n = (tr_lo[k, d, c] - theta_lower_12[c]) / rng
                hi_n = (tr_hi[k, d, c] - theta_lower_12[c]) / rng
                v = info['theta_n_vars'][k, d, c]
                v.LB = max(float(v.LB), float(lo_n))
                v.UB = min(float(v.UB), float(hi_n))

    # Build Chebyshev city objective (MIN form) — see Phase 4 plan
    # k=5 terminal: idx[adopt]=24, idx[rev]=25, idx[carb]=26
    # adoption: pop-weighted mean across zones; revenue/carbon: raw sum
    w_tr = CHEB_WEIGHTS[tr.tr_id]              # (3,)
    Y_MEAN = info['normalizer_y_mean']         # (Z, 27)
    Y_STD = info['normalizer_y_std']
    total_pop = float(zone_pop.sum())

    OBJ_IDX = [24, 25, 26]
    expr = gp.LinExpr()
    for w_idx, y_idx in enumerate(OBJ_IDX):
        coeff_per_zone = (zone_pop / total_pop) if w_idx == 0 else np.ones(Z)
        for d in range(Z):
            sc = float(Y_STD[d, y_idx])
            mu_var = info['mu_vars_per_zone'][d][y_idx]
            expr.add(mu_var, float(w_tr[w_idx] * coeff_per_zone[d] * sc))

    m.setObjective(expr, GRB.MINIMIZE)

    # Strict exact solve. Raises RuntimeError if not OPTIMAL within time_limit.
    status, t_solve, obj_val = encoder.solve_exact(
        m, time_limit=time_limit, mip_gap=mip_gap, mip_focus=1,
        threads=8, verbose=False)

    print(f"  [mip_dl] solve_exact OK: t={t_solve:.1f}s, obj={obj_val:.4e}, "
          f"binaries={info['binaries']}, status={status}")

    # Extract θ* in physical units (gurobipy uses .X)
    K, Z, _ = tr.center.shape
    theta_star = np.zeros((K, Z, 12), dtype=np.float64)
    for k in range(K):
        for d in range(Z):
            for c in range(12):
                v = info['theta_n_vars'][k, d, c]
                theta_n_val = float(v.X)
                rng = theta_upper_12[c] - theta_lower_12[c]
                theta_star[k, d, c] = theta_lower_12[c] + theta_n_val * rng

    # Predicted (denormalized) mu in min-form
    mu_predicted = np.zeros(3)
    for w_idx, y_idx in enumerate(OBJ_IDX):
        coeff_per_zone = (zone_pop / total_pop) if w_idx == 0 else np.ones(Z)
        accum = 0.0
        for d in range(Z):
            sc = float(Y_STD[d, y_idx])
            mu_off = float(Y_MEAN[d, y_idx])
            mu_var = info['mu_vars_per_zone'][d][y_idx]
            accum += coeff_per_zone[d] * (float(mu_var.X) * sc + mu_off)
        mu_predicted[w_idx] = accum

    return theta_star, mu_predicted, t_solve, info['binaries']


# ====================================================================
# Main Phase B loop
# ====================================================================
def run_phase_b(trs: List[TrustRegion360d],
                 phase_a_result: Dict,
                 abm_pd: PerDistrictABMSimulation,
                 surr_pd: PerDistrictAnalyticalSurrogate,
                 zone_id: np.ndarray,
                 district_pop: np.ndarray,
                 zone_pop: np.ndarray,
                 condition_global: np.ndarray,
                 theta_lower_12: np.ndarray,
                 theta_upper_12: np.ndarray,
                 model_b_ckpt_path: Optional[str] = None,
                 n_rounds: int = 100,
                 surrogate_kind: str = 'dl_model_b',
                 seed_abm_base: int = 2000,
                 mip_time_limit: int = 1800,
                 mip_gap: float = 1e-4,
                 relu_mode: str = 'bigm',
                 sound_bounds_only: bool = True,
                 pbt_budget_per_layer: int = 0,
                 finetune_every: int = 5,
                 finetune_epochs: int = 20,
                 verbose: bool = True) -> Dict:
    """Phase B: 3 TR × n_rounds. Surrogate-driven MIP acquisition.

    D2: DL Model B MIP path implemented.
    D3: Anal+GP MIP path (TODO).
    """
    pareto = phase_a_result['pareto']
    context_theta = list(phase_a_result['X_zone'])              # list of (K, Z, 12)
    context_y_city = list(phase_a_result['Y_city'])             # list of (3,) min-form
    context_y_zone = list(phase_a_result['Y_zone'])             # list of (Z, 3) terminal
    context_y_stage_cum = list(phase_a_result['Y_stage_cum'])   # list of (Z, K, 3)

    # Load DL Model B if needed
    state_dict, normalizer = None, None
    if surrogate_kind == 'dl_model_b':
        assert model_b_ckpt_path is not None, "Model B ckpt path required"
        if verbose:
            print(f"  Loading Model B from {model_b_ckpt_path}")
        _, state_dict, normalizer, _ = load_model_b(model_b_ckpt_path)

    ref_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    hv_history = [pareto.hypervolume(ref_point)]
    wall_clock = [0.0]
    abm_count_history = [len(context_theta)]
    solve_times = []
    binaries_log = []

    t_global = time.time()

    for round_idx in range(n_rounds):
        for tr_id, tr in enumerate(trs):
            # Build context arrays for MIP
            ctx_theta_arr = np.stack(context_theta, axis=0)             # (M, K, Z, 12)
            ctx_stage_cum_arr = np.stack(context_y_stage_cum, axis=0)   # (M, Z, K, 3)
            # Build (M, Z, 27) context y for Model B
            if surrogate_kind == 'dl_model_b':
                ctx_y27 = build_context_y_zone(
                    ctx_theta_arr, ctx_stage_cum_arr, surr_pd, zone_id, district_pop)
                # MIP acquisition — strict exact (MIPVerify big-M + sound IBP + LP-PBT)
                theta_star, mu_predicted, t_solve, n_bin = mip_dl_acquisition(
                    state_dict, normalizer, surr_pd,
                    condition_global,
                    ctx_theta_arr, ctx_y27,
                    tr, zone_pop,
                    theta_lower_12, theta_upper_12,
                    time_limit=mip_time_limit,
                    mip_gap=mip_gap,
                    relu_mode=relu_mode,
                    sound_bounds_only=sound_bounds_only,
                    pbt_budget_per_layer=pbt_budget_per_layer,
                    n_sample_bounds=0,
                    verbose=False)
                solve_times.append(t_solve)
                binaries_log.append(n_bin)
            else:
                # TODO: anal_gp branch (D3)
                raise NotImplementedError(f"surrogate_kind={surrogate_kind} not yet implemented")

            # ABM evaluate
            abm_seed = seed_abm_base + round_idx * N_TR + tr_id
            zone_obj_star, city_obj_star, stage_cum_zone_star = evaluate_zone_theta_via_abm(
                theta_star, abm_pd, zone_id, district_pop, zone_pop,
                abm_seed=abm_seed)

            # Update Pareto + context
            pareto.add(theta_star, city_obj_star)
            context_theta.append(theta_star)
            context_y_city.append(city_obj_star)
            context_y_zone.append(zone_obj_star)
            context_y_stage_cum.append(stage_cum_zone_star)

            # TR update with weighted-sum scalarized rho (min form, lower=better)
            w_tr = CHEB_WEIGHTS[tr_id]
            cheb_actual = chebyshev_value(city_obj_star, w_tr)
            cheb_pred = float(np.dot(w_tr, mu_predicted))
            best_so_far = min(chebyshev_value(y, w_tr) for y in context_y_city[:-1])
            improvement = best_so_far - cheb_actual
            if best_so_far - cheb_pred > 1e-12:
                rho = improvement / (best_so_far - cheb_pred)
            else:
                rho = 0.0
            tr.update(rho, new_center=theta_star if improvement > 0 else None)

        # End-of-round bookkeeping
        hv_now = pareto.hypervolume(ref_point)
        hv_history.append(hv_now)
        abm_count_history.append(len(context_theta))
        wall_clock.append(time.time() - t_global)

        if verbose and ((round_idx + 1) % 1 == 0 or round_idx == 0):
            recent_solve = np.mean(solve_times[-N_TR:]) if solve_times else 0
            print(f"  [Phase B] round {round_idx+1}/{n_rounds}  "
                  f"HV={hv_now:.3e}  |Pareto|={len(pareto.get_nondominated_indices())}  "
                  f"|context|={len(context_theta)}  "
                  f"avg_solve={recent_solve:.1f}s  "
                  f"wall={wall_clock[-1]:.1f}s")

    return {
        'pareto': pareto,
        'context_theta': context_theta,
        'context_y_city': context_y_city,
        'context_y_zone': context_y_zone,
        'context_y_stage_cum': context_y_stage_cum,
        'hv_history': hv_history,
        'abm_count_history': abm_count_history,
        'wall_clock': wall_clock,
        'tr_history': [tr.history for tr in trs],
        'solve_times': solve_times,
        'binaries_log': binaries_log,
    }


# ====================================================================
# Main entry
# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surrogate', choices=['dl_model_b', 'anal_gp'],
                     default='dl_model_b')
    ap.add_argument('--seed', type=int, default=2026,
                     help='LHS seed for Phase A init')
    ap.add_argument('--rounds', type=int, default=100, help='Phase B rounds')
    ap.add_argument('--phase-a-samples', type=int, default=50)
    ap.add_argument('--model-b-ckpt', default='checkpoints/ccnn_zone_v2_attn_seed2026.pt')
    ap.add_argument('--mip-time-limit', type=int, default=60)
    ap.add_argument('--output-dir', default='results/dyn_pd_runs')
    ap.add_argument('--quick', action='store_true',
                     help='Quick test: 5 Phase A samples, 3 rounds (smoke test)')
    args = ap.parse_args()

    if args.quick:
        args.phase_a_samples = 5
        args.rounds = 3

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Dyn PD Optimizer ===")
    print(f"  surrogate: {args.surrogate}")
    print(f"  seed: {args.seed}")
    print(f"  Phase A samples: {args.phase_a_samples}")
    print(f"  Phase B rounds: {args.rounds}")

    # Load agents + ABM + analytical
    print("Loading agents + PerDistrictABM + analytical surrogate...")
    agents = data_loader.create_agent_population()
    abm_pd = PerDistrictABMSimulation(agents, ch3_model, ch1_model, ch2_model)
    surr_pd = PerDistrictAnalyticalSurrogate(agents, ch3_model, ch1_model, ch2_model)

    # Global condition vector (96,)
    from neural_process import aggregate_condition_vector
    condition_global = aggregate_condition_vector(agents).astype(np.float64)

    # Zone mapping
    zm = np.load(os.path.join(ROOT, 'data/zone_mapping.npz'))
    zone_id = zm['zone_id']
    district_pop = zm['district_pop'].astype(np.float64)
    zone_pop = zm['zone_pop'].astype(np.float64)
    print(f"  D={len(zone_id)}, Z={int(zone_id.max()+1)}, total_pop={district_pop.sum():.0f}")

    # Phase A LHS
    pa = phase_a_lhs_init(
        abm_pd, zone_id, district_pop, zone_pop,
        DYN_LOWER, DYN_UPPER,
        n_samples=args.phase_a_samples,
        seed_lhs=args.seed * 7,           # distinct from training data LHS seeds
        seed_abm_base=args.seed * 11,
    )

    # Init TRs at Phase A bests
    trs = init_trs_from_phase_a(pa, DYN_LOWER, DYN_UPPER)
    print(f"  Initialized {len(trs)} TRs:")
    for i, tr in enumerate(trs):
        obj_name = ['adoption', 'revenue', 'carbon'][i]
        print(f"    TR_{i} ({obj_name}): center pop-mean={tr.center.mean():.3f}, "
              f"length={tr.length:.3f}")

    # Phase B (D2: real DL-MIP)
    print(f"Phase B (MIP-in-TR, surrogate={args.surrogate})...")
    pb = run_phase_b(
        trs, pa, abm_pd, surr_pd, zone_id, district_pop, zone_pop,
        condition_global, DYN_LOWER, DYN_UPPER,
        model_b_ckpt_path=os.path.join(ROOT, args.model_b_ckpt),
        n_rounds=args.rounds,
        surrogate_kind=args.surrogate,
        seed_abm_base=args.seed * 13,
        mip_time_limit=args.mip_time_limit,
    )

    # Save results
    out_path = os.path.join(args.output_dir,
                              f"{args.surrogate}_seed{args.seed}.npz")
    np.savez(out_path,
              X_zone=np.stack(pb['context_theta'], axis=0),
              Y_city=np.stack(pb['context_y_city'], axis=0),
              Y_zone=np.stack(pb['context_y_zone'], axis=0),
              hv_history=np.array(pb['hv_history']),
              abm_count_history=np.array(pb['abm_count_history']),
              wall_clock=np.array(pb['wall_clock']),
              )
    print(f"Saved {out_path}")
    print(f"Final HV: {pb['hv_history'][-1]:.3e}")
    print(f"Total ABM calls: {len(pb['context_theta'])}")
    print(f"Total wall-clock: {pb['wall_clock'][-1]:.1f}s")


if __name__ == '__main__':
    main()
