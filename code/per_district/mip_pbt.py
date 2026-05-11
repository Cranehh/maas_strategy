"""mip_pbt.py — Layer-Incremental LP-based Progressive Bounds Tightening.

Implements Tjeng, Xiao, Tedrake (ICLR 2019, MIPVerify) Appendix B PBT:

  "All the information required to determine the best possible bounds on v is
   contained in the subtree of G rooted at v, G_v. LP considers the full
   subtree G_v but relaxes all integer constraints."

The single exported function `tighten_layer_via_lp` is called by the encoder
ONCE PER LAYER, on the partial model `m` built up to (and including) the
current layer's pre-activations but NOT downstream layers. The partial model
is small at layer 1 and grows as the encoder progresses, so per-LP cost stays
bounded.

For each pre-activation x_i in the current layer:
  1. Take m.relax() (cheap because m is partial).
  2. setObjective(MAXIMIZE x_i), optimize → u_LP. Tighten x_i.UB.
  3. If x_i.UB ≤ ε_stable: stable-dead → caller will encode y_i = 0, no binary.
  4. Else setObjective(MINIMIZE x_i), optimize → l_LP. Tighten x_i.LB.
  5. If x_i.LB ≥ -ε_stable: stable-active → caller will encode y_i = x_i.

The LP snapshot is built ONCE per layer (passed in or rebuilt) and reused
across all per-unit LPs in that layer, with warm-started dual simplex on
objective changes.

Replaces the previous "single full-MIP relaxation" approach (2026-05-11
morning) which timed out at scale.
"""
from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def tighten_layer_via_lp(
        m: gp.Model,
        x_vars: List[gp.Var],
        ia_lo: np.ndarray,
        ia_hi: np.ndarray,
        eps_stable: float = 1e-6,
        budget: int = 128,
        skip_lp: bool = False,
        max_seconds: float = 30.0,
        verbose: bool = False,
) -> dict:
    """Tighten (LB, UB) of x_vars in-place on the current partial model m.

    Parameters
    ----------
    m : gurobipy.Model
        The encoder's partial model. Must contain x_vars' defining linear
        constraints + all upstream layers (encoded with bigm/IA bounds).
    x_vars : list[Var]
        The current layer's pre-activations. Their LB/UB will be tightened.
    ia_lo, ia_hi : np.ndarray
        Interval-arithmetic bounds for this layer's pre-activations. Used as
        the OUTER bound (LP cannot go beyond these — defense against numerical
        issues in LP solver).
    eps_stable : float
        A unit is stable-dead if UB ≤ eps_stable, stable-active if LB ≥
        −eps_stable.
    budget : int
        Maximum number of mixed pre-activations to tighten via LP (ranked by
        |IA_l| · |IA_u|). The remaining mixed units keep their IA bounds.
    skip_lp : bool
        If True (use for layer 1, per Tjeng footnote 4): apply IA bounds only,
        skip the LP step. IA is provably optimal at layer 1 because the input
        vars are independent.
    verbose : bool
        Print per-layer summary.

    Returns
    -------
    stats : dict
        {'n_total', 'n_mixed_pre_lp', 'n_lps', 'n_newly_dead',
         'n_newly_active', 'n_tightened', 't_seconds', 'lp_times_ms'}
    """
    t_start = time.time()
    n_total = len(x_vars)

    # Step 0: ensure freshly-added vars/constraints are resolved before
    # we read attributes off them.
    m.update()

    # Step 1: Apply IA bounds (intersect with whatever LB/UB the var has).
    for i, v in enumerate(x_vars):
        v.LB = max(float(v.LB), float(ia_lo[i]))
        v.UB = min(float(v.UB), float(ia_hi[i]))
    m.update()

    if skip_lp:
        n_dead = sum(1 for v in x_vars if float(v.UB) <= eps_stable)
        n_active = sum(1 for v in x_vars if float(v.LB) >= -eps_stable)
        return {
            'n_total': n_total,
            'n_mixed_pre_lp': n_total - n_dead - n_active,
            'n_lps': 0, 'n_newly_dead': 0, 'n_newly_active': 0,
            'n_tightened': 0,
            't_seconds': time.time() - t_start,
            'lp_times_ms': [],
            'skip_lp': True,
        }

    # Step 2: Rank mixed units by big-M magnitude (largest payoff first).
    mixed = []
    for i in range(n_total):
        lo, hi = float(x_vars[i].LB), float(x_vars[i].UB)
        if lo < -eps_stable and hi > eps_stable:
            mixed.append((i, abs(lo) * abs(hi)))
    mixed.sort(key=lambda t: t[1], reverse=True)
    n_mixed_pre_lp = len(mixed)
    candidates = mixed[:budget] if budget > 0 else mixed

    if not candidates:
        return {
            'n_total': n_total,
            'n_mixed_pre_lp': n_mixed_pre_lp,
            'n_lps': 0, 'n_newly_dead': 0, 'n_newly_active': 0,
            'n_tightened': 0,
            't_seconds': time.time() - t_start,
            'lp_times_ms': [],
        }

    # Step 3: Build LP snapshot of partial m.
    t_relax_start = time.time()
    m_lp = m.relax()
    m_lp.setParam('OutputFlag', 0)
    m_lp.setParam('Method', 1)        # dual simplex (warm-starts well)
    m_lp.setParam('Threads', 1)
    m_lp.setParam('Presolve', 0)
    m_lp.setParam('Crossover', 0)
    t_relax = time.time() - t_relax_start

    # Step 4: Per-candidate LP min/max with early-return-on-stable.
    n_lps = 0
    n_newly_dead = 0
    n_newly_active = 0
    n_tightened = 0
    lp_times_ms = []
    truncated = False

    for idx, _ in candidates:
        # Per-layer time cap — bail out if we've already spent too long.
        if time.time() - t_start > max_seconds:
            truncated = True
            break
        xv = x_vars[idx]
        lp_xv = m_lp.getVarByName(xv.VarName)
        if lp_xv is None:
            continue

        # 4a. Max → tighter UB.
        t0 = time.time()
        try:
            m_lp.setObjective(lp_xv, GRB.MAXIMIZE)
            m_lp.optimize()
        except gp.GurobiError as e:
            if verbose:
                print(f"      [PBT] LP max failed on {xv.VarName}: {e}")
            continue
        n_lps += 1
        lp_times_ms.append((time.time() - t0) * 1000)

        if m_lp.Status == GRB.OPTIMAL:
            new_u = float(m_lp.ObjVal)
            if new_u < xv.UB - 1e-9:
                new_u_safe = max(new_u, xv.LB)
                xv.UB = new_u_safe
                lp_xv.UB = new_u_safe
                n_tightened += 1

        # 4b. Early return if now stable-dead.
        if xv.UB <= eps_stable:
            n_newly_dead += 1
            continue

        # 4c. Min → tighter LB.
        t0 = time.time()
        try:
            m_lp.setObjective(lp_xv, GRB.MINIMIZE)
            m_lp.optimize()
        except gp.GurobiError as e:
            if verbose:
                print(f"      [PBT] LP min failed on {xv.VarName}: {e}")
            continue
        n_lps += 1
        lp_times_ms.append((time.time() - t0) * 1000)

        if m_lp.Status == GRB.OPTIMAL:
            new_l = float(m_lp.ObjVal)
            if new_l > xv.LB + 1e-9:
                new_l_safe = min(new_l, xv.UB)
                xv.LB = new_l_safe
                lp_xv.LB = new_l_safe
                n_tightened += 1

        if xv.LB >= -eps_stable:
            n_newly_active += 1

    m.update()
    t_total = time.time() - t_start

    stats = {
        'n_total': n_total,
        'n_mixed_pre_lp': n_mixed_pre_lp,
        'n_lps': n_lps,
        'n_newly_dead': n_newly_dead,
        'n_newly_active': n_newly_active,
        'n_tightened': n_tightened,
        't_seconds': t_total,
        't_relax_seconds': t_relax,
        'lp_times_ms': lp_times_ms,
        'mean_lp_ms': float(np.mean(lp_times_ms)) if lp_times_ms else 0.0,
        'max_lp_ms': float(np.max(lp_times_ms)) if lp_times_ms else 0.0,
        'truncated': truncated,
    }
    if verbose:
        print(f"      [PBT] {n_mixed_pre_lp} mixed (after IA) → "
              f"{n_newly_dead} new-dead + {n_newly_active} new-active "
              f"in {n_lps} LPs (mean {stats['mean_lp_ms']:.0f}ms / "
              f"max {stats['max_lp_ms']:.0f}ms), "
              f"{t_total:.1f}s total")
    return stats


def count_mixed_binaries(relu_groups: dict) -> int:
    """Count true mixed binaries in `relu_groups` (those whose z var is not
    pinned LB == UB)."""
    n = 0
    for gdict in relu_groups.values():
        for zv in gdict['z_vars'].values():
            if float(zv.LB) != float(zv.UB):
                n += 1
    return n
