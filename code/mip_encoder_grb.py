"""MIPEncoderGRB — gurobipy-direct encoder for Model B (MIPVerify-style).

Supports two ReLU encodings (selectable via `relu_mode`):

  'indicator': Gurobi GenConstrIndicator constraints. Tight LP relaxation, but
    Gurobi presolve typically expands each indicator into ~4 internal binaries
    (CLAUDE.md 2026-05-11: 3,224 → 12,752 binaries → MIP times out).

  'bigm' (default): Manual triangle big-M (Tjeng, Xiao, Tedrake, ICLR 2019):
        y >= 0
        y >= x
        y <= u·z              (z=0 ⇒ y=0)
        y <= x − l·(1−z)      (z=1 ⇒ y=x)
    With **tight** (l, u) from IBP + LP-PBT, this matches the convex hull
    (Tjeng §3) — equally strong LP relaxation as indicator — but avoids the
    presolve blowup.

The encoder also supports `sound_bounds_only=True` (skips
`compute_pd_sample_bounds`, uses IBP-only sound bounds) for the MIPVerify
exact-solving path, per the 2026-05-11 plan.
"""
from __future__ import annotations
import os
import sys
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


def _propagate_linear_bounds(x_lower, x_upper, W, b):
    W = np.asarray(W); b = np.asarray(b)
    W_pos = np.maximum(W, 0.0); W_neg = np.minimum(W, 0.0)
    y_lower = W_pos @ x_lower + W_neg @ x_upper + b
    y_upper = W_pos @ x_upper + W_neg @ x_lower + b
    return y_lower, y_upper


def _add_linear_layer_grb(m, x_vars, W, b, x_lower, x_upper, layer_name):
    """y = W·x + b. Returns (list of gp.Var, y_lower, y_upper)."""
    W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    out_dim = W.shape[0]
    y_lower, y_upper = _propagate_linear_bounds(x_lower, x_upper, W, b)
    y_vars = []
    for i in range(out_dim):
        yi = m.addVar(lb=float(y_lower[i]), ub=float(y_upper[i]),
                       name=f"{layer_name}_y_{i}")
        expr = gp.quicksum(float(W[i, j]) * x_vars[j] for j in range(W.shape[1])) + float(b[i])
        m.addConstr(yi == expr, name=f"{layer_name}_c_{i}")
        y_vars.append(yi)
    return y_vars, y_lower, y_upper


def _add_relu_layer_grb(m, x_vars, x_lower, x_upper, layer_name,
                          eps_zero=1e-6, mode='bigm'):
    """ReLU layer encoder. `mode` ∈ {'indicator', 'bigm'}.

    Branches per unit on bound status:
      x_upper[i] <= -eps_zero: stable dead   → y_i = 0      (no binary)
      x_lower[i] >=  eps_zero: stable active → y_i = x_i    (no binary)
      else:                    mixed         → add binary z_i + constraints

    'indicator' (legacy):
      z=1 → y = x AND x >= 0      (active)
      z=0 → y = 0 AND x <= 0      (inactive)

    'bigm' (Tjeng/Xiao/Tedrake ICLR 2019 triangle formulation):
      y >= 0, y >= x, y <= u·z, y <= x − l·(1−z)
      With tight (l, u) this is the convex hull of the ReLU disjunction.

    Returns (y_vars, y_lower, y_upper, n_bin, stats):
      stats = {
        'n_mixed', 'n_dead', 'n_active',
        'big_m_max', 'big_m_mean',
        'z_vars': dict[int → gp.Var] (binary var per mixed unit, for PBT lookup),
        'x_vars_ref': list[gp.Var]   (pre-activation refs, for PBT),
      }
    """
    if mode not in ('indicator', 'bigm'):
        raise ValueError(f"_add_relu_layer_grb: unknown mode {mode!r}")
    n = len(x_vars)
    y_lower = np.maximum(x_lower, 0.0)
    y_upper = np.maximum(x_upper, 0.0)
    y_vars = []
    n_mixed = 0
    n_dead = 0
    n_active = 0
    big_m_list = []
    z_vars_map = {}
    for i in range(n):
        xl = float(x_lower[i])
        xu = float(x_upper[i])
        yl = float(y_lower[i])
        yu = float(y_upper[i])
        yi = m.addVar(lb=yl, ub=yu, name=f"{layer_name}_y_{i}")
        y_vars.append(yi)
        if xu <= -eps_zero:
            # Stable dead. yi forced to 0 by bounds (yl=yu=0). No binary.
            n_dead += 1
            continue
        if xl >= eps_zero:
            # Stable active. yi = xi. No binary.
            m.addConstr(yi == x_vars[i], name=f"{layer_name}_relueq_{i}")
            n_active += 1
            continue
        # Mixed (xl < 0 < xu): need binary
        n_mixed += 1
        big_m_list.append(max(abs(xl), abs(xu)))
        if mode == 'indicator':
            m.addConstr(yi >= x_vars[i], name=f"{layer_name}_relubase_{i}")
            zi = m.addVar(vtype=GRB.BINARY, name=f"{layer_name}_z_{i}")
            # z=1 (active): y = x, x >= 0
            m.addGenConstrIndicator(zi, 1, yi - x_vars[i], GRB.EQUAL, 0.0,
                                     name=f"{layer_name}_ind1eq_{i}")
            m.addGenConstrIndicator(zi, 1, x_vars[i], GRB.GREATER_EQUAL, 0.0,
                                     name=f"{layer_name}_ind1geq_{i}")
            # z=0 (inactive): y = 0, x <= 0
            m.addGenConstrIndicator(zi, 0, yi, GRB.EQUAL, 0.0,
                                     name=f"{layer_name}_ind0eq_{i}")
            m.addGenConstrIndicator(zi, 0, x_vars[i], GRB.LESS_EQUAL, 0.0,
                                     name=f"{layer_name}_ind0leq_{i}")
        else:  # 'bigm'
            zi = m.addVar(vtype=GRB.BINARY, name=f"{layer_name}_z_{i}")
            # y >= 0 already enforced via yi.LB = yl = max(0, xl) = 0.
            m.addConstr(yi >= x_vars[i], name=f"{layer_name}_bm_ygex_{i}")
            m.addConstr(yi <= xu * zi, name=f"{layer_name}_bm_yleuz_{i}")
            m.addConstr(yi - x_vars[i] + xl * (1.0 - zi) <= 0.0,
                         name=f"{layer_name}_bm_ylex_{i}")
        z_vars_map[i] = zi
    stats = {
        'n_mixed': n_mixed,
        'n_dead': n_dead,
        'n_active': n_active,
        'n_total': n,
        'big_m_max': float(max(big_m_list)) if big_m_list else 0.0,
        'big_m_mean': float(np.mean(big_m_list)) if big_m_list else 0.0,
        'z_vars': z_vars_map,
        'x_vars_ref': list(x_vars),
    }
    return y_vars, y_lower, y_upper, n_mixed, stats


class MIPEncoderGRB:
    """Gurobi-direct encoder for ContextConditionedCNNPDv3Attn (Model B)."""

    def __init__(self):
        pass

    def encode_pd_v3_attn(self, state_dict, theta_lower_12, theta_upper_12,
                            condition, context_theta_pd, context_y_pd,
                            tr_center_pd, normalizer,
                            tr_lower_pd=None, tr_upper_pd=None,
                            n_sample_bounds=2000,
                            relu_mode='bigm',
                            sound_bounds_only=False,
                            pbt_budget_per_layer=0,
                            pbt_max_seconds_per_layer=30.0,
                            pbt_layer_order=('dec3', 'dec0', 'tqe', 'psa',
                                              'proj', 'conv3', 'conv2', 'conv1'),
                            verbose=False):
        """Build gurobipy MIP encoding Model B at TR center. Returns (model, info).

        Parameters
        ----------
        relu_mode : {'indicator', 'bigm'}
            ReLU encoding. Default 'bigm' (Tjeng/Xiao/Tedrake ICLR 2019 triangle).
        sound_bounds_only : bool
            If True, skip `compute_pd_sample_bounds` (unsound: samples don't
            certify TR corners) and use IBP-only bounds. Required for exact
            MIPVerify-style solving with stability proofs.
        pbt_budget_per_layer : int
            If > 0, run LP-PBT (progressive bounds tightening) after IBP. Tighten
            up to this many mixed ReLUs per layer, ranked by |l|·|u|.
        pbt_layer_order : tuple[str]
            Layer-name prefixes to tighten (in order). Defaults to decoder→conv.
        """
        sys.path.insert(0, os.path.join(ROOT, 'per_district'))
        from per_district.mip_pd_helpers import (
            extract_pd_constants, extract_pd_linearizations,
            cda_affine_map, conv1d_to_linear,
            xa_affine_map,
        )
        from per_district.ccnn_pd_v3_attn_model import ContextConditionedCNNPDv3Attn

        D = int(state_dict['district_embed.weight'].shape[0])
        y_dim = int(state_dict['decoder.mu_head.weight'].shape[0])
        K = 6

        model_pt = ContextConditionedCNNPDv3Attn(
            D=D, y_dim=y_dim, K=K, decoder_dropout=0.05)
        model_pt.load_state_dict(state_dict); model_pt.eval()

        constants = extract_pd_constants(
            model_pt, condition, context_theta_pd, context_y_pd, normalizer)
        lin = extract_pd_linearizations(
            model_pt, tr_center_pd, constants, normalizer)
        if verbose:
            print(f"  [encode_pd_grb] CDA LN diff: {lin['cda']['ln_diff_check']:.2e}")

        if tr_lower_pd is None:
            tr_lower_pd = np.tile(theta_lower_12, (K, D, 1)).astype(np.float64)
        if tr_upper_pd is None:
            tr_upper_pd = np.tile(theta_upper_12, (K, D, 1)).astype(np.float64)
        # NOTE (2026-05-11 evening, post-Tjeng Appendix B refactor):
        # `sound_bounds_only` and `n_sample_bounds` parameters are retained
        # for backwards compatibility but no longer affect behavior. The new
        # layer-incremental flow always uses IA (sound) bounds, optionally
        # tightened by LP-PBT on partial models. Sample bounds (unsound,
        # corner-incomplete) have been removed from the exact path.

        # Layer-incremental PBT helper (Tjeng/Xiao/Tedrake ICLR 2019 Appendix B).
        # IMPORTANT: the encoder now BUILDS the model layer by layer, running
        # LP-PBT on the PARTIAL model after each linear layer is added but
        # BEFORE that layer's ReLU is encoded. This keeps each LP small.
        from per_district.mip_pbt import tighten_layer_via_lp

        # ===== Build Gurobi model (layer-incremental) =====
        m = gp.Model("PDv3Attn_GRB")
        m.setParam('OutputFlag', 1 if verbose else 0)
        n_bin = 0
        relu_groups = {}       # layer_name → {'x_vars', 'z_vars', 'y_vars'}
        relu_stats = {}        # layer_name → ReLU encoding stats
        pbt_stats_per_layer = {}  # layer_name → tighten_layer_via_lp output

        def _encode_relu_layer(name, x_pre_vars, ia_lo, ia_hi, is_layer1=False):
            """Tighten bounds on x_pre_vars via LP-PBT (skip if layer 1 or no
            PBT budget), then encode the ReLU on top. Records pbt_stats and
            updates relu_groups/relu_stats. Returns (y_vars, y_lo, y_hi, n_bin_inc)."""
            nonlocal n_bin
            stats_pbt = tighten_layer_via_lp(
                m, list(x_pre_vars), np.asarray(ia_lo, dtype=np.float64),
                np.asarray(ia_hi, dtype=np.float64),
                eps_stable=1e-6, budget=pbt_budget_per_layer,
                skip_lp=(is_layer1 or pbt_budget_per_layer <= 0),
                max_seconds=pbt_max_seconds_per_layer,
                verbose=verbose)
            pbt_stats_per_layer[name] = stats_pbt
            # After PBT, var LB/UB are tightened in-place. Read back.
            tight_lo = np.array([float(v.LB) for v in x_pre_vars])
            tight_hi = np.array([float(v.UB) for v in x_pre_vars])
            y_vars, y_lo, y_hi, n_bin_inc, st = _add_relu_layer_grb(
                m, x_pre_vars, tight_lo, tight_hi, name, mode=relu_mode)
            relu_groups[name] = {'x_vars': list(x_pre_vars),
                                   'z_vars': st['z_vars'],
                                   'y_vars': list(y_vars)}
            relu_stats[name] = {k: v for k, v in st.items()
                                  if k not in ('z_vars', 'x_vars_ref')}
            n_bin += n_bin_inc
            return y_vars, y_lo, y_hi, n_bin_inc

        # 1. θ_n vars — normalized to [0,1] within the FULL parameter range,
        # but bounded to the TR box. CRITICAL (fixed 2026-05-12): the IBP
        # bound propagation that classifies ReLU stability MUST start from the
        # TR box, not the full [0,1] box — otherwise TR width has no effect on
        # the binary count (which was the bug: 10,917 binaries regardless of
        # whether the TR is 5%, 20% or 100% wide).
        theta_range_12 = np.asarray(theta_upper_12, dtype=np.float64) - \
                         np.asarray(theta_lower_12, dtype=np.float64)
        theta_range_12 = np.where(theta_range_12 < 1e-12, 1.0, theta_range_12)
        # Normalized TR box per (k, d, c)
        theta_n_lo = np.empty((K, D, 12), dtype=np.float64)
        theta_n_hi = np.empty((K, D, 12), dtype=np.float64)
        for k in range(K):
            for d in range(D):
                for c in range(12):
                    lo = (float(tr_lower_pd[k, d, c]) - float(theta_lower_12[c])) / theta_range_12[c]
                    hi = (float(tr_upper_pd[k, d, c]) - float(theta_lower_12[c])) / theta_range_12[c]
                    theta_n_lo[k, d, c] = float(np.clip(lo, 0.0, 1.0))
                    theta_n_hi[k, d, c] = float(np.clip(hi, 0.0, 1.0))
        theta_n_vars = np.empty((K, D, 12), dtype=object)
        for k in range(K):
            for d in range(D):
                for c in range(12):
                    theta_n_vars[k, d, c] = m.addVar(
                        lb=float(theta_n_lo[k, d, c]),
                        ub=float(theta_n_hi[k, d, c]),
                        name=f"th_n_{k}_{d}_{c}")
        m.update()

        # 2. Per-zone CNN. Conv1 uses IA only (Tjeng footnote 4: IA optimal
        # at the input layer because input vars are independent). Conv2/3/proj
        # use IA + LP-PBT on the current partial model.
        Wc1, bc1_ = conv1d_to_linear(
            state_dict['theta_encoder.conv1.weight'].cpu().numpy(),
            state_dict['theta_encoder.conv1.bias'].cpu().numpy(), K=K)
        Wc2, bc2_ = conv1d_to_linear(
            state_dict['theta_encoder.conv2.weight'].cpu().numpy(),
            state_dict['theta_encoder.conv2.bias'].cpu().numpy(), K=K)
        Wc3, bc3_ = conv1d_to_linear(
            state_dict['theta_encoder.conv3.weight'].cpu().numpy(),
            state_dict['theta_encoder.conv3.bias'].cpu().numpy(), K=K)
        Wproj = state_dict['theta_encoder.proj.weight'].cpu().numpy()
        bproj = state_dict['theta_encoder.proj.bias'].cpu().numpy()

        per_d_pre_vars = np.empty(D, dtype=object)
        per_d_pre_lo = np.empty(D, dtype=object)
        per_d_pre_hi = np.empty(D, dtype=object)
        for d in range(D):
            if verbose:
                print(f"  [encode_pd_grb] zone {d}/{D-1}: encoding CNN layers")
            # Conv input order matches ThetaEncoder1DCNN_Shared.forward:
            #   permute(...,3,4,2).reshape(B*M*D, C=12, K) → x0 indexed [c*K + k].
            x0 = []
            x0_lo = np.empty(12 * K, dtype=np.float64)
            x0_hi = np.empty(12 * K, dtype=np.float64)
            for c in range(12):
                for k in range(K):
                    x0.append(theta_n_vars[k, d, c])
                    x0_lo[c * K + k] = theta_n_lo[k, d, c]
                    x0_hi[c * K + k] = theta_n_hi[k, d, c]

            # conv1 — layer 1, IA only (no LP-PBT)
            z1, z1_lo, z1_hi = _add_linear_layer_grb(
                m, x0, Wc1, bc1_, x0_lo, x0_hi, f'L1_z{d}')
            a1, a1_lo, a1_hi, _ = _encode_relu_layer(
                f'L1r_z{d}', z1, z1_lo, z1_hi, is_layer1=True)
            # conv2
            z2, z2_lo, z2_hi = _add_linear_layer_grb(
                m, a1, Wc2, bc2_, a1_lo, a1_hi, f'L2_z{d}')
            a2, a2_lo, a2_hi, _ = _encode_relu_layer(
                f'L2r_z{d}', z2, z2_lo, z2_hi, is_layer1=False)
            # conv3
            z3, z3_lo, z3_hi = _add_linear_layer_grb(
                m, a2, Wc3, bc3_, a2_lo, a2_hi, f'L3_z{d}')
            a3, a3_lo, a3_hi, _ = _encode_relu_layer(
                f'L3r_z{d}', z3, z3_lo, z3_hi, is_layer1=False)
            # proj
            zp, zp_lo, zp_hi = _add_linear_layer_grb(
                m, a3, Wproj, bproj, a3_lo, a3_hi, f'Pr_z{d}')
            ap, ap_lo, ap_hi, _ = _encode_relu_layer(
                f'Prr_z{d}', zp, zp_lo, zp_hi, is_layer1=False)
            per_d_pre_vars[d] = ap
            per_d_pre_lo[d] = ap_lo
            per_d_pre_hi[d] = ap_hi

        # 3. CDA affine map (linear, no ReLU; IA bounds propagate)
        if verbose:
            print(f"  [encode_pd_grb] CDA (linear, no ReLU)")
        E = 128
        flat_dim = D * E
        flat_vars = []
        flat_lo = np.zeros(flat_dim); flat_hi = np.zeros(flat_dim)
        for d in range(D):
            for i in range(E):
                flat_vars.append(per_d_pre_vars[d][i])
                flat_lo[d * E + i] = float(per_d_pre_lo[d][i])
                flat_hi[d * E + i] = float(per_d_pre_hi[d][i])
        A_cda, b_cda = cda_affine_map(flat_dim, lin['cda'], D)
        per_d_post_vars, post_lo, post_hi = _add_linear_layer_grb(
            m, flat_vars, A_cda, b_cda, flat_lo, flat_hi, 'CDA')
        per_d_post = [per_d_post_vars[d * E:(d + 1) * E] for d in range(D)]
        per_d_post_lo = [post_lo[d * E:(d + 1) * E] for d in range(D)]
        per_d_post_hi = [post_hi[d * E:(d + 1) * E] for d in range(D)]

        # 4. per_sample_agg (mean + Linear + ReLU)
        if verbose:
            print(f"  [encode_pd_grb] PSA (per-sample aggregator)")
        A_mean = np.zeros((E, flat_dim))
        for i in range(E):
            for d in range(D):
                A_mean[i, d * E + i] = 1.0 / D
        b_mean = np.zeros(E)
        z_mean_vars, zm_lo, zm_hi = _add_linear_layer_grb(
            m, per_d_post_vars, A_mean, b_mean, post_lo, post_hi, 'AggMean')

        Wpsa = state_dict['per_sample_agg.proj.weight'].cpu().numpy()
        bpsa = state_dict['per_sample_agg.proj.bias'].cpu().numpy()
        zpsa, zpsa_lo, zpsa_hi = _add_linear_layer_grb(
            m, z_mean_vars, Wpsa, bpsa, zm_lo, zm_hi, 'PSA')
        per_sample, ps_lo, ps_hi, _ = _encode_relu_layer(
            'PSAr', zpsa, zpsa_lo, zpsa_hi, is_layer1=False)

        # 5. target_query_encoder
        if verbose:
            print(f"  [encode_pd_grb] TQE")
        c_glob = constants['c_embed_global']
        Wtqe = state_dict['target_query_encoder.0.weight'].cpu().numpy()
        btqe = state_dict['target_query_encoder.0.bias'].cpu().numpy()
        cat_vars = list(per_sample) + [
            m.addVar(lb=float(c_glob[i]), ub=float(c_glob[i]), name=f'cglob_{i}')
            for i in range(16)
        ]
        cat_lo = np.concatenate([ps_lo, c_glob])
        cat_hi = np.concatenate([ps_hi, c_glob])
        ztqe, ztqe_lo, ztqe_hi = _add_linear_layer_grb(
            m, cat_vars, Wtqe, btqe, cat_lo, cat_hi, 'TQE')
        queries, q_lo, q_hi, _ = _encode_relu_layer(
            'TQEr', ztqe, ztqe_lo, ztqe_hi, is_layer1=False)

        # 6. cross-attention: frozen at TR center → constant
        _A_xa, b_xa = xa_affine_map(64, lin['xa'], constants['r_context'])
        attended_r_const = b_xa

        # 7. Decoder per zone (D0r, D3r with PBT)
        Wd0 = state_dict['decoder.backbone.0.weight'].cpu().numpy()
        bd0 = state_dict['decoder.backbone.0.bias'].cpu().numpy()
        Wd3 = state_dict['decoder.backbone.3.weight'].cpu().numpy()
        bd3 = state_dict['decoder.backbone.3.bias'].cpu().numpy()
        Wmu = state_dict['decoder.mu_head.weight'].cpu().numpy()
        bmu = state_dict['decoder.mu_head.bias'].cpu().numpy()
        dist_embed_const = constants['district_embed']
        condition_pd_const = constants['condition_pd']
        z_mean_ctx_const = constants['z_mean_ctx']

        mu_vars_per_zone = {}
        for d in range(D):
            if verbose:
                print(f"  [encode_pd_grb] decoder zone {d}/{D-1}")
            dec_in = list(per_d_post[d]) + [
                m.addVar(lb=float(dist_embed_const[d, i]), ub=float(dist_embed_const[d, i]),
                         name=f'demb_{d}_{i}') for i in range(8)
            ] + [
                m.addVar(lb=float(condition_pd_const[d, i]), ub=float(condition_pd_const[d, i]),
                         name=f'cpd_{d}_{i}') for i in range(6)
            ] + [
                m.addVar(lb=float(attended_r_const[i]), ub=float(attended_r_const[i]),
                         name=f'arc_{d}_{i}') for i in range(64)
            ] + [
                m.addVar(lb=float(z_mean_ctx_const[i]), ub=float(z_mean_ctx_const[i]),
                         name=f'zmc_{d}_{i}') for i in range(32)
            ]
            dec_lo = np.concatenate([per_d_post_lo[d], dist_embed_const[d],
                                       condition_pd_const[d], attended_r_const, z_mean_ctx_const])
            dec_hi = np.concatenate([per_d_post_hi[d], dist_embed_const[d],
                                       condition_pd_const[d], attended_r_const, z_mean_ctx_const])

            zh1, zh1_lo, zh1_hi = _add_linear_layer_grb(
                m, dec_in, Wd0, bd0, dec_lo, dec_hi, f'D0_z{d}')
            h1, h1_lo, h1_hi, _ = _encode_relu_layer(
                f'D0r_z{d}', zh1, zh1_lo, zh1_hi, is_layer1=False)

            zh2, zh2_lo, zh2_hi = _add_linear_layer_grb(
                m, h1, Wd3, bd3, h1_lo, h1_hi, f'D3_z{d}')
            h2, h2_lo, h2_hi, _ = _encode_relu_layer(
                f'D3r_z{d}', zh2, zh2_lo, zh2_hi, is_layer1=False)

            mu_z, _, _ = _add_linear_layer_grb(m, h2, Wmu, bmu, h2_lo, h2_hi, f'Mu_z{d}')
            mu_vars_per_zone[d] = mu_z

        m.update()

        # Aggregate PBT stats across layers for telemetry
        pbt_stats = None
        if pbt_budget_per_layer > 0:
            pbt_stats = {
                'per_layer': pbt_stats_per_layer,
                'total_lps': sum(s.get('n_lps', 0) for s in pbt_stats_per_layer.values()),
                'total_newly_stable_dead': sum(
                    s.get('n_newly_dead', 0) for s in pbt_stats_per_layer.values()),
                'total_newly_stable_active': sum(
                    s.get('n_newly_active', 0) for s in pbt_stats_per_layer.values()),
                'total_tightened': sum(
                    s.get('n_tightened', 0) for s in pbt_stats_per_layer.values()),
                't_seconds_total': sum(
                    s.get('t_seconds', 0.0) for s in pbt_stats_per_layer.values()),
            }
            if verbose:
                print(f"  [encode_pd_grb] PBT total: {pbt_stats['total_lps']} LPs, "
                        f"{pbt_stats['total_newly_stable_dead']} dead + "
                        f"{pbt_stats['total_newly_stable_active']} active eliminated, "
                        f"{pbt_stats['t_seconds_total']:.1f}s")

        info = {
            'binaries': n_bin,
            'continuous_vars': m.NumVars - n_bin,
            'constraints': m.NumConstrs,
            'theta_n_vars': theta_n_vars,
            'mu_vars_per_zone': mu_vars_per_zone,
            'theta_lower_12': np.asarray(theta_lower_12, dtype=np.float64),
            'theta_upper_12': np.asarray(theta_upper_12, dtype=np.float64),
            'tr_center_pd': tr_center_pd,
            'D': D, 'K': K, 'y_dim': y_dim,
            'normalizer_y_mean': normalizer.y_mean.copy(),
            'normalizer_y_std': normalizer.y_std.copy(),
            'relu_mode': relu_mode,
            'sound_bounds_only': sound_bounds_only,
            'relu_groups': relu_groups,
            'relu_stats': relu_stats,
            'pbt_stats': pbt_stats,
            'theta_n_lo': theta_n_lo,   # normalized TR box actually used for IBP
            'theta_n_hi': theta_n_hi,
        }
        return m, info

    def solve_exact(self, m, time_limit=1800, mip_gap=1e-4, mip_focus=1,
                       threads=8, mip_start=None, verbose=True):
        """Solve exactly. Per the 2026-05-11 plan, status MUST be OPTIMAL;
        otherwise raises RuntimeError. No silent LP fallback.

        Parameters
        ----------
        m : gurobipy.Model
            MIP model built via encode_pd_v3_attn.
        time_limit : float
            Hard ceiling (default 1800 s = 30 min, matching Goal 4).
        mip_gap : float
            Required gap (default 1e-4 ≈ exact).
        mip_start : dict | None
            Optional warm start: {var_name: value} pairs. Typically set z_i =
            round(y_i_LP > 0) from an LP relaxation pre-solve.

        Returns (status, elapsed_seconds, obj_val).
        Raises RuntimeError if Status != OPTIMAL (or no incumbent if SUBOPTIMAL).
        """
        if mip_start is not None:
            for vname, val in mip_start.items():
                v = m.getVarByName(vname)
                if v is not None:
                    v.Start = float(val)
            m.update()
        m.setParam('TimeLimit', float(time_limit))
        m.setParam('MIPGap', float(mip_gap))
        m.setParam('MIPFocus', int(mip_focus))
        m.setParam('Threads', int(threads))
        m.setParam('Presolve', 2)
        m.setParam('Cuts', 3)
        m.setParam('Heuristics', 0.5)
        m.setParam('OutputFlag', 1 if verbose else 0)
        t0 = time.time()
        m.optimize()
        elapsed = time.time() - t0
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"solve_exact: Gurobi returned Status={m.Status} "
                f"(expected OPTIMAL={GRB.OPTIMAL}) after {elapsed:.1f}s. "
                f"SolCount={m.SolCount}, MIPGap={getattr(m, 'MIPGap', 'n/a')}. "
                f"Goal-4 budget breach (TimeLimit={time_limit}s).")
        return m.Status, elapsed, m.ObjVal

    def solve(self, m, time_limit=300, mip_gap=0.02, mip_focus=1,
                lp_relaxation=False, verbose=True):
        """Solve. If lp_relaxation=True, relax all binaries to [0,1] continuous
        and solve LP only (much faster, sub-optimal but feasible θ extracted).
        """
        if lp_relaxation:
            # Relax all integer/binary vars to continuous
            for v in m.getVars():
                if v.VType in (GRB.BINARY, GRB.INTEGER):
                    v.VType = GRB.CONTINUOUS
                    if v.LB < 0:
                        v.LB = 0.0
                    if v.UB > 1.0 and v.UB > v.LB:
                        v.UB = 1.0
            # Drop indicator constraints (they require binary controllers)
            # so the LP has only standard linear constraints + ReLU base bounds.
            for gc in m.getGenConstrs():
                m.remove(gc)
            m.update()
        m.setParam('TimeLimit', float(time_limit))
        m.setParam('MIPGap', float(mip_gap))
        m.setParam('MIPFocus', int(mip_focus))
        m.setParam('Presolve', 2)
        m.setParam('Cuts', 3)
        m.setParam('Heuristics', 0.5)
        m.setParam('OutputFlag', 1 if verbose else 0)
        t0 = time.time()
        m.optimize()
        elapsed = time.time() - t0
        ok_status = m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]
        obj = m.ObjVal if (ok_status and m.SolCount > 0) else None
        return m.Status, elapsed, obj
