"""Tests for `mip_encoder_grb.MIPEncoderGRB` exact-MIP path (Tjeng/Xiao/Tedrake
ICLR 2019, MIPVerify-style big-M + sound IBP + LP-PBT).

Test matrix:
  1. test_bigm_indicator_match_torch_at_center
       Build MIP with relu_mode ∈ {'indicator', 'bigm'}, pin θ to TR center
       via equality, call solve_exact. Compare μ to PyTorch forward.
  2. test_sound_bounds_only_smoke
       Build MIP with sound_bounds_only=True (no sample bounds, IBP only).
       At TR center, must still match PyTorch forward.
  3. test_pbt_bounds_monotonic
       Build MIP with pbt_budget_per_layer=64. Verify post-PBT pre-activation
       bounds are tighter than pre-PBT (monotonicity).
  4. test_pbt_reduces_binaries
       Same as 3, but verify post-PBT n_bin ≤ pre-PBT n_bin.
  5. test_solve_exact_raises_on_failure
       Build infeasible setting (manually invert bounds) → solve_exact raises.

All tests skip cleanly if the Model B ckpt or zone data isn't present.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'per_district'))


# ------------------------------------------------------------------ Fixtures
@pytest.fixture(scope='module')
def env():
    """Shared fixture: load Model B + small context + θ_c at TR center."""
    try:
        import gurobipy as gp  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy not installed")

    from per_district.ccnn_pd_v3_attn_model import ContextConditionedCNNPDv3Attn
    from per_district.ccnn_pd_v3_model import CCNNPDv3Normalizer

    ckpt_path = os.path.join(ROOT, 'checkpoints/ccnn_zone_v2_attn_seed2026.pt')
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Model B ckpt not found: {ckpt_path}")
    data_path = os.path.join(ROOT, 'data/zone_train_n1000_v2.npz')
    if not os.path.exists(data_path):
        pytest.skip(f"Zone train data not found: {data_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    D, y_dim, K = ckpt['D'], ckpt['y_dim'], 6

    model = ContextConditionedCNNPDv3Attn(
        D=D, y_dim=y_dim, K=K, decoder_dropout=0.05)
    model.load_state_dict(sd)
    model.eval()

    normalizer = CCNNPDv3Normalizer(
        ckpt['theta_lower'], ckpt['theta_upper'],
        D=D, y_dim=y_dim, idx_obj_start=ckpt['idx_obj_start'])
    normalizer.y_mean = ckpt['y_mean']
    normalizer.y_std = ckpt['y_std']
    normalizer.fitted = True

    data = np.load(data_path)
    np.random.seed(0)
    M = 20
    ctx_idx = np.random.choice(1000, M, replace=False)
    context_theta = data['theta_dyn'][ctx_idx].astype(np.float64)
    context_y = np.zeros((M, D, y_dim), dtype=np.float64)
    context_y[..., 9:] = data['stage_cum_obj_zone'][ctx_idx].astype(
        np.float64).reshape(M, D, K * 3)
    condition = data['condition'].astype(np.float64)

    tr_center_pd = np.tile(
        ckpt['theta_lower'] + 0.5 * (ckpt['theta_upper'] - ckpt['theta_lower']),
        (K, D, 1)).astype(np.float64)

    return {
        'model': model, 'sd': sd, 'normalizer': normalizer,
        'condition': condition, 'context_theta': context_theta,
        'context_y': context_y, 'tr_center_pd': tr_center_pd,
        'D': D, 'K': K, 'y_dim': y_dim,
        'theta_lower_12': ckpt['theta_lower'],
        'theta_upper_12': ckpt['theta_upper'],
    }


def _torch_forward_at(env, target_theta_pd):
    """Run torch forward at given target. Returns mu_norm (D, y_dim)."""
    model = env['model']
    nz = env['normalizer']
    cond_t = torch.from_numpy(env['condition'].astype(np.float32)).unsqueeze(0)
    ctx_t = torch.from_numpy(
        nz.normalize_theta_dyn(env['context_theta']).astype(np.float32)
    ).unsqueeze(0)
    ctx_y = torch.from_numpy(
        nz.normalize_y(env['context_y']).astype(np.float32)
    ).unsqueeze(0)
    tgt_t = torch.from_numpy(
        nz.normalize_theta_dyn(target_theta_pd[None]).astype(np.float32)
    ).unsqueeze(0)
    with torch.no_grad():
        out = model(ctx_t, ctx_y, tgt_t, cond_t)
    return out['mu'].squeeze(0).squeeze(0).cpu().numpy()  # (D, y_dim)


def _pin_theta_to_center(m, info, env):
    """Fix θ_n vars to normalized TR center via equality constraints."""
    import gurobipy as gp
    theta_lower_12 = info['theta_lower_12']
    theta_upper_12 = info['theta_upper_12']
    rng = theta_upper_12 - theta_lower_12
    rng = np.where(rng < 1e-12, 1.0, rng)
    K, D = info['K'], info['D']
    tr_c = env['tr_center_pd']
    theta_n_target = (tr_c - theta_lower_12) / rng
    for k in range(K):
        for d in range(D):
            for c in range(12):
                v = info['theta_n_vars'][k, d, c]
                m.addConstr(v == float(theta_n_target[k, d, c]),
                              name=f'pin_th_n_{k}_{d}_{c}')
    m.update()


def _build_mip(env, relu_mode='bigm', sound_bounds_only=False,
                pbt_budget_per_layer=0, verbose=False):
    """Build MIP for env. Returns (m, info, encoder)."""
    from mip_encoder_grb import MIPEncoderGRB
    enc = MIPEncoderGRB()
    m, info = enc.encode_pd_v3_attn(
        env['sd'],
        env['theta_lower_12'], env['theta_upper_12'],
        env['condition'], env['context_theta'], env['context_y'],
        env['tr_center_pd'], env['normalizer'],
        relu_mode=relu_mode,
        sound_bounds_only=sound_bounds_only,
        pbt_budget_per_layer=pbt_budget_per_layer,
        n_sample_bounds=300,
        verbose=verbose)
    return m, info, enc


# ------------------------------------------------------------------ Tests
@pytest.mark.parametrize("relu_mode", ['indicator', 'bigm'])
def test_relu_mode_matches_torch_at_center(env, relu_mode):
    """At θ_c, MIP output must match torch forward to rtol 1e-3 in both modes."""
    import gurobipy as gp
    from gurobipy import GRB

    print(f"\n[test] relu_mode={relu_mode}")
    t0 = time.time()
    m, info, enc = _build_mip(env, relu_mode=relu_mode,
                                sound_bounds_only=False,
                                pbt_budget_per_layer=0, verbose=False)
    print(f"  encode time: {time.time() - t0:.1f}s, binaries: {info['binaries']}")

    _pin_theta_to_center(m, info, env)

    # Trivial objective (model is feasibility-equivalent once θ is pinned)
    first_mu = info['mu_vars_per_zone'][0][0]
    m.setObjective(first_mu, GRB.MINIMIZE)

    print("  Solving (θ pinned → presolve trivial)...")
    status, t_solve, obj_val = enc.solve_exact(
        m, time_limit=120, mip_gap=1e-4, verbose=False)
    print(f"  status={status}, t_solve={t_solve:.1f}s")
    assert status == GRB.OPTIMAL

    # Extract μ_MIP (normalized)
    D, y_dim = info['D'], info['y_dim']
    mu_mip_norm = np.zeros((D, y_dim))
    for d in range(D):
        for i, v in enumerate(info['mu_vars_per_zone'][d]):
            mu_mip_norm[d, i] = float(v.X)

    mu_torch_norm = _torch_forward_at(env, env['tr_center_pd'])
    diff = np.abs(mu_mip_norm - mu_torch_norm)
    rel = diff / (np.abs(mu_torch_norm) + 1e-6)
    print(f"  diff: max={diff.max():.3e}, mean={diff.mean():.3e}; "
          f"rel: max={rel.max():.3e}")
    assert rel.max() < 1e-3, \
        f"MIP/NN mismatch in {relu_mode!r}: max rel diff = {rel.max():.3e}"


def test_sound_bounds_only_matches_torch_at_center(env):
    """IBP-only (no sample bounds) must still produce correct MIP at θ_c."""
    import gurobipy as gp
    from gurobipy import GRB

    print("\n[test] sound_bounds_only=True (IBP only)")
    t0 = time.time()
    m, info, enc = _build_mip(env, relu_mode='bigm',
                                sound_bounds_only=True,
                                pbt_budget_per_layer=0, verbose=False)
    print(f"  encode time: {time.time() - t0:.1f}s, binaries: {info['binaries']}")
    assert info['sound_bounds_only'] is True

    _pin_theta_to_center(m, info, env)
    first_mu = info['mu_vars_per_zone'][0][0]
    m.setObjective(first_mu, GRB.MINIMIZE)

    status, t_solve, obj_val = enc.solve_exact(
        m, time_limit=180, mip_gap=1e-4, verbose=False)
    assert status == GRB.OPTIMAL

    D, y_dim = info['D'], info['y_dim']
    mu_mip_norm = np.zeros((D, y_dim))
    for d in range(D):
        for i, v in enumerate(info['mu_vars_per_zone'][d]):
            mu_mip_norm[d, i] = float(v.X)

    mu_torch_norm = _torch_forward_at(env, env['tr_center_pd'])
    rel = np.abs(mu_mip_norm - mu_torch_norm) / (np.abs(mu_torch_norm) + 1e-6)
    print(f"  max rel diff = {rel.max():.3e}")
    assert rel.max() < 1e-3


def test_pbt_bounds_monotonic(env):
    """Post-PBT pre-activation bounds must be a subset of pre-PBT bounds."""
    print("\n[test] PBT bounds monotonicity")

    # Build twice: once without PBT (to record IBP bounds), once with PBT
    t0 = time.time()
    m_noPBT, info_noPBT, _ = _build_mip(
        env, relu_mode='bigm', sound_bounds_only=True,
        pbt_budget_per_layer=0, verbose=False)
    print(f"  no-PBT encode: {time.time() - t0:.1f}s, binaries: {info_noPBT['binaries']}")

    # Record pre-PBT bounds per ReLU pre-activation var
    pre_bounds = {}
    for layer_name, gdict in info_noPBT['relu_groups'].items():
        pre_bounds[layer_name] = [
            (float(xv.LB), float(xv.UB)) for xv in gdict['x_vars']]

    t0 = time.time()
    m_PBT, info_PBT, _ = _build_mip(
        env, relu_mode='bigm', sound_bounds_only=True,
        pbt_budget_per_layer=64, verbose=True)
    print(f"  PBT encode: {time.time() - t0:.1f}s, binaries: {info_PBT['binaries']}")
    assert info_PBT['pbt_stats'] is not None

    # Verify monotonicity per layer
    eps = 1e-6
    n_total = 0
    n_tightened = 0
    for layer_name, gdict_PBT in info_PBT['relu_groups'].items():
        for i, xv_PBT in enumerate(gdict_PBT['x_vars']):
            pre_lo, pre_hi = pre_bounds[layer_name][i]
            post_lo = float(xv_PBT.LB)
            post_hi = float(xv_PBT.UB)
            assert post_lo >= pre_lo - eps, \
                f"{layer_name}[{i}]: post LB {post_lo} < pre LB {pre_lo}"
            assert post_hi <= pre_hi + eps, \
                f"{layer_name}[{i}]: post UB {post_hi} > pre UB {pre_hi}"
            if post_lo > pre_lo + eps or post_hi < pre_hi - eps:
                n_tightened += 1
            n_total += 1
    print(f"  PBT tightened {n_tightened} / {n_total} pre-activation bounds "
          f"({100.0*n_tightened/max(1,n_total):.1f}%)")
    # We should see SOME tightening
    assert n_tightened > 0


def test_pbt_reduces_binaries(env):
    """Post-PBT n_bin should be <= pre-PBT n_bin."""
    print("\n[test] PBT binary reduction")
    _, info_noPBT, _ = _build_mip(
        env, relu_mode='bigm', sound_bounds_only=True,
        pbt_budget_per_layer=0, verbose=False)
    _, info_PBT, _ = _build_mip(
        env, relu_mode='bigm', sound_bounds_only=True,
        pbt_budget_per_layer=64, verbose=False)
    print(f"  no-PBT binaries: {info_noPBT['binaries']}")
    print(f"  PBT binaries:    {info_PBT['binaries']}")
    print(f"  PBT eliminated dead/active: "
          f"{info_PBT['pbt_stats']['total_newly_stable_dead']} dead + "
          f"{info_PBT['pbt_stats']['total_newly_stable_active']} active")
    assert info_PBT['binaries'] <= info_noPBT['binaries']


if __name__ == '__main__':
    # Run as: python code/tests/test_mip_exact.py
    import warnings
    warnings.filterwarnings('ignore')

    env_dict = None  # will be populated below
    from per_district.ccnn_pd_v3_attn_model import ContextConditionedCNNPDv3Attn
    from per_district.ccnn_pd_v3_model import CCNNPDv3Normalizer

    ckpt = torch.load(os.path.join(ROOT, 'checkpoints/ccnn_zone_v2_attn_seed2026.pt'),
                       map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    D, y_dim, K = ckpt['D'], ckpt['y_dim'], 6
    model = ContextConditionedCNNPDv3Attn(D=D, y_dim=y_dim, K=K, decoder_dropout=0.05)
    model.load_state_dict(sd)
    model.eval()
    nz = CCNNPDv3Normalizer(
        ckpt['theta_lower'], ckpt['theta_upper'],
        D=D, y_dim=y_dim, idx_obj_start=ckpt['idx_obj_start'])
    nz.y_mean = ckpt['y_mean']; nz.y_std = ckpt['y_std']; nz.fitted = True
    data = np.load(os.path.join(ROOT, 'data/zone_train_n1000_v2.npz'))
    np.random.seed(0)
    M = 20
    ctx_idx = np.random.choice(1000, M, replace=False)
    env_dict = {
        'model': model, 'sd': sd, 'normalizer': nz,
        'condition': data['condition'].astype(np.float64),
        'context_theta': data['theta_dyn'][ctx_idx].astype(np.float64),
        'context_y': np.zeros((M, D, y_dim), dtype=np.float64),
        'tr_center_pd': np.tile(
            ckpt['theta_lower'] + 0.5 * (ckpt['theta_upper'] - ckpt['theta_lower']),
            (K, D, 1)).astype(np.float64),
        'D': D, 'K': K, 'y_dim': y_dim,
        'theta_lower_12': ckpt['theta_lower'],
        'theta_upper_12': ckpt['theta_upper'],
    }
    env_dict['context_y'][..., 9:] = data['stage_cum_obj_zone'][ctx_idx].astype(
        np.float64).reshape(M, D, K * 3)

    print("\n=== test_relu_mode_matches_torch_at_center bigm ===")
    test_relu_mode_matches_torch_at_center(env_dict, 'bigm')
    print("\n=== test_sound_bounds_only_matches_torch_at_center ===")
    test_sound_bounds_only_matches_torch_at_center(env_dict)
    print("\n=== test_pbt_bounds_monotonic ===")
    test_pbt_bounds_monotonic(env_dict)
    print("\n=== test_pbt_reduces_binaries ===")
    test_pbt_reduces_binaries(env_dict)
    print("\n✓ ALL TESTS PASSED")
