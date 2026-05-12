"""Phase 1 (2026-05-12 plan): sweep MIP encoding over TR widths to test the
hypothesis that the timeout is driven by input-box size.

For each ckpt {first_mip seed2026, robust-v1 seed2026/2027/2028}:
  - encode bigm + IBP at TR widths {20%, 10%, 5%} centered at theta_c
  - report binaries + per-layer stable fractions
  - for any config with binaries < 3000, run solve_exact(1800s)

Run: LD_PRELOAD=... python code/tests/test_mip_trwidth_sweep.py
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'code'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'per_district'))
import numpy as np
import torch
from per_district.ccnn_pd_v3_attn_model import ContextConditionedCNNPDv3Attn
from per_district.ccnn_pd_v3_model import CCNNPDv3Normalizer
from mip_encoder_grb import MIPEncoderGRB
from gurobipy import GRB
import gurobipy as gp


CKPTS = [
    ('first_mip_2026', 'checkpoints/ccnn_zone_v2_attn_seed2026.pt'),
    ('robust_v1_2026', 'checkpoints/ccnn_zone_v2_attn_robust_seed2026.pt'),
    ('robust_v1_2027', 'checkpoints/ccnn_zone_v2_attn_robust_seed2027.pt'),
    ('robust_v1_2028', 'checkpoints/ccnn_zone_v2_attn_robust_seed2028.pt'),
]
TR_WIDTHS = [0.20, 0.10, 0.05]   # fractional width (full-range fractions)
SOLVE_BINARY_THRESHOLD = 3000
SOLVE_TIME_LIMIT = 1800


def load_ckpt_env(ckpt_rel):
    ckpt = torch.load(os.path.join(ROOT, ckpt_rel), map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    D, y_dim, K = ckpt['D'], ckpt['y_dim'], 6
    nz = CCNNPDv3Normalizer(
        ckpt['theta_lower'], ckpt['theta_upper'],
        D=D, y_dim=y_dim, idx_obj_start=ckpt['idx_obj_start'])
    nz.y_mean = ckpt['y_mean']; nz.y_std = ckpt['y_std']; nz.fitted = True
    data = np.load(os.path.join(ROOT, 'data/zone_train_n1000_v2.npz'))
    rng = np.random.RandomState(0)
    M = 20
    ctx_idx = rng.choice(1000, M, replace=False)
    context_theta = data['theta_dyn'][ctx_idx].astype(np.float64)
    context_y = np.zeros((M, D, y_dim), dtype=np.float64)
    context_y[..., 9:] = data['stage_cum_obj_zone'][ctx_idx].astype(np.float64).reshape(M, D, K*3)
    condition = data['condition'].astype(np.float64)
    tr_center_pd = np.tile(
        ckpt['theta_lower'] + 0.5 * (ckpt['theta_upper'] - ckpt['theta_lower']),
        (K, D, 1)).astype(np.float64)
    return ckpt, sd, nz, condition, context_theta, context_y, tr_center_pd, D, y_dim, K


def encode_at_width(ckpt, sd, nz, condition, context_theta, context_y, tr_center_pd, D, K, width_frac):
    theta_lo = ckpt['theta_lower']; theta_hi = ckpt['theta_upper']
    half = 0.5 * width_frac * (theta_hi - theta_lo)
    center12 = tr_center_pd[0, 0]
    tr_lo_12 = np.clip(center12 - half, theta_lo, theta_hi)
    tr_hi_12 = np.clip(center12 + half, theta_lo, theta_hi)
    tr_lower_pd = np.tile(tr_lo_12, (K, D, 1)).astype(np.float64)
    tr_upper_pd = np.tile(tr_hi_12, (K, D, 1)).astype(np.float64)
    enc = MIPEncoderGRB()
    t0 = time.time()
    m, info = enc.encode_pd_v3_attn(
        sd, theta_lo, theta_hi, condition, context_theta, context_y,
        tr_center_pd, nz, tr_lower_pd=tr_lower_pd, tr_upper_pd=tr_upper_pd,
        relu_mode='bigm', pbt_budget_per_layer=0, verbose=False)
    return enc, m, info, time.time() - t0, tr_lower_pd, tr_upper_pd


def conv_binaries(info):
    """Sum mixed binaries in conv layers (L1r/L2r/L3r/Prr)."""
    n = 0
    for ln, st in info['relu_stats'].items():
        if ln.startswith(('L1r', 'L2r', 'L3r', 'Prr')):
            n += st.get('n_mixed', 0)
    return n


def try_solve(enc, m, info, ckpt, tr_lower_pd, tr_upper_pd, K, D, y_dim):
    """Restrict theta_n vars to the TR box, set sum-of-mu objective, solve_exact."""
    theta_lo = ckpt['theta_lower']; theta_hi = ckpt['theta_upper']
    for k in range(K):
        for d in range(D):
            for c in range(12):
                rng = theta_hi[c] - theta_lo[c]
                if rng < 1e-12: continue
                lo_n = (tr_lower_pd[k, d, c] - theta_lo[c]) / rng
                hi_n = (tr_upper_pd[k, d, c] - theta_lo[c]) / rng
                v = info['theta_n_vars'][k, d, c]
                v.LB = max(float(v.LB), float(lo_n))
                v.UB = min(float(v.UB), float(hi_n))
    expr = gp.LinExpr()
    for d in range(D):
        for i in range(y_dim):
            expr.add(info['mu_vars_per_zone'][d][i], 1.0)
    m.setObjective(expr, GRB.MINIMIZE)
    try:
        status, t_solve, obj_val = enc.solve_exact(
            m, time_limit=SOLVE_TIME_LIMIT, mip_gap=1e-4, threads=8, verbose=False)
        return 'OPTIMAL', t_solve, obj_val
    except RuntimeError as e:
        err = str(e); t_solve = -1
        if 'after' in err:
            try: t_solve = float(err.split('after')[1].split('s')[0])
            except Exception: pass
        return 'TIMEOUT', t_solve, None


def main():
    results = []
    for tag, ckpt_rel in CKPTS:
        if not os.path.exists(os.path.join(ROOT, ckpt_rel)):
            print(f"[skip] {tag}: ckpt missing {ckpt_rel}", flush=True)
            continue
        print(f"\n========== {tag} ==========", flush=True)
        ckpt, sd, nz, condition, context_theta, context_y, tr_center_pd, D, y_dim, K = load_ckpt_env(ckpt_rel)
        for width in TR_WIDTHS:
            enc, m, info, t_enc, tr_lo, tr_hi = encode_at_width(
                ckpt, sd, nz, condition, context_theta, context_y, tr_center_pd, D, K, width)
            n_bin = info['binaries']; n_conv = conv_binaries(info)
            print(f"  TR={width*100:.0f}%: encode {t_enc:.1f}s, binaries={n_bin} (conv={n_conv})", flush=True)
            entry = {'ckpt': tag, 'width': width, 'binaries': n_bin, 'conv_bin': n_conv,
                     'solve_status': '-', 'solve_time': -1}
            if n_bin < SOLVE_BINARY_THRESHOLD:
                print(f"    binaries < {SOLVE_BINARY_THRESHOLD} → solving (1800s budget)...", flush=True)
                st, t_solve, obj = try_solve(enc, m, info, ckpt, tr_lo, tr_hi, K, D, y_dim)
                print(f"    SOLVE: {st}, t={t_solve:.1f}s, obj={obj}", flush=True)
                entry['solve_status'] = st; entry['solve_time'] = t_solve
            results.append(entry)
            del m, info, enc

    print("\n========== SUMMARY ==========")
    print(f'{"ckpt":>16} {"TR%":>6} {"binaries":>10} {"conv_bin":>10} {"solve":>10} {"t_solve":>10}')
    for e in results:
        print(f'{e["ckpt"]:>16} {e["width"]*100:>5.0f}% {e["binaries"]:>10} {e["conv_bin"]:>10} {e["solve_status"]:>10} {e["solve_time"]:>10.1f}')


if __name__ == '__main__':
    main()
