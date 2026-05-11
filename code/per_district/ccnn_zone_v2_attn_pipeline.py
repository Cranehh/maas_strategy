"""Zone-level v2 pipeline (3-obj, 50 pretrain epochs, ABM-dominated finetune).

Phase C of PD-5zones v2 plan (2026-05-08).

Key changes vs `ccnn_zone_pipeline.py`:
  - **3-obj output** (no gini): y_dim = 27 = 9 inter + 6 stages × 3 obj
  - **Pretrain reduced to 50 epochs** (warm-start only)
  - **Finetune dominates with 800 epochs**
  - Uses _v2 data (per-obj proper aggregation: adoption mean, revenue/carbon raw_sum)
  - obj_scale = (N_STAGE_OBJ × Z) / 3.0 = 6 × Z (vs 4-obj's 6 × Z too — same since both denominators scale)

Inputs (from regen_zone_data_v2.py):
  data/zone_pretrain_v2.npz       (theta_dyn, y27, condition, zone_id, zone_pop)
  data/zone_train_n1000_v2.npz    (theta_dyn, cum_obj_zone (N,Z,3), stage_cum_obj_zone (N,Z,K,3), cum_obj_city (N,3))
  data/zone_test_holdout50_v2.npz (same schema)

Output ckpts: checkpoints/ccnn_zone_v2_seed{2026,2027,2028}.pt
"""
from __future__ import annotations
import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader  # noqa: E402
import ch1_model  # noqa: E402
import ch2_model  # noqa: E402
import ch3_model  # noqa: E402
from ccnn_pipeline import DYN_LOWER, DYN_UPPER, STAGE_WEEKS  # noqa: E402

from analytical_pd import PerDistrictAnalyticalSurrogate
from labels_pd import build_stagewise_label_pd
from zone_aggregator import aggregate_per_district_to_zone
from regen_zone_data_v2 import aggregate_pd_obj_3obj
from ccnn_pd_v3_attn_model import (
    ContextConditionedCNNPDv3Attn as ContextConditionedCNNPDv3,
    CCNNPDv3Trainer, CCNNPDv3Predictor, CCNNPDv3Normalizer,
)


# 3-obj schema
N_INTER = 9
K = len(STAGE_WEEKS)
N_OBJ = 3                       # ← 3 not 4
N_STAGE_OBJ = K * N_OBJ          # = 18
Y_DIM = N_INTER + N_STAGE_OBJ    # = 27 ← was 33
IDX_OBJ_START = N_INTER          # = 9


def _mape(p, t):
    p = np.asarray(p, dtype=np.float64).flatten()
    t = np.asarray(t, dtype=np.float64).flatten()
    denom = float(np.mean(np.abs(t)))
    if denom < 1e-12:
        return 0.0
    return 100.0 * float(np.mean(np.abs(p - t)) / denom)


def build_stagewise_label_zone_3obj(surr_pd, theta_dyn_zone, zone_id, district_pop,
                                      softmax_revenue=True):
    """Build (Z, 27) zone-level analytical labels in 3-obj schema.

    Layout:
      [0:9]   = 9 inter, pop-weighted MEAN aggregation
      [9:27]  = 6 stages × 3 obj (adoption, revenue, carbon) — per-obj aggregation rules
    """
    Kz, Zsz, _ = theta_dyn_zone.shape
    D = len(zone_id)
    # Broadcast zone θ to per-district
    theta_dyn_pd = np.zeros((Kz, D, 12), dtype=np.float64)
    for d in range(D):
        theta_dyn_pd[:, d, :] = theta_dyn_zone[:, zone_id[d], :]

    y_pd = build_stagewise_label_pd(
        surr_pd, theta_dyn_pd, softmax_revenue=softmax_revenue)  # (D, 33)

    # 9 inter: pop-weighted mean
    inter_pd = y_pd[:, 0:9]  # (D, 9)
    inter_zone = aggregate_per_district_to_zone(
        inter_pd[None, :, :], zone_id, district_pop, method='mean')[0]  # (Z, 9)

    # 6 stages × 4 obj per stage; convert to 3 obj per stage with per-obj rules
    Z = inter_zone.shape[0]
    obj_per_stage = np.zeros((Z, Kz, 3), dtype=np.float64)
    for k in range(Kz):
        obj_pd_k = y_pd[:, 9 + k * 4 : 9 + (k + 1) * 4]  # (D, 4) [adoption, revenue, gini, carbon]
        obj_zone_k = aggregate_pd_obj_3obj(
            obj_pd_k[None, :, :], zone_id, district_pop)[0]  # (Z, 3)
        obj_per_stage[:, k, :] = obj_zone_k
    obj_flat = obj_per_stage.reshape(Z, Kz * 3)  # (Z, 18)

    return np.concatenate([inter_zone, obj_flat], axis=-1)  # (Z, 27)


def train_and_eval(pretrain_path, train_path, test_path,
                   zone_mapping_path,
                   checkpoint_path, report_path,
                   pretrain_epochs=50, finetune_epochs=800,
                   decoder_first_epochs=200, seed=2026,
                   decoder_dropout=0.05,
                   l1_lambda=0.0, stability_lambda=0.0,
                   stability_scale=0.5):
    print(f"[CC-CNN-Zone v2-attn (3-obj + cross-zone attn)] seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load aggregated data
    pre = np.load(pretrain_path)
    tr = np.load(train_path)
    te = np.load(test_path)
    zm = np.load(zone_mapping_path)
    zone_id = zm['zone_id']
    district_pop = zm['district_pop'].astype(np.float64)
    zone_pop = zm['zone_pop'].astype(np.float64)

    pretrain_theta_z = pre['theta_dyn'].astype(np.float64)  # (N, K, Z, 12)
    pretrain_y_z = pre['y27'].astype(np.float64)            # (N, Z, 27)
    condition = pre['condition'].astype(np.float64)

    train_theta_z = tr['theta_dyn'].astype(np.float64)
    train_stage_cum_z = tr['stage_cum_obj_zone'].astype(np.float64)  # (N, Z, K, 3)
    train_cum_obj_z = tr['cum_obj_zone'].astype(np.float64)          # (N, Z, 3)
    train_cum_obj_city = tr['cum_obj_city'].astype(np.float64)       # (N, 3)

    test_theta_z = te['theta_dyn'].astype(np.float64)
    test_stage_cum_z = te['stage_cum_obj_zone'].astype(np.float64)
    test_cum_obj_z = te['cum_obj_zone'].astype(np.float64)
    test_cum_obj_city = te['cum_obj_city'].astype(np.float64)

    Z = pretrain_theta_z.shape[2]
    print(f"  Z = {Z} zones, y_dim = {Y_DIM} (9 inter + 6×3 obj)")
    print(f"  pretrain: theta {pretrain_theta_z.shape}, y27 {pretrain_y_z.shape}")
    print(f"  train:    theta {train_theta_z.shape}, stage_cum {train_stage_cum_z.shape}")
    print(f"  test:     theta {test_theta_z.shape}, stage_cum {test_stage_cum_z.shape}")

    # Build mixed train labels (analytical inter + ABM 3-obj at zone level)
    print("  Loading agents + analytical_pd for TRAINING-TIME zone labels...")
    agents = data_loader.create_agent_population()
    surr_pd = PerDistrictAnalyticalSurrogate(
        agents, ch3_model, ch1_model, ch2_model)

    print("  Building mixed zone labels (3-obj)...")
    train_y_z = np.zeros(
        (train_theta_z.shape[0], Z, Y_DIM), dtype=np.float64)
    t0 = time.time()
    for i in range(train_theta_z.shape[0]):
        train_y_z[i] = build_stagewise_label_zone_3obj(
            surr_pd, train_theta_z[i], zone_id, district_pop,
            softmax_revenue=True)
        # Replace [9:27] with ABM stage cum_obj per zone (3-obj per stage)
        # train_stage_cum_z[i] shape (Z, K, 3) → reshape (Z, K*3=18)
        train_y_z[i, :, IDX_OBJ_START:] = (
            train_stage_cum_z[i].reshape(Z, K * N_OBJ))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{train_theta_z.shape[0]} ({time.time()-t0:.1f}s)")

    # Normalizer
    normalizer = CCNNPDv3Normalizer(
        DYN_LOWER, DYN_UPPER, D=Z, y_dim=Y_DIM, idx_obj_start=IDX_OBJ_START)
    merged_y_z = np.concatenate([pretrain_y_z, train_y_z], axis=0)
    normalizer.fit(merged_y_z)
    print(f"  normalizer fitted on {merged_y_z.shape[0]} merged samples")

    # Model (v3 with D=Z, y_dim=27)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}")
    model = ContextConditionedCNNPDv3(
        D=Z, y_dim=Y_DIM, idx_obj_start=IDX_OBJ_START, K=K,
        decoder_dropout=decoder_dropout)
    print(f"  parameters: {model.num_parameters():,}")
    if l1_lambda > 0 or stability_lambda > 0:
        print(f"  robust-train: l1_lambda={l1_lambda:.1e}, "
              f"stability_lambda={stability_lambda:.1e}, "
              f"stability_scale={stability_scale}")
    trainer = CCNNPDv3Trainer(model, normalizer, lr=1e-3, device=device,
                                l1_lambda=l1_lambda,
                                stability_lambda=stability_lambda,
                                stability_scale=stability_scale)

    obj_scale = (N_STAGE_OBJ * Z) / 3.0  # = 6 × Z = 30
    w_obj_pretrain = 0.5 * obj_scale
    w_obj_finetune = 1.0 * obj_scale
    print(f"  obj_scale = {obj_scale}")

    # Pretrain (only 50 epochs — warm-start)
    t0 = time.time()
    trainer.pretrain(pretrain_theta_z, pretrain_y_z, condition,
                     epochs=pretrain_epochs, batch_size=64,
                     w_inter=1.0, w_obj=w_obj_pretrain, w_kl=0.1)
    print(f"  pretrain ({pretrain_epochs} epochs) done in {time.time()-t0:.1f}s")

    # Finetune (800 epochs, ABM-dominated)
    t0 = time.time()
    trainer.finetune(train_theta_z, train_y_z, condition,
                     epochs=finetune_epochs,
                     decoder_first_epochs=decoder_first_epochs,
                     w_inter=0.3, w_obj=w_obj_finetune, w_kl=0.05)
    print(f"  finetune ({finetune_epochs} epochs) done in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    trainer.save(checkpoint_path)
    print(f"  ckpt saved {checkpoint_path}")

    # Eval — per-zone per-stage MAPE (Goal 3)
    model.eval()
    predictor = CCNNPDv3Predictor(
        model, normalizer, train_theta_z, train_y_z, condition, device=device)

    print("  Predicting on test...")
    pred_y_z = np.zeros((test_theta_z.shape[0], Z, Y_DIM), dtype=np.float64)
    for i in range(test_theta_z.shape[0]):
        mu, _ = predictor.predict(test_theta_z[i:i+1])
        pred_y_z[i] = mu[0]

    # Per-zone per-stage MAPE (3 obj)
    pred_stage = pred_y_z[..., IDX_OBJ_START:].reshape(-1, Z, K, N_OBJ)
    truth_stage = test_stage_cum_z

    obj_names = ['Adoption', 'Revenue', 'Carbon']
    per_z_per_stage = np.zeros((Z, K, N_OBJ))
    for z in range(Z):
        for k in range(K):
            for j in range(N_OBJ):
                per_z_per_stage[z, k, j] = _mape(
                    pred_stage[:, z, k, j], truth_stage[:, z, k, j])

    pop_w = zone_pop / max(zone_pop.sum(), 1e-12)
    per_stage_mean = (per_z_per_stage * pop_w[:, None, None]).sum(axis=0)
    overall_mean = float(per_stage_mean.mean())
    terminal_mean = float(per_stage_mean[-1].mean())

    # City-level MAPE: aggregate predicted zone obj to city, compare to ABM city truth
    # Use per-obj aggregation rules (mirroring data v2):
    # adoption: Σ_z (zone_pop[z]/total_pop) × y_zone[z, terminal_idx_adoption]
    # revenue: Σ_z y_zone[z, terminal_idx_revenue]  (raw sum, since zone revenue is already zone-internal sum)
    # carbon: Σ_z y_zone[z, terminal_idx_carbon]   (same)
    pred_terminal_3obj = pred_stage[:, :, -1, :]  # (N_te, Z, 3)
    pop_share = zone_pop / max(zone_pop.sum(), 1e-12)

    city_pred = np.zeros((test_theta_z.shape[0], 3), dtype=np.float64)
    city_pred[:, 0] = (pred_terminal_3obj[:, :, 0] * pop_share[None, :]).sum(axis=1)  # adoption
    city_pred[:, 1] = pred_terminal_3obj[:, :, 1].sum(axis=1)                         # revenue
    city_pred[:, 2] = pred_terminal_3obj[:, :, 2].sum(axis=1)                         # carbon

    city_mape = np.array([_mape(city_pred[:, j], test_cum_obj_city[:, j]) for j in range(3)])
    city_overall = float(city_mape.mean())

    print(f"\n  === RESULTS ===")
    print(f"  Per-zone per-stage 3-obj MAPE: 6-stage mean {overall_mean:.2f}%, "
          f"k=5 terminal {terminal_mean:.2f}%")
    print(f"  City-level 3-obj MAPE: Adoption {city_mape[0]:.2f}%, "
          f"Revenue {city_mape[1]:.2f}%, Carbon {city_mape[2]:.2f}%, "
          f"Overall {city_overall:.2f}%")

    pred_npz = checkpoint_path.replace('.pt', '_predictions.npz')
    np.savez(pred_npz,
             pred_y_z=pred_y_z.astype(np.float32),
             pred_terminal_3obj=pred_terminal_3obj.astype(np.float32),
             city_pred=city_pred.astype(np.float32),
             test_stage_cum_z=test_stage_cum_z.astype(np.float32),
             test_cum_obj_city=test_cum_obj_city.astype(np.float32),
             test_theta_z=test_theta_z.astype(np.float32))

    # Report
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# CC-CNN-Zone v2-attn seed={seed} (3-obj, ABM-dominated, with cross-zone attn + LN)\n\n")
        f.write(f"## Setup\n")
        f.write(f"- Architecture: v3 model with D=Z={Z}, y_dim=27 (3-obj)\n")
        f.write(f"- Parameters: {model.num_parameters():,}\n")
        f.write(f"- Pretrain: {pretrain_theta_z.shape[0]} samples × {pretrain_epochs} epochs (warm-start)\n")
        f.write(f"- Finetune: {train_theta_z.shape[0]} ABM × {finetune_epochs} epochs (decoder-first {decoder_first_epochs})\n")
        f.write(f"- Test: {test_theta_z.shape[0]} held-out\n\n")

        f.write("## Per-zone Per-stage MAPE (3-obj, pop-weighted across zones)\n\n")
        f.write("| Stage k | Weeks | Adoption | Revenue | Carbon | Mean |\n")
        f.write("|---|---|---|---|---|---|\n")
        cum_w = 0
        for k in range(K):
            cum_w += STAGE_WEEKS[k]
            row_mean = float(per_stage_mean[k].mean())
            f.write(f"| k={k} | {cum_w} | "
                    f"{per_stage_mean[k, 0]:.2f}% | {per_stage_mean[k, 1]:.2f}% | "
                    f"{per_stage_mean[k, 2]:.2f}% | "
                    f"**{row_mean:.2f}%** |\n")
        f.write(f"| **6-stage mean** | — | — | — | — | **{overall_mean:.2f}%** |\n\n")

        f.write("## City-level Terminal MAPE (3-obj, vs ABM city truth)\n\n")
        f.write("| Objective | MAPE |\n|---|---|\n")
        for j, name in enumerate(obj_names):
            f.write(f"| {name} | {city_mape[j]:.2f}% |\n")
        f.write(f"| **Overall** | **{city_overall:.2f}%** |\n\n")

        f.write("## Goal evaluation\n")
        f.write(f"- Goal 3 (DL ≤ 15%): per-zone 6-stage "
                f"{'✅' if overall_mean <= 15.0 else '❌'} ({overall_mean:.2f}%); "
                f"city-level "
                f"{'✅' if city_overall <= 15.0 else '❌'} ({city_overall:.2f}%)\n")

    print(f"  Report saved to {report_path}")
    return {
        'overall_mape': overall_mean,
        'terminal_mape': terminal_mean,
        'city_mape': city_overall,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train-eval')
    p.add_argument('--pretrain', default='data/zone_pretrain_v2.npz')
    p.add_argument('--train', default='data/zone_train_n1000_v2.npz')
    p.add_argument('--test', default='data/zone_test_holdout50_v2.npz')
    p.add_argument('--zone-mapping', default='data/zone_mapping.npz')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--pretrain-epochs', type=int, default=50)
    p.add_argument('--finetune-epochs', type=int, default=800)
    p.add_argument('--decoder-first-epochs', type=int, default=200)
    p.add_argument('--decoder-dropout', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=2026)
    # Robust-training (L1 sparsity + ReLU stability) — 2026-05-11 plan
    p.add_argument('--l1-lambda', type=float, default=0.0,
                    help='L1 penalty on conv/decoder weights (Tjeng App.H sparsity). '
                         'Recommended 5e-5 for robust retraining; 0 = disabled.')
    p.add_argument('--stability-lambda', type=float, default=0.0,
                    help='Stability penalty mean(exp(-|x|/scale)) on ReLU pre-activations. '
                         'Recommended 0.05 for robust retraining; 0 = disabled.')
    p.add_argument('--stability-scale', type=float, default=0.5,
                    help='Scale of the exp(-|x|/scale) kernel (normalized activation space).')
    args = ap.parse_args()

    if args.cmd == 'train-eval':
        train_and_eval(
            args.pretrain, args.train, args.test, args.zone_mapping,
            args.ckpt, args.report,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            decoder_first_epochs=args.decoder_first_epochs,
            decoder_dropout=args.decoder_dropout,
            seed=args.seed,
            l1_lambda=args.l1_lambda,
            stability_lambda=args.stability_lambda,
            stability_scale=args.stability_scale,
        )


if __name__ == '__main__':
    main()
