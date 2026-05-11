# CC-CNN-Zone v2-attn seed=2026 (3-obj, ABM-dominated, with cross-zone attn + LN)

## Setup
- Architecture: v3 model with D=Z=5, y_dim=27 (3-obj)
- Parameters: 582,926
- Pretrain: 2000 samples × 50 epochs (warm-start)
- Finetune: 1000 ABM × 800 epochs (decoder-first 200)
- Test: 50 held-out

## Per-zone Per-stage MAPE (3-obj, pop-weighted across zones)

| Stage k | Weeks | Adoption | Revenue | Carbon | Mean |
|---|---|---|---|---|---|
| k=0 | 26 | 11.92% | 20.12% | 23.37% | **18.47%** |
| k=1 | 52 | 11.72% | 17.30% | 15.71% | **14.91%** |
| k=2 | 78 | 12.29% | 19.05% | 16.12% | **15.82%** |
| k=3 | 104 | 12.53% | 16.69% | 14.23% | **14.48%** |
| k=4 | 130 | 10.78% | 16.36% | 13.02% | **13.39%** |
| k=5 | 156 | 10.81% | 16.62% | 10.57% | **12.67%** |
| **6-stage mean** | — | — | — | — | **14.96%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 6.49% |
| Revenue | 9.68% |
| Carbon | 6.29% |
| **Overall** | **7.49%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ✅ (14.96%); city-level ✅ (7.49%)
