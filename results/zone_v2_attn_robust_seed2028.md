# CC-CNN-Zone v2-attn seed=2028 (3-obj, ABM-dominated, with cross-zone attn + LN)

## Setup
- Architecture: v3 model with D=Z=5, y_dim=27 (3-obj)
- Parameters: 582,926
- Pretrain: 2000 samples × 50 epochs (warm-start)
- Finetune: 1000 ABM × 800 epochs (decoder-first 200)
- Test: 50 held-out

## Per-zone Per-stage MAPE (3-obj, pop-weighted across zones)

| Stage k | Weeks | Adoption | Revenue | Carbon | Mean |
|---|---|---|---|---|---|
| k=0 | 26 | 12.03% | 18.30% | 23.51% | **17.95%** |
| k=1 | 52 | 11.23% | 16.62% | 15.93% | **14.59%** |
| k=2 | 78 | 12.40% | 17.81% | 16.41% | **15.54%** |
| k=3 | 104 | 12.45% | 17.18% | 14.06% | **14.56%** |
| k=4 | 130 | 10.90% | 16.72% | 13.25% | **13.62%** |
| k=5 | 156 | 10.46% | 16.50% | 10.94% | **12.63%** |
| **6-stage mean** | — | — | — | — | **14.82%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 6.27% |
| Revenue | 10.63% |
| Carbon | 6.43% |
| **Overall** | **7.78%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ✅ (14.82%); city-level ✅ (7.78%)
