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
| k=0 | 26 | 12.10% | 18.78% | 23.02% | **17.97%** |
| k=1 | 52 | 11.63% | 18.16% | 15.82% | **15.20%** |
| k=2 | 78 | 12.14% | 19.25% | 15.94% | **15.77%** |
| k=3 | 104 | 12.59% | 17.06% | 14.19% | **14.61%** |
| k=4 | 130 | 10.80% | 17.06% | 13.01% | **13.62%** |
| k=5 | 156 | 11.24% | 16.71% | 10.54% | **12.83%** |
| **6-stage mean** | — | — | — | — | **15.00%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 6.49% |
| Revenue | 9.47% |
| Carbon | 6.23% |
| **Overall** | **7.40%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ❌ (15.00%); city-level ✅ (7.40%)
