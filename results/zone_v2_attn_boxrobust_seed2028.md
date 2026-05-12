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
| k=0 | 26 | 12.76% | 20.39% | 22.57% | **18.57%** |
| k=1 | 52 | 11.47% | 17.83% | 15.94% | **15.08%** |
| k=2 | 78 | 13.27% | 19.26% | 16.72% | **16.41%** |
| k=3 | 104 | 13.08% | 17.41% | 14.43% | **14.98%** |
| k=4 | 130 | 12.08% | 18.63% | 12.90% | **14.54%** |
| k=5 | 156 | 12.35% | 17.30% | 10.79% | **13.48%** |
| **6-stage mean** | — | — | — | — | **15.51%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 6.93% |
| Revenue | 10.23% |
| Carbon | 6.54% |
| **Overall** | **7.90%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ❌ (15.51%); city-level ✅ (7.90%)
