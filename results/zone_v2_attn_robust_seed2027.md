# CC-CNN-Zone v2-attn seed=2027 (3-obj, ABM-dominated, with cross-zone attn + LN)

## Setup
- Architecture: v3 model with D=Z=5, y_dim=27 (3-obj)
- Parameters: 582,926
- Pretrain: 2000 samples × 50 epochs (warm-start)
- Finetune: 1000 ABM × 800 epochs (decoder-first 200)
- Test: 50 held-out

## Per-zone Per-stage MAPE (3-obj, pop-weighted across zones)

| Stage k | Weeks | Adoption | Revenue | Carbon | Mean |
|---|---|---|---|---|---|
| k=0 | 26 | 12.03% | 19.29% | 22.89% | **18.07%** |
| k=1 | 52 | 11.62% | 16.62% | 15.56% | **14.60%** |
| k=2 | 78 | 12.84% | 18.35% | 16.33% | **15.84%** |
| k=3 | 104 | 12.19% | 16.37% | 14.03% | **14.20%** |
| k=4 | 130 | 11.15% | 16.74% | 13.11% | **13.67%** |
| k=5 | 156 | 11.29% | 17.31% | 10.95% | **13.18%** |
| **6-stage mean** | — | — | — | — | **14.93%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 6.29% |
| Revenue | 9.76% |
| Carbon | 6.64% |
| **Overall** | **7.56%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ✅ (14.93%); city-level ✅ (7.56%)
