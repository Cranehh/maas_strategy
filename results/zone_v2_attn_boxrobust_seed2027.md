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
| k=0 | 26 | 13.11% | 22.06% | 22.74% | **19.30%** |
| k=1 | 52 | 11.83% | 20.28% | 15.87% | **15.99%** |
| k=2 | 78 | 13.93% | 18.81% | 16.75% | **16.49%** |
| k=3 | 104 | 13.92% | 19.08% | 14.64% | **15.88%** |
| k=4 | 130 | 13.66% | 19.79% | 13.28% | **15.58%** |
| k=5 | 156 | 13.70% | 18.48% | 10.91% | **14.37%** |
| **6-stage mean** | — | — | — | — | **16.27%** |

## City-level Terminal MAPE (3-obj, vs ABM city truth)

| Objective | MAPE |
|---|---|
| Adoption | 7.23% |
| Revenue | 9.87% |
| Carbon | 6.51% |
| **Overall** | **7.87%** |

## Goal evaluation
- Goal 3 (DL ≤ 15%): per-zone 6-stage ❌ (16.27%); city-level ✅ (7.87%)
