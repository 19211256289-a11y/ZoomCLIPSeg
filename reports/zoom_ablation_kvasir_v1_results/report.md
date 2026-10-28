# zoom_ablation_kvasir_v1: locked two-pass zoom, evaluation only

Ablation. The two-pass zoom configuration confirmed on the detail arm (test_zoom_kvasir_v1) is applied, unchanged, to the other three accepted Kvasir arms. Evaluation only; nothing is selected or tuned.

Seeds 666, 667, 668; means over seeds; contrast is two-pass minus single-pass on the same checkpoints.
Fixed sentence: "a polyp in a colonoscopy image". Configs: t15 (target fraction 0.1500). Primary config: t15.

**No gate: descriptive ablation.**

## DSC by arm, config t15 (primary)

| arm | Kvasir single | Kvasir two-pass | gain | ColonDB single | ColonDB two-pass | gain | ClinicDB single | ClinicDB two-pass | gain | CVC300 single | CVC300 two-pass | gain | BKAI single | BKAI two-pass | gain | macro OOD gain | per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 88.93 | 89.63 | +0.70 | 69.72 | 70.75 | +1.03 | 81.86 | 83.34 | +1.48 | 79.17 | 81.72 | +2.54 | 74.39 | 78.60 | +4.22 | +2.32 | +1.61, +2.69, +2.65 |
| global_mix | 88.85 | 89.50 | +0.66 | 69.45 | 71.41 | +1.96 | 82.39 | 83.99 | +1.59 | 79.07 | 83.28 | +4.21 | 75.56 | 79.71 | +4.14 | +2.98 | +2.95, +3.38, +2.60 |
| detail_global_mix | 89.72 | 89.94 | +0.22 | 70.61 | 71.61 | +1.01 | 83.41 | 83.81 | +0.40 | 80.19 | 81.57 | +1.38 | 76.81 | 80.22 | +3.41 | +1.55 | +2.53, +1.24, +0.88 |

## Arm baseline, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.70 | +0.81, +0.73, +0.55 | +0.15 to +1.33 | 0.0170 | 29 / 14 | +0.59 | +0.70 | 22 | +2.29 | 2.0 -> 1.7 | 0.0 -> 0.0 | 40% |
| ColonDB | 360 | +1.03 | +1.01, +0.75, +1.34 | +0.08 to +1.96 | 0.0285 | 176 / 92 | +1.03 | +1.03 | 241 | +1.90 | 74.0 -> 74.0 | 0.0 -> 0.0 | 67% |
| ClinicDB | 61 | +1.48 | -0.57, +2.61, +2.40 | +0.10 to +2.92 | 0.0448 | 35 / 14 | +1.22 | +1.48 | 24 | +2.07 | 4.7 -> 5.0 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +2.54 | +1.94, +2.67, +3.02 | -0.91 to +5.24 | 0.1030 | 49 / 9 | +2.47 | +2.54 | 56 | +3.09 | 6.3 -> 6.7 | 0.0 -> 0.0 | 90% |
| BKAI | 100 | +4.22 | +4.07, +4.74, +3.83 | +2.60 to +6.07 | 0.0000 | 60 / 12 | +4.13 | +4.22 | 71 | +5.74 | 20.0 -> 16.0 | 0.0 -> 0.0 | 62% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 22.89 | 31.02 | 65% | 42.7 -> 36.7 |
| 1-3% | 179 | 72.18 | 75.42 | 85% | 30.7 -> 28.0 |
| 3-6% | 158 | 82.56 | 82.95 | 88% | 11.0 -> 15.0 |
| 6-15% | 129 | 85.00 | 84.85 | 53% | 5.3 -> 6.3 |
| >15% | 60 | 68.21 | 67.65 | 16% | 15.3 -> 15.7 |

## Arm global_mix, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.66 | +0.87, +0.55, +0.55 | +0.16 to +1.25 | 0.0124 | 28 / 15 | +0.58 | +0.66 | 22 | +2.63 | 2.7 -> 2.7 | 0.0 -> 0.0 | 37% |
| ColonDB | 360 | +1.96 | +2.30, +0.99, +2.58 | +1.14 to +2.79 | 0.0000 | 181 / 85 | +1.94 | +1.96 | 241 | +2.98 | 74.0 -> 68.3 | 0.0 -> 0.0 | 66% |
| ClinicDB | 61 | +1.59 | +0.74, +3.82, +0.22 | +0.17 to +3.08 | 0.0353 | 35 / 14 | +1.27 | +1.59 | 24 | +2.24 | 4.3 -> 4.0 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +4.21 | +4.73, +4.26, +3.63 | +1.47 to +6.81 | 0.0022 | 50 / 9 | +4.14 | +4.21 | 56 | +4.57 | 6.3 -> 5.0 | 0.0 -> 0.0 | 91% |
| BKAI | 100 | +4.14 | +4.03, +4.44, +3.95 | +2.57 to +5.84 | 0.0000 | 58 / 10 | +4.07 | +4.14 | 71 | +5.62 | 16.3 -> 15.0 | 0.0 -> 0.0 | 64% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 22.18 | 32.16 | 67% | 42.3 -> 35.7 |
| 1-3% | 179 | 71.88 | 76.45 | 84% | 29.7 -> 26.0 |
| 3-6% | 158 | 83.61 | 83.99 | 89% | 9.3 -> 11.3 |
| 6-15% | 129 | 85.23 | 85.60 | 52% | 3.7 -> 4.0 |
| >15% | 60 | 67.30 | 67.20 | 12% | 16.0 -> 15.3 |

## Arm detail_global_mix, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.22 | +0.13, +0.43, +0.09 | -0.33 to +0.84 | 0.4927 | 23 / 18 | +0.16 | +0.22 | 22 | +0.89 | 1.3 -> 1.3 | 0.0 -> 0.0 | 39% |
| ColonDB | 360 | +1.01 | +2.07, +1.07, -0.12 | +0.12 to +1.95 | 0.0290 | 175 / 100 | +1.04 | +1.01 | 241 | +1.71 | 69.0 -> 68.7 | 0.0 -> 0.0 | 71% |
| ClinicDB | 61 | +0.40 | +0.80, -0.47, +0.86 | -0.93 to +1.70 | 0.5715 | 32 / 19 | +0.25 | +0.40 | 24 | +0.14 | 4.7 -> 4.7 | 0.0 -> 0.0 | 76% |
| CVC300 | 60 | +1.38 | +3.61, +1.11, -0.57 | -1.02 to +3.49 | 0.2434 | 42 / 16 | +1.24 | +1.38 | 56 | +1.55 | 5.7 -> 7.0 | 0.0 -> 0.0 | 91% |
| BKAI | 100 | +3.41 | +3.65, +3.24, +3.36 | +1.76 to +5.34 | 0.0000 | 58 / 12 | +3.34 | +3.41 | 71 | +4.65 | 17.0 -> 14.0 | 0.0 -> 0.0 | 64% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 25.52 | 33.50 | 72% | 41.0 -> 36.0 |
| 1-3% | 179 | 74.09 | 76.73 | 87% | 27.7 -> 25.3 |
| 3-6% | 158 | 83.46 | 82.95 | 91% | 8.7 -> 13.3 |
| 6-15% | 129 | 85.62 | 85.91 | 56% | 4.0 -> 4.3 |
| >15% | 60 | 68.35 | 67.38 | 15% | 15.0 -> 15.3 |

Single-pass results reproduce the accepted evaluation within 0.0014 pp.
Disclosures: The size-prior weakness that motivated the method was quantified on Test/OOD predictions of these checkpoints. The five test sets were used before; ColonDB and CVC300 share 52 exact image hashes.
