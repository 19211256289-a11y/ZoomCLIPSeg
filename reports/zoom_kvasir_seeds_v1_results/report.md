# zoom_kvasir_seeds_v1: locked two-pass zoom, evaluation only

The locked two-pass zoom configuration applied to the new Kvasir seeds 669 and 670 (all four arms) of formal_detail_mix_kvasir_v2, so that every row of the main table has five seeds. Evaluation only.

Seeds 669, 670; means over seeds; contrast is two-pass minus single-pass on the same checkpoints.
Fixed sentence: "a polyp in a colonoscopy image". Configs: t15 (target fraction 0.1500). Primary config: t15.

**Verdict: pass** (arm detail, config t15): macro OOD DSC gain +1.98 pp (per seed +2.57, +1.39)

## DSC by arm, config t15 (primary)

| arm | Kvasir single | Kvasir two-pass | gain | ColonDB single | ColonDB two-pass | gain | ClinicDB single | ClinicDB two-pass | gain | CVC300 single | CVC300 two-pass | gain | BKAI single | BKAI two-pass | gain | macro OOD gain | per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 88.65 | 89.37 | +0.72 | 69.27 | 71.26 | +2.00 | 80.84 | 83.16 | +2.33 | 80.27 | 82.69 | +2.43 | 75.33 | 79.73 | +4.40 | +2.79 | +3.00, +2.57 |
| global_mix | 88.60 | 89.03 | +0.43 | 69.50 | 71.37 | +1.87 | 81.35 | 81.86 | +0.51 | 80.35 | 82.85 | +2.50 | 76.24 | 78.69 | +2.46 | +1.83 | +1.53, +2.13 |
| detail | 89.66 | 89.91 | +0.25 | 71.39 | 72.58 | +1.20 | 83.64 | 85.05 | +1.41 | 80.98 | 82.41 | +1.43 | 75.92 | 79.79 | +3.87 | +1.98 | +2.57, +1.39 |
| detail_global_mix | 89.44 | 89.54 | +0.10 | 71.18 | 72.26 | +1.08 | 82.87 | 84.00 | +1.13 | 80.84 | 83.18 | +2.33 | 77.34 | 80.04 | +2.70 | +1.81 | +1.44, +2.18 |

## Arm baseline, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.72 | +0.54, +0.91 | +0.23 to +1.35 | 0.0055 | 26 / 15 | +0.63 | +0.72 | 22 | +2.28 | 2.5 -> 2.5 | 0.0 -> 0.0 | 38% |
| ColonDB | 360 | +2.00 | +2.26, +1.73 | +0.90 to +3.15 | 0.0003 | 178 / 76 | +2.04 | +2.00 | 241 | +3.15 | 80.5 -> 75.5 | 0.0 -> 0.0 | 70% |
| ClinicDB | 61 | +2.33 | +2.60, +2.06 | +0.49 to +4.15 | 0.0160 | 39 / 10 | +2.04 | +2.33 | 24 | +2.54 | 6.5 -> 6.0 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +2.43 | +2.85, +2.00 | -0.36 to +4.93 | 0.0797 | 46 / 11 | +2.30 | +2.43 | 56 | +2.58 | 6.5 -> 7.0 | 0.0 -> 0.0 | 92% |
| BKAI | 100 | +4.40 | +4.30, +4.50 | +2.67 to +6.31 | 0.0000 | 57 / 9 | +4.31 | +4.40 | 71 | +5.88 | 18.5 -> 14.0 | 0.0 -> 0.0 | 62% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 23.02 | 34.45 | 69% | 43.0 -> 33.5 |
| 1-3% | 179 | 72.97 | 78.02 | 85% | 30.5 -> 26.0 |
| 3-6% | 158 | 82.87 | 81.91 | 89% | 13.0 -> 17.0 |
| 6-15% | 129 | 85.01 | 85.54 | 56% | 6.5 -> 7.0 |
| >15% | 60 | 63.82 | 63.74 | 18% | 19.0 -> 19.0 |

## Arm global_mix, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.43 | +0.37, +0.49 | -0.10 to +1.02 | 0.1409 | 25 / 18 | +0.34 | +0.43 | 22 | +2.03 | 2.5 -> 2.5 | 0.0 -> 0.0 | 39% |
| ColonDB | 360 | +1.87 | +2.23, +1.51 | +0.94 to +2.86 | 0.0001 | 181 / 75 | +1.83 | +1.87 | 241 | +3.10 | 81.5 -> 72.5 | 0.0 -> 0.0 | 69% |
| ClinicDB | 61 | +0.51 | +1.07, -0.06 | -1.92 to +3.03 | 0.6995 | 28 / 18 | +0.20 | +0.51 | 24 | -0.65 | 6.0 -> 6.0 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +2.50 | +1.75, +3.25 | +0.76 to +4.12 | 0.0056 | 42 / 14 | +2.35 | +2.50 | 56 | +2.73 | 7.5 -> 7.5 | 0.0 -> 0.0 | 92% |
| BKAI | 100 | +2.46 | +1.08, +3.84 | +0.80 to +4.21 | 0.0040 | 52 / 16 | +2.33 | +2.46 | 71 | +3.08 | 16.5 -> 14.5 | 0.0 -> 0.0 | 64% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 25.85 | 34.04 | 65% | 43.5 -> 34.5 |
| 1-3% | 179 | 72.90 | 76.64 | 87% | 31.0 -> 27.0 |
| 3-6% | 158 | 82.55 | 82.45 | 89% | 14.0 -> 15.5 |
| 6-15% | 129 | 84.93 | 85.21 | 55% | 5.5 -> 5.5 |
| >15% | 60 | 65.96 | 65.29 | 14% | 17.5 -> 18.0 |

## Arm detail, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.25 | +0.18, +0.32 | -0.27 to +0.85 | 0.3943 | 25 / 16 | +0.18 | +0.25 | 22 | +1.89 | 3.0 -> 2.5 | 0.0 -> 0.0 | 38% |
| ColonDB | 360 | +1.20 | +1.49, +0.91 | +0.36 to +2.07 | 0.0057 | 178 / 85 | +1.17 | +1.20 | 241 | +2.03 | 72.0 -> 66.0 | 0.0 -> 0.0 | 73% |
| ClinicDB | 61 | +1.41 | +1.50, +1.33 | +0.19 to +2.60 | 0.0257 | 34 / 14 | +1.21 | +1.41 | 24 | +1.59 | 5.0 -> 3.5 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +1.43 | +2.90, -0.05 | -1.12 to +3.67 | 0.2643 | 42 / 16 | +1.26 | +1.43 | 56 | +1.74 | 6.5 -> 7.5 | 0.0 -> 0.0 | 93% |
| BKAI | 100 | +3.87 | +4.38, +3.36 | +2.14 to +6.15 | 0.0000 | 57 / 10 | +3.79 | +3.87 | 71 | +5.27 | 17.0 -> 14.0 | 0.0 -> 0.0 | 64% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 27.29 | 35.97 | 72% | 40.5 -> 35.0 |
| 1-3% | 179 | 74.91 | 78.22 | 90% | 26.0 -> 23.0 |
| 3-6% | 158 | 83.32 | 82.86 | 89% | 13.5 -> 13.5 |
| 6-15% | 129 | 86.42 | 86.40 | 58% | 6.0 -> 4.5 |
| >15% | 60 | 67.13 | 67.05 | 18% | 14.5 -> 15.0 |

## Arm detail_global_mix, config t15: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | +0.10 | +0.06, +0.14 | -0.36 to +0.63 | 0.7006 | 25 / 15 | +0.07 | +0.10 | 22 | +0.23 | 2.5 -> 2.5 | 0.0 -> 0.0 | 38% |
| ColonDB | 360 | +1.08 | +1.42, +0.73 | +0.16 to +2.04 | 0.0259 | 167 / 106 | +1.17 | +1.08 | 241 | +1.84 | 69.5 -> 62.5 | 0.0 -> 0.0 | 73% |
| ClinicDB | 61 | +1.13 | +1.02, +1.24 | -0.27 to +2.80 | 0.1676 | 29 / 20 | +1.16 | +1.13 | 24 | +2.27 | 6.0 -> 4.0 | 0.0 -> 0.0 | 75% |
| CVC300 | 60 | +2.33 | +1.60, +3.07 | +0.72 to +3.89 | 0.0057 | 40 / 18 | +2.25 | +2.33 | 56 | +2.56 | 6.0 -> 6.5 | 0.0 -> 0.0 | 93% |
| BKAI | 100 | +2.70 | +1.72, +3.67 | +1.26 to +4.24 | 0.0002 | 55 / 13 | +2.58 | +2.70 | 71 | +3.60 | 16.5 -> 14.0 | 0.0 -> 0.0 | 64% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 30.03 | 34.97 | 73% | 39.5 -> 32.0 |
| 1-3% | 179 | 74.93 | 78.56 | 88% | 24.5 -> 20.5 |
| 3-6% | 158 | 83.39 | 83.23 | 91% | 10.5 -> 12.5 |
| 6-15% | 129 | 85.29 | 85.45 | 57% | 6.0 -> 6.0 |
| >15% | 60 | 67.07 | 66.20 | 21% | 17.5 -> 16.0 |

## Gate (fixed in advance)

- g1 macro OOD gain +1.98 pp (need >= +1.0): True
- g2 OOD sets with a positive gain 4/4 (need >= 3): True
- g3 source Test change +0.25 pp (need >= -0.2): True
- g4 largest regression on any set +0.25 pp (need >= -0.5): True
- g5 every seed positive on the macro OOD gain: True

Single-pass results reproduce the accepted evaluation within 0.0296 pp.
Disclosures: Same test sets as the three-seed confirmation; this checks new training seeds, not new data.
