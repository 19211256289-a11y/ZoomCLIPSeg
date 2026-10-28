# zoom_busi_v1: locked two-pass zoom, evaluation only

Replication on a second dataset and modality. Two-pass zoom applied to the BUSI checkpoints of formal_detail_mix_busi_v1 (three arms, three seeds). Locked while those checkpoints were still training and before any BUSI Test result existed. Evaluation only; nothing is selected or tuned on BUSI.

Seeds 666, 667, 668; means over seeds; contrast is two-pass minus single-pass on the same checkpoints.
Fixed sentence: "a tumor in a breast ultrasound image". Configs: tcal (target fraction 0.0862); t15 (target fraction 0.1500). Primary config: tcal.

Source calibration from Train masks only: 0.15 is the 62.6th percentile of Kvasir Train lesion fractions; the same percentile of the source Train set is 0.0862 (medians: Kvasir 0.1136, source 0.0555).

**Verdict: pass** (arm detail, config tcal): macro OOD DSC gain +1.06 pp (per seed +1.38, +0.41, +1.40)

## DSC by arm, config tcal (primary)

| arm | BUSI single | BUSI two-pass | gain | BUSBRA single | BUSBRA two-pass | gain | BUSUC single | BUSUC two-pass | gain | BUID single | BUID two-pass | gain | UDIAT single | UDIAT two-pass | gain | macro OOD gain | per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 79.61 | 80.35 | +0.74 | 57.71 | 57.57 | -0.14 | 76.56 | 77.19 | +0.63 | 66.34 | 67.93 | +1.59 | 56.78 | 57.16 | +0.38 | +0.62 | +1.17, +1.39, -0.71 |
| detail | 81.04 | 81.24 | +0.21 | 68.94 | 70.01 | +1.06 | 83.58 | 83.73 | +0.16 | 74.63 | 75.24 | +0.62 | 72.95 | 75.36 | +2.41 | +1.06 | +1.38, +0.41, +1.40 |
| detail_global_mix | 80.45 | 80.43 | -0.02 | 68.56 | 68.70 | +0.13 | 82.38 | 82.60 | +0.22 | 72.26 | 72.45 | +0.19 | 67.09 | 70.76 | +3.67 | +1.05 | +0.72, -0.16, +2.59 |

## DSC by arm, config t15 (secondary)

| arm | BUSI single | BUSI two-pass | gain | BUSBRA single | BUSBRA two-pass | gain | BUSUC single | BUSUC two-pass | gain | BUID single | BUID two-pass | gain | UDIAT single | UDIAT two-pass | gain | macro OOD gain | per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 79.61 | 80.42 | +0.82 | 57.71 | 57.31 | -0.40 | 76.56 | 76.57 | +0.01 | 66.34 | 67.42 | +1.08 | 56.78 | 57.63 | +0.85 | +0.39 | -0.39, +1.05, +0.50 |
| detail | 81.04 | 79.69 | -1.34 | 68.94 | 69.36 | +0.41 | 83.58 | 83.46 | -0.11 | 74.63 | 74.27 | -0.36 | 72.95 | 74.67 | +1.72 | +0.42 | +0.72, +0.17, +0.36 |
| detail_global_mix | 80.45 | 80.30 | -0.15 | 68.56 | 68.51 | -0.05 | 82.38 | 81.99 | -0.39 | 72.26 | 71.95 | -0.31 | 67.09 | 70.90 | +3.81 | +0.76 | +1.00, -1.54, +2.83 |

## Arm baseline, config tcal: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BUSI | 78 | +0.74 | +0.56, +1.68, -0.01 | -0.26 to +2.03 | 0.2530 | 19 / 8 | +0.76 | +0.89 | 14 | +1.36 | 9.0 -> 8.0 | 1.3 -> 1.3 | 31% |
| BUSBRA | 282 | -0.14 | +0.98, -2.39, +0.98 | -0.73 to +0.44 | 0.6355 | 59 / 44 | -0.24 | -0.14 | 35 | -1.11 | 91.7 -> 94.7 | 0.0 -> 0.0 | 32% |
| BUSUC | 122 | +0.63 | +0.49, +1.04, +0.36 | +0.10 to +1.29 | 0.0360 | 18 / 11 | +0.76 | +0.63 | 6 | -0.28 | 13.0 -> 11.3 | 0.0 -> 0.0 | 18% |
| BUID | 35 | +1.59 | +0.35, +3.75, +0.68 | +0.26 to +3.31 | 0.0310 | 7 / 1 | +1.59 | +1.59 | 1 | +0.00 | 8.3 -> 7.0 | 0.0 -> 0.0 | 11% |
| UDIAT | 25 | +0.38 | +2.84, +3.18, -4.88 | -2.53 to +3.18 | 0.8058 | 9 / 7 | +0.22 | +0.38 | 8 | +3.60 | 8.7 -> 8.3 | 0.0 -> 0.0 | 49% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 9 | 38.48 | 41.81 | 59% | 4.7 -> 4.7 |
| 1-3% | 71 | 52.34 | 51.07 | 68% | 25.3 -> 28.7 |
| 3-6% | 112 | 53.61 | 54.37 | 46% | 42.7 -> 40.3 |
| 6-15% | 164 | 68.04 | 68.48 | 10% | 32.7 -> 31.7 |
| >15% | 108 | 75.28 | 75.32 | 6% | 16.3 -> 16.0 |

## Arm detail, config tcal: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BUSI | 78 | +0.21 | -0.39, -0.14, +1.15 | -0.79 to +1.28 | 0.7157 | 17 / 11 | +0.19 | +0.25 | 14 | +3.02 | 6.7 -> 6.3 | 3.3 -> 3.3 | 39% |
| BUSBRA | 282 | +1.06 | +1.24, +1.30, +0.64 | +0.29 to +1.87 | 0.0091 | 73 / 49 | +1.00 | +1.06 | 35 | +3.40 | 56.7 -> 53.7 | 0.0 -> 0.0 | 38% |
| BUSUC | 122 | +0.16 | +0.19, -0.08, +0.37 | -0.52 to +0.92 | 0.6892 | 15 / 15 | +0.14 | +0.16 | 6 | +4.30 | 6.7 -> 6.3 | 0.0 -> 0.0 | 23% |
| BUID | 35 | +0.62 | +0.46, +0.36, +1.03 | +0.00 to +1.48 | 0.2515 | 3 / 0 | +0.66 | +0.62 | 1 | +0.00 | 4.7 -> 4.7 | 0.0 -> 0.0 | 11% |
| UDIAT | 25 | +2.41 | +3.63, +0.05, +3.55 | -0.68 to +6.16 | 0.2054 | 13 / 5 | +2.24 | +2.41 | 8 | +2.53 | 4.3 -> 3.7 | 0.0 -> 0.0 | 65% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 9 | 50.77 | 55.71 | 70% | 4.0 -> 3.0 |
| 1-3% | 71 | 67.15 | 69.17 | 88% | 14.0 -> 14.3 |
| 3-6% | 112 | 67.84 | 68.01 | 60% | 24.3 -> 23.7 |
| 6-15% | 164 | 77.64 | 78.74 | 9% | 16.3 -> 14.0 |
| >15% | 108 | 78.88 | 78.99 | 6% | 13.7 -> 13.3 |

## Arm detail_global_mix, config tcal: details

| set | n | gain (pp) | per seed | 95% CI | perm. p | better / worse | NSD gain | lesion-only gain | small n | small gain | failures | normal-image FP | second pass on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BUSI | 78 | -0.02 | +0.77, -0.38, -0.45 | -0.77 to +0.84 | 0.9681 | 14 / 14 | -0.07 | -0.02 | 14 | +1.47 | 6.7 -> 6.7 | 3.0 -> 3.0 | 38% |
| BUSBRA | 282 | +0.13 | +0.23, -0.50, +0.67 | -0.54 to +0.78 | 0.6926 | 62 / 51 | +0.11 | +0.13 | 35 | +1.99 | 56.3 -> 57.3 | 0.0 -> 0.0 | 31% |
| BUSUC | 122 | +0.22 | +0.67, +0.29, -0.31 | -0.31 to +0.78 | 0.4410 | 16 / 14 | +0.23 | +0.22 | 6 | +2.19 | 8.0 -> 7.0 | 0.0 -> 0.0 | 22% |
| BUID | 35 | +0.19 | +0.00, +0.12, +0.44 | -0.59 to +1.14 | 0.6260 | 3 / 3 | +0.22 | +0.19 | 1 | +12.97 | 5.3 -> 5.7 | 0.0 -> 0.0 | 9% |
| UDIAT | 25 | +3.67 | +1.99, -0.55, +9.57 | +0.71 to +6.86 | 0.0277 | 15 / 6 | +3.63 | +3.67 | 8 | +8.23 | 6.3 -> 5.0 | 0.0 -> 0.0 | 63% |

| OOD lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 9 | 40.83 | 50.02 | 67% | 5.3 -> 4.0 |
| 1-3% | 71 | 62.72 | 64.25 | 78% | 17.3 -> 17.3 |
| 3-6% | 112 | 66.27 | 65.72 | 50% | 24.0 -> 25.7 |
| 6-15% | 164 | 77.76 | 77.99 | 7% | 16.7 -> 15.3 |
| >15% | 108 | 79.59 | 79.55 | 5% | 12.7 -> 12.7 |

## Gate (fixed in advance)

- g1 macro OOD gain +1.06 pp (need >= +0.5): True
- g2 OOD sets with a positive gain 4/4 (need >= 3): True
- g3 source Test change +0.21 pp (need >= -0.2): True
- g4 largest regression on any set +0.16 pp (need >= -0.5): True
- g5 every seed positive on the macro OOD gain: True

Single-pass results reproduce the accepted evaluation within 0.0028 pp.
Disclosures: BUID (35 images) and UDIAT (25 images) are small; their per-set gains are noisy. The method and its fixed parameters were developed on polyp data.
