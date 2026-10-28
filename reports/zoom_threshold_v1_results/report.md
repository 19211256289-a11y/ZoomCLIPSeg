# zoom_threshold_v1: sensitivity of SCZ to the binarisation threshold (detail-branch model, five seeds)

Nothing was trained. The locked method (threshold 0.5) is unchanged whatever these numbers show. ETIS-Larib is not touched.

| threshold | single pass: Kvasir / OOD | SCZ: Kvasir / OOD | second pass (OOD) | SCZ gain, OOD mean [95% CI], p | per seed |
|---|---|---|---|---|---|
| 0.3 | 89.56 / 76.97 | 90.09 / 79.31 | 74% | +2.35 [+1.55, +3.09], p=0.0000 | +2.76, +2.12, +2.02, +2.83, +2.01 |
| 0.4 | 89.57 / 77.18 | 90.10 / 79.49 | 74% | +2.31 [+1.47, +3.12], p=0.0000 | +2.36, +2.01, +2.14, +2.92, +2.09 |
| 0.5 | 89.54 / 77.36 | 89.92 / 79.48 | 75% | +2.13 [+1.31, +2.93], p=0.0000 | +2.37, +2.05, +2.25, +2.57, +1.40 |
| 0.6 | 89.49 / 77.50 | 89.84 / 79.40 | 76% | +1.90 [+1.07, +2.67], p=0.0000 | +1.75, +2.17, +1.99, +2.16, +1.42 |
| 0.7 | 89.39 / 77.62 | 89.63 / 79.42 | 77% | +1.79 [+1.03, +2.57], p=0.0000 | +2.10, +2.39, +1.78, +1.68, +1.01 |

Reproduction: scz_t50 minus the accepted t15, OOD mean +0.01 [-0.00, +0.02], p=0.3570 (tolerance 0.25 pp): **True**

## Verdict by the rule fixed in advance

- The gain of SCZ is robust to the threshold if scz_t minus single_t (OOD mean) is positive with a 95% interval that excludes zero at every threshold in 0.3 to 0.7: **True**
- thresholds with a positive mean gain: 30, 40, 50, 60, 70; with the interval above zero: 30, 40, 50, 60, 70

## Change against threshold 0.5 (OOD mean, pp)

| threshold | single pass | SCZ |
|---|---|---|
| 0.3 | -0.39 [-0.49, -0.28], p=0.0000 | -0.17 [-0.47, +0.11], p=0.2621 |
| 0.4 | -0.17 [-0.22, -0.12], p=0.0000 | +0.01 [-0.19, +0.21], p=0.9565 |
| 0.6 | +0.15 [+0.10, +0.20], p=0.0000 | -0.08 [-0.31, +0.13], p=0.4635 |
| 0.7 | +0.27 [+0.16, +0.37], p=0.0000 | -0.07 [-0.40, +0.25], p=0.6971 |

Per test set, DSC (%):

| configuration | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single_t30 | 89.56 | 70.51 | 83.01 | 79.14 | 75.21 |
| single_t40 | 89.57 | 70.64 | 83.06 | 79.45 | 75.59 |
| single_t50 | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| single_t60 | 89.49 | 70.77 | 83.05 | 79.99 | 76.20 |
| single_t70 | 89.39 | 70.78 | 83.00 | 80.23 | 76.48 |
| scz_t30 | 90.09 | 72.02 | 84.14 | 82.15 | 78.94 |
| scz_t40 | 90.10 | 72.10 | 84.01 | 82.39 | 79.46 |
| scz_t50 | 89.92 | 71.95 | 84.16 | 82.13 | 79.70 |
| scz_t60 | 89.84 | 71.82 | 84.30 | 82.10 | 79.38 |
| scz_t70 | 89.63 | 71.93 | 84.11 | 82.34 | 79.29 |
| t15 | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |

## Definitions

- **single_t**: mean probability map of 30 stochastic passes of the whole image, computed exactly as in the accepted evaluation, binarised at t (single_t50 is the accepted single pass and serves as the reproduction check)
- **scz_t**: the locked two-pass rule with threshold t in the window rule, in the fallback (first prediction kept if the second map has no pixel above t) and in the final mask; the second run draws 30 new stochastic passes for every t
- **t15**: the locked method (threshold 0.5); read from the accepted results; scz_t50 must reproduce it within the tolerance
- **metrics**: DSC and NSD as in the accepted evaluation; Brier scores are not computed at shifted thresholds

Consequence fixed in advance: The manuscript reports the verdict and the gain at every threshold as they are. If the rule fails, it states at which thresholds the gain is not demonstrated. No threshold of this suite replaces 0.5 in the locked method.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated.
- ETIS-Larib is not touched.
- The thresholds share one first pass per image; the second pass is drawn anew for every threshold.
