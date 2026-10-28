# zoom_box_v2: SCZ window rule against larger fixed crops of the predicted box (detail-branch model, five seeds)

Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.

OOD = equal-weight mean of the four OOD polyp sets. Second pass = share of OOD images that receive a second pass; window = median window side of those images, as a fraction of the image side.

| method | second pass | window | Kvasir DSC | OOD DSC | gain over single pass [95% CI], p | t15 minus method [95% CI], p | verdict |
|---|---|---|---|---|---|---|---|
| single | 0% |  | 89.54 | 77.36 |  | +2.12 [+1.31, +2.90], p=0.0000 |  |
| t15 | 75% | 0.52 | 89.92 | 79.48 | +2.12 [+1.31, +2.91], p=0.0000 |  |  |
| plain250 | 49% | 0.58 | 89.81 | 78.75 | +1.39 [+0.99, +1.80], p=0.0000 | +0.72 [+0.12, +1.32], p=0.0165 | t15_better |
| plain300 | 36% | 0.64 | 89.83 | 78.33 | +0.98 [+0.67, +1.30], p=0.0000 | +1.14 [+0.42, +1.85], p=0.0012 | t15_better |
| plain400 | 12% | 0.68 | 89.63 | 77.54 | +0.18 [+0.03, +0.33], p=0.0155 | +1.94 [+1.12, +2.69], p=0.0000 | t15_better |

**Overall verdict by the rule fixed in advance: rule_not_exceeded.**

Per test set, DSC (%):

| method | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single | 89.54 | 70.71 | 83.08 | 79.73 | 75.91 |
| t15 | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |
| plain250 | 89.81 | 71.39 | 83.50 | 82.33 | 77.78 |
| plain300 | 89.83 | 70.98 | 83.46 | 82.18 | 76.72 |
| plain400 | 89.63 | 70.89 | 83.23 | 79.98 | 76.04 |

t15 minus method, per test set (pp):

| method | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| plain250 | +0.10 [-0.23, +0.47], p=0.5596 | +0.56 [-0.04, +1.20], p=0.0730 | +0.66 [-0.39, +1.75], p=0.2356 | -0.20 [-1.92, +1.27], p=0.8303 | +1.88 [+0.77, +3.25], p=0.0004 |
| plain300 | +0.08 [-0.26, +0.47], p=0.6626 | +0.97 [+0.29, +1.65], p=0.0050 | +0.71 [-0.47, +1.94], p=0.2617 | -0.06 [-2.13, +1.66], p=0.9633 | +2.95 [+1.47, +4.74], p=0.0000 |
| plain400 | +0.29 [-0.10, +0.73], p=0.1730 | +1.06 [+0.33, +1.79], p=0.0039 | +0.93 [-0.29, +2.25], p=0.1547 | +2.15 [-0.24, +4.10], p=0.0502 | +3.62 [+2.13, +5.48], p=0.0000 |

t15 minus method on the OOD mean, per seed (pp):

- plain250: +1.38, +0.03, +1.01, +1.22, -0.01
- plain300: +1.45, +1.01, +1.25, +1.62, +0.37
- plain400: +2.21, +1.99, +2.02, +2.41, +1.06

## Definitions

- **t15**: side = max( sqrt(a / 0.15), 1.25 b, 0.2 ), second pass if side <= 0.8  (accepted method; a = predicted area fraction, b = longer side of the predicted box)
- **box**: side = max( margin b, min_window ), second pass if side <= 0.8  (no area term)
- **plain250**: box rule with margin 2.5, min_window 0.02
- **plain300**: box rule with margin 3.0, min_window 0.02
- **plain400**: box rule with margin 4.0, min_window 0.02
- plain: plain crop of the predicted box with a fixed factor; min_window 0.02 only prevents a degenerate window.
- earlier_factors: 1.25, 1.5 and 2.0 are reported in zoom_box_v1 and are not repeated here.
- trigger_note: with the common trigger (second pass if the window side is at most 0.8) a larger factor leaves more images without a second pass; this is part of what a fixed factor does and is reported.

## Reading rules fixed in advance

- consequence: fixed_factor_exceeds_rule: the manuscript states that the magnification chosen by the rule is not the best one on polyps and names the better factor. rule_not_exceeded: the manuscript may state that no fixed factor between 1.25 and 4.0 exceeded the rule on polyps.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated; the baseline is not.
- ETIS-Larib is not touched.
- The factors 2.5, 3.0 and 4.0 were fixed before any result of this suite existed; no factor was tuned.
