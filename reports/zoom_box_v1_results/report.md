# zoom_box_v1: SCZ window rule against a plain crop of the predicted box (detail-branch model, five seeds)

Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.

OOD = equal-weight mean of the four OOD polyp sets. Second pass = share of OOD images that receive a second pass; window = median window side of those images, as a fraction of the image side.

| method | second pass | window | Kvasir DSC | OOD DSC | gain over single pass [95% CI], p | t15 minus method [95% CI], p | verdict |
|---|---|---|---|---|---|---|---|
| single | 0% |  | 89.54 | 77.36 |  | +2.12 [+1.31, +2.91], p=0.0000 |  |
| t15 | 75% | 0.52 | 89.92 | 79.48 | +2.12 [+1.31, +2.90], p=0.0000 |  |  |
| plain125 | 88% | 0.37 | 89.43 | 77.18 | -0.18 [-1.31, +0.90], p=0.7650 | +2.30 [+1.71, +2.93], p=0.0000 | t15_better |
| plain150 | 80% | 0.42 | 89.84 | 78.69 | +1.33 [+0.49, +2.15], p=0.0015 | +0.79 [+0.39, +1.22], p=0.0001 | t15_better |
| plain200 | 62% | 0.50 | 89.98 | 79.15 | +1.80 [+1.11, +2.44], p=0.0000 | +0.32 [-0.06, +0.73], p=0.1120 | not_distinguishable |
| box125 | 88% | 0.37 | 89.62 | 77.34 | -0.02 [-1.14, +1.05], p=0.9762 | +2.14 [+1.56, +2.73], p=0.0000 | t15_better |

**Overall verdict by the rule fixed in advance: plain_crop_sufficient.**

Per test set, DSC (%):

| method | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| t15 | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |
| plain125 | 89.43 | 69.07 | 82.12 | 78.32 | 79.21 |
| plain150 | 89.84 | 71.17 | 84.00 | 81.02 | 78.57 |
| plain200 | 89.98 | 71.61 | 84.34 | 81.80 | 78.87 |
| box125 | 89.62 | 69.31 | 82.23 | 78.35 | 79.47 |

t15 minus method, per test set (pp):

| method | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| plain125 | +0.49 [-0.11, +1.19], p=0.1535 | +2.88 [+2.27, +3.52], p=0.0000 | +2.04 [+0.85, +3.35], p=0.0022 | +3.81 [+2.33, +5.42], p=0.0000 | +0.45 [-0.72, +1.65], p=0.4799 |
| plain150 | +0.07 [-0.23, +0.40], p=0.6656 | +0.78 [+0.34, +1.25], p=0.0006 | +0.16 [-0.63, +0.93], p=0.6947 | +1.11 [+0.04, +2.37], p=0.0611 | +1.10 [+0.38, +1.92], p=0.0048 |
| plain200 | -0.06 [-0.32, +0.19], p=0.6461 | +0.34 [-0.16, +0.88], p=0.2034 | -0.18 [-0.88, +0.47], p=0.6184 | +0.33 [-0.54, +1.31], p=0.5131 | +0.79 [-0.12, +1.76], p=0.1034 |
| box125 | +0.30 [-0.21, +0.82], p=0.2579 | +2.64 [+2.04, +3.28], p=0.0000 | +1.93 [+0.75, +3.20], p=0.0020 | +3.78 [+2.32, +5.37], p=0.0000 | +0.19 [-0.91, +1.35], p=0.7587 |

t15 minus method on the OOD mean, per seed (pp):

- plain125: +0.83, +3.03, +2.72, +2.13, +2.77
- plain150: +0.37, +0.26, +1.35, +1.45, +0.51
- plain200: +0.37, -0.05, +0.78, +0.99, -0.49
- box125: +0.83, +2.85, +2.64, +2.08, +2.29

## Definitions

- **t15**: side = max( sqrt(a / 0.15), 1.25 b, 0.2 ), second pass if side <= 0.8  (accepted method; a = predicted area fraction, b = longer side of the predicted box)
- **box**: side = max( margin b, min_window ), second pass if side <= 0.8  (no area term)
- **plain125**: box rule with margin 1.25, min_window 0.02
- **plain150**: box rule with margin 1.5, min_window 0.02
- **plain200**: box rule with margin 2.0, min_window 0.02
- **box125**: box rule with margin 1.25, min_window 0.2
- plain: plain crop of the predicted box with a fixed margin. min_window 0.02 only prevents a degenerate window of a few pixels; it is far below every value at which the lower bound of t15 acts.
- box125: t15 with the area term removed and everything else unchanged (margin 1.25, lower bound 0.2); isolates the area term.
- geometry_note: for a compact (roughly circular) prediction the area term of t15 equals about 2.3 b, so that t15 and plain200 choose similar windows for such lesions; they differ for elongated or fragmented predictions and through the lower bound.

## Reading rules fixed in advance

- consequence: if plain_crop_sufficient, the manuscript must not claim that the window rule itself improves on a plain crop; the contribution is then the diagnosis of the scale dependence and a training-free two-pass remedy.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated; the baseline is not.
- ETIS-Larib is not touched.
- The margins 1.25, 1.5 and 2.0 were fixed before any result of this suite existed; no margin was tuned.
