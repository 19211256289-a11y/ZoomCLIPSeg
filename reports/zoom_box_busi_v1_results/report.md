# zoom_box_busi_v1: SCZ window rule against a plain crop of the predicted box, breast ultrasound (detail-branch model, three seeds)

Nothing was trained. The locked configuration (tcal: the window rule with the quantile-matched target fraction) remains the method whatever these numbers show.

OOD = equal-weight mean of the four OOD ultrasound sets. Second pass = share of OOD images that receive a second pass; window = median window side of those images, as a fraction of the image side.

| method | second pass | window | BUSI DSC | OOD DSC | gain over single pass [95% CI], p | rule minus method [95% CI], p | verdict |
|---|---|---|---|---|---|---|---|
| single | 0% |  | 81.04 | 75.03 |  | +1.06 [+0.20, +2.05], p=0.0125 |  |
| tcal | 35% | 0.60 | 81.24 | 76.09 | +1.06 [+0.22, +2.04], p=0.0119 |  |  |
| plain125 | 78% | 0.52 | 78.36 | 74.66 | -0.37 [-1.57, +0.83], p=0.5690 | +1.43 [+0.47, +2.48], p=0.0041 | rule_better |
| plain150 | 62% | 0.54 | 78.90 | 75.63 | +0.61 [-0.27, +1.59], p=0.2357 | +0.46 [-0.29, +1.20], p=0.2551 | not_distinguishable |
| plain200 | 39% | 0.59 | 80.01 | 76.01 | +0.99 [+0.26, +1.77], p=0.0168 | +0.08 [-0.59, +0.74], p=0.8316 | not_distinguishable |
| plain300 | 15% | 0.59 | 81.41 | 75.12 | +0.10 [-0.11, +0.34], p=0.4503 | +0.96 [+0.15, +1.92], p=0.0205 | rule_better |
| plain400 | 9% | 0.56 | 81.46 | 75.05 | +0.02 [-0.13, +0.19], p=0.8076 | +1.04 [+0.22, +1.98], p=0.0110 | rule_better |

**Transfer verdict by the rule fixed in advance (rule against plain200): not_supported.**

Per test set, DSC (%):

| method | BUSI | BUSBRA | BUSUC | BUID | UDIAT |
|---|---|---|---|---|---|
| single | 81.04 | 68.94 | 83.58 | 74.63 | 72.95 |
| tcal | 81.24 | 70.01 | 83.73 | 75.24 | 75.36 |
| plain125 | 78.36 | 68.15 | 82.96 | 73.39 | 74.12 |
| plain150 | 78.90 | 68.54 | 83.73 | 73.84 | 76.42 |
| plain200 | 80.01 | 69.54 | 83.65 | 74.05 | 76.79 |
| plain300 | 81.41 | 69.07 | 83.40 | 74.63 | 73.39 |
| plain400 | 81.46 | 69.02 | 83.35 | 74.63 | 73.19 |

rule minus method, per test set (pp):

| method | BUSI | BUSBRA | BUSUC | BUID | UDIAT |
|---|---|---|---|---|---|
| plain125 | +2.88 [+0.37, +6.05], p=0.0346 | +1.85 [+0.70, +3.12], p=0.0023 | +0.77 [-0.50, +2.15], p=0.2706 | +1.86 [-0.35, +4.41], p=0.1496 | +1.24 [-1.31, +4.20], p=0.4210 |
| plain150 | +2.34 [+0.35, +5.12], p=0.0331 | +1.47 [+0.55, +2.47], p=0.0024 | +0.01 [-0.96, +1.02], p=0.9895 | +1.40 [-0.19, +3.49], p=0.1629 | -1.06 [-3.15, +0.65], p=0.3333 |
| plain200 | +1.24 [-0.12, +3.15], p=0.1525 | +0.47 [-0.20, +1.18], p=0.1812 | +0.08 [-0.62, +0.83], p=0.8378 | +1.19 [+0.13, +2.68], p=0.0256 | -1.43 [-3.55, +0.57], p=0.2173 |
| plain300 | -0.17 [-1.02, +0.82], p=0.7367 | +0.94 [+0.17, +1.73], p=0.0211 | +0.34 [-0.23, +1.01], p=0.3377 | +0.62 [+0.00, +1.46], p=0.0606 | +1.97 [-1.03, +5.56], p=0.2758 |
| plain400 | -0.22 [-1.06, +0.76], p=0.6742 | +0.99 [+0.22, +1.81], p=0.0138 | +0.38 [-0.26, +1.12], p=0.2932 | +0.62 [+0.00, +1.48], p=0.0630 | +2.17 [-0.89, +5.77], p=0.2443 |

rule minus method on the OOD mean, per seed (pp):

- plain125: +1.17, +2.16, +0.97
- plain150: +0.76, +0.16, +0.45
- plain200: +0.41, -0.85, +0.67
- plain300: +1.38, +0.20, +1.32
- plain400: +1.34, +0.25, +1.53

## Definitions

- **tcal**: side = max( sqrt(a / tau), 1.25 b, 0.2 ) with the quantile-matched tau, second pass if side <= 0.8  (accepted method)
- **box**: side = max( margin b, 0.02 ), second pass if side <= 0.8  (plain crop of the predicted box; no area term)
- **plain125**: box rule with margin 1.25, min_window 0.02
- **plain150**: box rule with margin 1.5, min_window 0.02
- **plain200**: box rule with margin 2.0, min_window 0.02
- **plain300**: box rule with margin 3.0, min_window 0.02
- **plain400**: box rule with margin 4.0, min_window 0.02
- factors: 1.25, 1.5 and 2.0 are the factors of zoom_box_v1; 3.0 and 4.0 were added before any ultrasound result existed.
- geometry_note: for a compact prediction the area term of the rule equals about 3.0 b with tau = 0.086 (about 2.3 b on polyps), so that plain300 is expected to choose windows similar to the rule on ultrasound.

## Reading rules fixed in advance

- consequence: supported: the manuscript may state that a factor chosen on one dataset does not carry over while the rule does. not_supported: the manuscript states that a fixed factor of about two performed like the rule on both modalities, and does not present the transfer rule as an advantage in accuracy.

## Disclosures

- The ultrasound test sets were used in earlier stages of this project.
- BUID (35 images) and UDIAT (25 images) are small; the study has three seeds.
- Only the detail-branch model is evaluated.
- No factor was tuned; whether a factor outside this list does better is unknown.
