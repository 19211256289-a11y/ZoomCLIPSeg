# formal_controls_kvasir_v1: control experiments (Kvasir source, trained on this PC)

Seeds 666, 667, 668; 100 epochs; fixed sentence; MC30; final-epoch weights.
Arms complete: detail, detail_scale, baseline, orig_fixed, orig_fixed_dice, flip, detail_up224.

## DSC (%), mean ± SD over seeds

| arm | inference | Val | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean |
|---|---|---|---|---|---|---|---|---|
| detail | single | 88.97 ± 0.35 | 89.69 ± 0.44 | 70.27 ± 0.61 | 82.52 ± 0.32 | 79.11 ± 1.61 | 76.75 ± 1.03 | 77.16 ± 0.39 |
| detail | two-pass |  | 89.96 ± 0.27 | 71.38 ± 0.85 | 83.48 ± 0.33 | 81.22 ± 0.68 | 79.69 ± 0.68 | 78.94 ± 0.56 |
| detail_scale | single | 88.31 ± 0.71 | 89.62 ± 0.25 | 70.78 ± 1.15 | 82.04 ± 1.62 | 79.15 ± 1.42 | 77.30 ± 0.72 | 77.32 ± 0.67 |
| detail_scale | two-pass |  | 89.80 ± 0.58 | 72.41 ± 1.87 | 83.39 ± 0.82 | 79.86 ± 1.65 | 79.54 ± 0.61 | 78.80 ± 0.69 |
| baseline | single | 87.80 ± 0.42 | 88.72 ± 0.18 | 69.30 ± 0.45 | 82.19 ± 0.24 | 80.28 ± 0.95 | 75.71 ± 2.52 | 76.87 ± 0.27 |
| baseline | two-pass |  | 89.18 ± 0.25 | 71.18 ± 0.26 | 83.29 ± 0.51 | 83.02 ± 1.72 | 79.29 ± 1.45 | 79.19 ± 0.45 |
| orig_fixed | single | 87.24 ± 0.29 | 88.73 ± 0.22 | 66.47 ± 0.61 | 77.56 ± 2.26 | 80.38 ± 0.99 | 73.38 ± 1.55 | 74.45 ± 1.00 |
| orig_fixed_dice | single | 87.41 ± 0.25 | 88.29 ± 0.30 | 67.74 ± 0.72 | 79.77 ± 1.37 | 81.00 ± 0.21 | 72.99 ± 1.36 | 75.37 ± 0.89 |
| flip | single | 88.08 ± 0.31 | 88.69 ± 0.38 | 69.28 ± 0.70 | 81.96 ± 1.54 | 81.10 ± 1.04 | 76.29 ± 0.88 | 77.16 ± 0.15 |
| detail_up224 | single | 88.74 ± 0.20 | 89.79 ± 0.91 | 69.90 ± 0.57 | 81.71 ± 0.41 | 79.35 ± 0.87 | 77.18 ± 1.51 | 77.03 ± 0.36 |

## Pre-specified contrasts (paired DSC difference in pp; 95% bootstrap interval over images)

| contrast | definition | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean | per seed |
|---|---|---|---|---|---|---|---|---|
| dice_fix | orig_fixed_dice − orig_fixed | -0.44 [-1.10, +0.11] | +1.28 [+0.53, +2.05] | +2.20 [+0.37, +4.09] | +0.61 [-0.56, +1.82] | -0.39 [-1.84, +1.26] | +0.92 | +0.24, +1.04, +1.49 |
| random_flip | flip − orig_fixed_dice | +0.40 [-0.53, +1.37] | +1.53 [+0.44, +2.66] | +2.20 [+0.66, +3.78] | +0.10 [-0.98, +1.10] | +3.31 [+1.36, +5.42] | +1.78 | +1.45, +2.91, +0.99 |
| paired_vs_flip | baseline − flip | +0.03 [-0.75, +0.82] | +0.02 [-0.67, +0.72] | +0.23 [-0.98, +1.41] | -0.82 [-2.32, +0.56] | -0.59 [-1.83, +0.60] | -0.29 | -0.25, -0.65, +0.03 |
| recipe_total | baseline − orig_fixed | -0.01 [-0.75, +0.74] | +2.83 [+1.63, +4.08] | +4.62 [+2.53, +6.89] | -0.10 [-2.21, +1.85] | +2.33 [+0.08, +4.82] | +2.42 | +1.45, +3.31, +2.50 |
| detail_branch | detail − baseline | +0.97 [+0.57, +1.39] | +0.97 [+0.27, +1.70] | +0.33 [-0.69, +1.25] | -1.17 [-2.37, -0.02] | +1.05 [-0.09, +2.23] | +0.29 | +0.20, +0.94, -0.26 |
| true_448_vs_upsampled | detail − detail_up224 | -0.10 [-0.58, +0.43] | +0.36 [-0.30, +1.04] | +0.80 [-0.63, +2.17] | -0.23 [-1.21, +0.80] | -0.42 [-1.44, +0.61] | +0.13 | +0.11, +0.80, -0.53 |
| scale_aug | detail_scale − detail | -0.08 [-0.98, +0.77] | +0.51 [-0.87, +1.92] | -0.48 [-2.54, +1.73] | +0.04 [-2.48, +2.76] | +0.55 [-1.78, +2.81] | +0.16 | +0.81, -1.05, +0.71 |
| two_pass_on_detail | detail two-pass − detail | +0.27 [-0.15, +0.73] | +1.12 [+0.27, +1.93] | +0.96 [-0.81, +2.75] | +2.11 [-0.34, +4.31] | +2.93 [+1.22, +4.81] | +1.78 | +2.35, +1.61, +1.38 |
| two_pass_on_detail_scale | detail_scale two-pass − detail_scale | +0.19 [-0.50, +0.91] | +1.63 [+0.82, +2.48] | +1.35 [+0.25, +2.46] | +0.71 [-1.22, +2.56] | +2.24 [+0.07, +4.48] | +1.48 | +1.77, +1.62, +1.05 |
| two_pass_on_baseline | baseline two-pass − baseline | +0.46 [-0.05, +1.02] | +1.88 [+1.08, +2.73] | +1.10 [-0.44, +2.69] | +2.74 [+0.05, +5.21] | +3.58 [+1.70, +5.63] | +2.33 | +1.96, +2.95, +2.06 |
| scale_aug_under_two_pass | detail_scale two-pass − detail two-pass | -0.16 [-1.33, +0.92] | +1.03 [-0.33, +2.43] | -0.09 [-2.16, +1.81] | -1.35 [-3.66, +0.97] | -0.15 [-2.12, +1.84] | -0.14 | +0.23, -1.04, +0.39 |

## Readings fixed in advance

- Two-pass inference after scale augmentation: **retained** (macro OOD gain +1.48 pp; retained >= 1.0 and every seed positive, absorbed < 0.5).
- Detail branch needs the real 448 image: **supported** (mean over five sets +0.08 pp; per seed +0.10, +0.48, -0.33).

Small-lesion and NSD differences, permutation p-values and per-seed counts are in results.json.
Disclosures: The five polyp test sets were used in earlier stages of this project. CVC-300 and CVC-ColonDB share 52 exact image hashes. The single-view arms use half the forward and backward passes per optimisation step of the paired arms.
