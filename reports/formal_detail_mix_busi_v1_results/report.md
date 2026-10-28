# formal_detail_mix_busi_v1: results

Source BUSI; arms baseline, detail, detail_global_mix; new seeds 666, 667, 668; epoch 100; MC30, batch 32, threshold 0.5; fixed text `a tumor in a breast ultrasound image`.
Small target: foreground fraction <= 0.021694 (source Train positive 25th percentile). Gates and the final-model rule were fixed before training.

## Gates

**detail_replicates** (detail - baseline, seeds 666, 667, 668): **PASS**

| Criterion | Value | Result |
|---|---|---|
| source_test_dsc_gain | +1.43 pp | PASS |
| ood_sets_positive | 4 (BUSBRA, BUSUC, BUID, UDIAT) | PASS |
| macro_small_dsc_gain | +8.24 pp | PASS |
| guard_no_large_regression | +1.43 pp | PASS |

Macro gain per seed: +3.44 / +15.61 / +7.43 pp; seed-level paired t p = 0.133.

| Dataset | DSC gain pp | Small-target DSC gain pp |
|---|---:|---:|
| BUSI | +1.43 | +6.19 |
| BUSBRA | +11.24 | +19.09 |
| BUSUC | +7.01 | +11.42 |
| BUID | +8.29 | +0.00 |
| UDIAT | +16.17 | +4.50 |

**combination_adds** (detail_global_mix - detail, seeds 666, 667, 668): **FAIL**

| Criterion | Value | Result |
|---|---|---|
| sets_positive | 0 (none) | FAIL |
| macro_dsc_gain | -2.08 pp | FAIL |
| seeds_positive_macro | 1 of 3 | FAIL |

Macro gain per seed: -3.12 / -4.16 / +1.05 pp; seed-level paired t p = 0.322.

| Dataset | DSC gain pp | Small-target DSC gain pp |
|---|---:|---:|
| BUSI | -0.58 | -2.39 |
| BUSBRA | -0.38 | -6.27 |
| BUSUC | -1.19 | -6.46 |
| BUID | -2.36 | +0.97 |
| UDIAT | -5.86 | -4.30 |

Final-model rule: Fixed before training. The combination becomes the final model only if combination_adds passes here AND combination_adds_five_seed passes in formal_detail_mix_kvasir_v2. Otherwise the final model is detail alone. Statistic for every criterion: mean over seeds of paired differences in percentage points.

## Cohort notes

| Dataset | Records | Unique images | Exact overlap with source Train/Val | Shared with other test sets |
|---|---:|---:|---:|---|
| BUSI | 78 | 78 | 0 | none |
| BUSBRA | 282 | 282 | 0 | none |
| BUSUC | 122 | 122 | 0 | none |
| BUID | 35 | 35 | 0 | none |
| UDIAT | 25 | 25 | 0 | none |

## New seeds only

### Val (source Val (development))

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 77.97 ± 1.21 | 75.29 ± 2.72 | 78.18 ± 1.95 | 80.47 ± 1.19 | 76.21 ± 4.14 | 78.54 ± 1.21 | 8/9/8 | 3/2/4 of 13 |
| detail | 77.63 ± 1.07 | 77.45 ± 0.42 | 80.84 ± 0.30 | 80.18 ± 1.09 | 81.69 ± 0.79 | 78.41 ± 1.69 | 7/8/7 | 5/4/6 of 13 |
| detail_global_mix | 77.12 ± 2.24 | 76.37 ± 1.11 | 80.24 ± 1.49 | 79.63 ± 2.23 | 78.67 ± 2.36 | 79.24 ± 3.48 | 8/9/7 | 6/3/6 of 13 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 78 | -0.34 | [-4.08, +2.60] | -1.86 / +1.29 / -0.46 | 1/3 | 53/14/11 | +1.33 | 0.855 | <0.001 | -1.03 |
| detail_global_mix - baseline | 78 | -0.85 | [-3.71, +1.79] | -4.52 / +1.85 / +0.12 | 2/3 | 46/24/8 | +0.56 | 0.572 | 0.007 | -1.73 |
| detail_global_mix - detail | 78 | -0.51 | [-2.51, +1.77] | -2.66 / +0.56 / +0.59 | 2/3 | 25/43/10 | -0.24 | 0.706 | 0.022 | -1.55 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 16 | +2.16 | [+0.32, +3.96] | -0.48 / +5.63 / +1.33 | 2/3 | 12/4/0 | +2.48 | 0.036 | 0.059 | +1.33 |
| detail_global_mix - baseline | 16 | +1.08 | [-1.08, +3.06] | -2.52 / +4.73 / +1.03 | 2/3 | 10/6/0 | +1.97 | 0.348 | 0.141 | +0.21 |
| detail_global_mix - detail | 16 | -1.08 | [-2.91, +0.19] | -2.05 / -0.91 / -0.30 | 0/3 | 5/11/0 | -0.31 | 0.202 | 0.163 | -1.53 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.85 / +4.63 / +2.52 | +3.53 / +9.46 / +3.45 | +1.13 / -2.44 / +0.90 | +2 / +2 / +2 | -1 / -1 / -1 |
| detail_global_mix - baseline | -0.81 / +3.76 / +3.22 | -2.95 / +9.05 / +1.28 | +4.74 / -3.93 / +1.27 | +3 / +1 / +2 | +0 / +0 / -1 |
| detail_global_mix - detail | -1.66 / -0.87 / +0.70 | -6.48 / -0.41 / -2.17 | +3.61 / -1.49 / +0.37 | +1 / -1 / +0 | +1 / +1 / +0 |

### BUSI (Test, source domain)

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 79.61 ± 2.62 | 70.95 ± 7.46 | 77.58 ± 3.18 | 82.19 ± 2.68 | 78.44 ± 5.80 | 86.93 ± 2.57 | 8/10/9 | 1/1/2 of 13 |
| detail | 81.03 ± 1.17 | 77.14 ± 3.31 | 82.37 ± 0.63 | 83.65 ± 1.18 | 84.26 ± 2.29 | 85.63 ± 1.63 | 7/7/6 | 3/4/3 of 13 |
| detail_global_mix | 80.45 ± 2.22 | 74.75 ± 2.53 | 81.16 ± 1.54 | 82.98 ± 2.26 | 79.48 ± 4.96 | 87.17 ± 2.62 | 8/6/6 | 2/2/5 of 13 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 78 | +1.43 | [-2.63, +5.18] | -0.31 / +2.60 / +2.01 | 2/3 | 53/14/11 | +2.10 | 0.508 | <0.001 | +0.01 |
| detail_global_mix - baseline | 78 | +0.84 | [-2.36, +4.32] | -2.07 / +5.59 / -0.98 | 1/3 | 53/15/10 | +1.32 | 0.643 | 0.003 | -0.63 |
| detail_global_mix - detail | 78 | -0.58 | [-2.32, +1.24] | -1.76 / +2.99 / -2.98 | 1/3 | 21/46/11 | -0.19 | 0.536 | 0.006 | -1.48 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 14 | +6.19 | [+3.20, +10.52] | +4.80 / +13.12 / +0.66 | 3/3 | 13/0/1 | +5.00 | <0.001 | 0.002 | +3.93 |
| detail_global_mix - baseline | 14 | +3.80 | [-1.48, +9.36] | -3.68 / +15.07 / +0.02 | 2/3 | 12/1/1 | +3.84 | 0.231 | 0.021 | +1.08 |
| detail_global_mix - detail | 14 | -2.39 | [-6.37, +0.12] | -8.48 / +1.95 / -0.64 | 1/3 | 5/8/1 | -0.09 | 0.192 | 0.442 | -3.00 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +2.70 / +7.73 / +3.95 | +1.33 / +8.76 / +7.38 | +0.11 / -3.78 / -0.22 | +2 / +3 / +1 | -1 / -3 / -3 |
| detail_global_mix - baseline | -0.95 / +8.24 / +3.44 | -11.07 / +7.96 / +6.24 | +5.82 / -4.40 / -0.71 | +1 / +1 / +3 | +0 / -4 / -3 |
| detail_global_mix - detail | -3.65 / +0.51 / -0.50 | -12.40 / -0.80 / -1.14 | +5.71 / -0.62 / -0.49 | -1 / -2 / +2 | +1 / -1 / +0 |

### BUSBRA (Test, OOD)

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 57.71 ± 10.13 | 42.18 ± 11.97 | 57.71 ± 10.13 | 61.16 ± 10.39 | 70.09 ± 3.55 | 68.15 ± 9.64 | 62/122/91 | 0/0/0 of 0 |
| detail | 68.94 ± 0.57 | 61.27 ± 2.50 | 68.94 ± 0.57 | 72.88 ± 0.60 | 78.21 ± 2.78 | 73.56 ± 0.49 | 60/58/52 | 0/0/0 of 0 |
| detail_global_mix | 68.56 ± 3.99 | 55.00 ± 11.55 | 68.56 ± 3.99 | 72.33 ± 4.27 | 74.27 ± 5.20 | 77.04 ± 5.44 | 64/63/42 | 0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 282 | +11.24 | [+9.32, +13.28] | +0.58 / +21.95 / +11.18 | 3/3 | 240/38/4 | +5.97 | <0.001 | <0.001 | +10.77 |
| detail_global_mix - baseline | 282 | +10.86 | [+8.92, +12.80] | -1.94 / +19.00 / +15.51 | 2/3 | 228/52/2 | +4.39 | <0.001 | <0.001 | +10.38 |
| detail_global_mix - detail | 282 | -0.38 | [-1.67, +0.91] | -2.52 / -2.96 / +4.33 | 1/3 | 114/165/3 | -0.99 | 0.565 | 0.055 | -0.70 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 35 | +19.09 | [+12.10, +26.55] | +2.41 / +26.61 / +28.25 | 3/3 | 29/5/1 | +9.44 | <0.001 | <0.001 | +16.21 |
| detail_global_mix - baseline | 35 | +12.82 | [+6.70, +19.51] | -12.23 / +18.46 / +32.24 | 2/3 | 28/6/1 | +6.77 | <0.001 | <0.001 | +9.60 |
| detail_global_mix - detail | 35 | -6.27 | [-11.07, -1.54] | -14.65 / -8.15 / +3.98 | 1/3 | 11/22/2 | -1.63 | 0.014 | 0.023 | -8.21 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.58 / +21.95 / +11.18 | +5.14 / +11.42 / +7.80 | -2.47 / +15.54 / +3.16 | +0 / +0 / +0 | -2 / -64 / -39 |
| detail_global_mix - baseline | -1.94 / +19.00 / +15.51 | -5.69 / +9.33 / +8.91 | +4.53 / +13.21 / +8.91 | +0 / +0 / +0 | +2 / -59 / -49 |
| detail_global_mix - detail | -2.52 / -2.96 / +4.33 | -10.83 / -2.09 / +1.10 | +7.00 / -2.33 / +5.76 | +0 / +0 / +0 | +4 / +5 / -10 |

### BUSUC (Test, OOD)

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 76.56 ± 5.73 | 53.75 ± 8.64 | 76.56 ± 5.73 | 80.96 ± 5.69 | 79.40 ± 3.17 | 88.00 ± 2.63 | 9/20/10 | 0/0/0 of 0 |
| detail | 83.58 ± 0.84 | 65.16 ± 2.07 | 83.58 ± 0.84 | 88.07 ± 0.84 | 85.06 ± 1.38 | 89.79 ± 1.13 | 6/8/6 | 0/0/0 of 0 |
| detail_global_mix | 82.38 ± 1.64 | 58.70 ± 10.80 | 82.38 ± 1.64 | 86.83 ± 1.73 | 83.04 ± 2.96 | 90.54 ± 3.53 | 9/9/6 | 0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 122 | +7.01 | [+5.40, +8.81] | +3.42 / +12.81 / +4.81 | 3/3 | 114/7/1 | +3.57 | <0.001 | <0.001 | +6.32 |
| detail_global_mix - baseline | 122 | +5.82 | [+4.17, +7.62] | -0.08 / +11.77 / +5.77 | 2/3 | 106/14/2 | +2.93 | <0.001 | <0.001 | +5.17 |
| detail_global_mix - detail | 122 | -1.19 | [-2.30, -0.08] | -3.50 / -1.04 / +0.95 | 1/3 | 35/86/1 | -0.53 | 0.038 | <0.001 | -1.71 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 6 | +11.42 | [+0.92, +22.10] | +3.46 / +16.51 / +14.28 | 3/3 | 5/1/0 | +2.99 | 0.094 | 0.093 | +1.35 |
| detail_global_mix - baseline | 6 | +4.95 | [+0.80, +10.43] | -16.72 / +16.33 / +15.25 | 2/3 | 4/1/1 | +2.63 | 0.125 | n/a | +1.20 |
| detail_global_mix - detail | 6 | -6.46 | [-14.66, -0.08] | -20.18 / -0.18 / +0.97 | 1/3 | 1/5/0 | -0.35 | 0.125 | 0.142 | -9.73 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +3.42 / +12.81 / +4.81 | +3.01 / +8.58 / +5.39 | +1.20 / +3.62 / +0.55 | +0 / +0 / +0 | -3 / -12 / -4 |
| detail_global_mix - baseline | -0.08 / +11.77 / +5.77 | -3.40 / +7.78 / +6.52 | +4.91 / +2.86 / -0.14 | +0 / +0 / +0 | +0 / -11 / -4 |
| detail_global_mix - detail | -3.50 / -1.04 / +0.95 | -6.41 / -0.80 / +1.13 | +3.70 / -0.76 / -0.70 | +0 / +0 / +0 | +3 / +1 / +0 |

### BUID (Test, OOD)

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 66.34 ± 6.23 | 0.00 ± 0.00 | 66.34 ± 6.23 | 68.23 ± 6.26 | 79.83 ± 2.48 | 80.69 ± 7.70 | 7/11/7 | 0/0/0 of 0 |
| detail | 74.63 ± 0.87 | 0.00 ± 0.00 | 74.63 ± 0.87 | 76.65 ± 0.91 | 83.14 ± 2.42 | 85.30 ± 1.98 | 5/5/4 | 0/0/0 of 0 |
| detail_global_mix | 72.26 ± 3.87 | 0.97 ± 1.68 | 72.26 ± 3.87 | 74.20 ± 4.10 | 81.35 ± 2.10 | 83.82 ± 4.29 | 5/7/4 | 0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 35 | +8.29 | [+3.93, +13.59] | +4.36 / +14.88 / +5.62 | 3/3 | 29/2/4 | +2.11 | <0.001 | <0.001 | +5.39 |
| detail_global_mix - baseline | 35 | +5.92 | [+2.76, +9.83] | +2.17 / +9.36 / +6.24 | 3/3 | 30/4/1 | +1.58 | <0.001 | <0.001 | +3.76 |
| detail_global_mix - detail | 35 | -2.36 | [-4.43, -0.79] | -2.18 / -5.52 / +0.61 | 1/3 | 10/24/1 | -0.71 | <0.001 | 0.002 | -2.65 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 1 | +0.00 | [+0.00, +0.00] | +0.00 / +0.00 / +0.00 | 0/3 | 0/0/1 | +0.00 | 1.000 | n/a | n/a |
| detail_global_mix - baseline | 1 | +0.97 | [+0.97, +0.97] | +0.00 / +0.00 / +2.91 | 1/3 | 1/0/0 | +0.97 | 1.000 | n/a | n/a |
| detail_global_mix - detail | 1 | +0.97 | [+0.97, +0.97] | +0.00 / +0.00 / +2.91 | 1/3 | 1/0/0 | +0.97 | 1.000 | n/a | n/a |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +4.36 / +14.88 / +5.62 | +1.11 / +6.66 / +2.16 | +1.77 / +11.89 / +0.17 | +0 / +0 / +0 | -2 / -6 / -3 |
| detail_global_mix - baseline | +2.17 / +9.36 / +6.24 | -3.67 / +4.59 / +3.64 | +3.64 / +6.60 / -0.83 | +0 / +0 / +0 | -2 / -4 / -3 |
| detail_global_mix - detail | -2.18 / -5.52 / +0.61 | -4.78 / -2.07 / +1.48 | +1.86 / -5.29 / -1.01 | +0 / +0 / +0 | +0 / +2 / +0 |

### UDIAT (Test, OOD)

Seeds 666, 667, 668; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 56.78 ± 8.11 | 53.70 ± 6.09 | 56.78 ± 8.11 | 60.00 ± 8.10 | 77.23 ± 6.95 | 63.69 ± 10.08 | 8/10/8 | 0/0/0 of 0 |
| detail | 72.95 ± 0.76 | 58.20 ± 4.30 | 72.95 ± 0.76 | 76.72 ± 0.76 | 83.05 ± 3.76 | 77.33 ± 2.03 | 5/4/4 | 0/0/0 of 0 |
| detail_global_mix | 67.09 ± 8.31 | 53.90 ± 6.48 | 67.09 ± 8.31 | 70.63 ± 8.95 | 79.25 ± 6.05 | 75.05 ± 8.25 | 6/9/4 | 0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 25 | +16.17 | [+8.52, +25.39] | +9.18 / +25.82 / +13.52 | 3/3 | 22/1/2 | +6.54 | <0.001 | <0.001 | +11.39 |
| detail_global_mix - baseline | 25 | +10.31 | [+3.61, +17.51] | +3.53 / +11.55 / +15.85 | 3/3 | 20/4/1 | +4.81 | 0.006 | <0.001 | +6.94 |
| detail_global_mix - detail | 25 | -5.86 | [-12.64, -0.21] | -5.66 / -14.26 / +2.33 | 1/3 | 9/15/1 | -0.34 | 0.089 | 0.119 | -7.54 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 8 | +4.50 | [+2.04, +7.12] | -6.82 / +12.72 / +7.62 | 2/3 | 6/0/2 | +3.82 | 0.031 | 0.036 | +2.65 |
| detail_global_mix - baseline | 8 | +0.20 | [-10.00, +7.16] | -10.97 / +0.83 / +10.74 | 2/3 | 6/1/1 | +2.94 | 1.000 | 0.272 | -3.12 |
| detail_global_mix - detail | 8 | -4.30 | [-16.59, +4.33] | -4.14 / -11.89 / +3.13 | 1/3 | 3/4/1 | -0.17 | 0.659 | 0.673 | -8.38 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +9.18 / +25.82 / +13.52 | -1.87 / +11.32 / +8.00 | +9.45 / +22.92 / +8.56 | +0 / +0 / +0 | -3 / -6 / -4 |
| detail_global_mix - baseline | +3.53 / +11.55 / +15.85 | -12.31 / +8.85 / +9.53 | +10.46 / +13.47 / +10.14 | +0 / +0 / +0 | -2 / -1 / -4 |
| detail_global_mix - detail | -5.66 / -14.26 / +2.33 | -10.44 / -2.47 / +1.53 | +1.01 / -9.45 / +1.59 | +0 / +0 / +0 | +1 / +5 / +0 |

## Limitations

- Image-level tests are conditional on the trained seeds, not patient clustered, and not corrected for multiple comparisons.
