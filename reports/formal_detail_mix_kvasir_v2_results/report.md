# formal_detail_mix_kvasir_v2: results

Source Kvasir; arms baseline, global_mix, detail, detail_global_mix; new seeds 669, 670; epoch 100; MC30, batch 32, threshold 0.5; fixed text `a polyp in a colonoscopy image`.
Small target: foreground fraction <= 0.058990 (source Train positive 25th percentile). Gates and the final-model rule were fixed before training.

## Gates

**detail_five_seed** (detail - baseline, seeds 666, 667, 668, 669, 670): **PASS**

| Criterion | Value | Result |
|---|---|---|
| macro_dsc_gain | +0.96 pp | PASS |
| macro_small_dsc_gain | +1.29 pp | PASS |
| seeds_positive_macro | 5 of 5 | PASS |

Macro gain per seed: +0.76 / +0.31 / +0.83 / +1.23 / +1.66 pp; seed-level paired t p = 0.014.

| Dataset | DSC gain pp | Small-target DSC gain pp |
|---|---:|---:|
| Kvasir | +0.73 | +0.97 |
| ColonDB | +1.17 | +1.49 |
| ClinicDB | +1.62 | +2.54 |
| CVC300 | +0.11 | +0.11 |
| BKAI | +1.15 | +1.32 |

**combination_adds_five_seed** (detail_global_mix - detail, seeds 666, 667, 668, 669, 670): **PASS**

| Criterion | Value | Result |
|---|---|---|
| sets_positive | 5 (Kvasir, ColonDB, ClinicDB, CVC300, BKAI) | PASS |
| macro_dsc_gain | +0.43 pp | PASS |
| seeds_positive_macro | 4 of 5 | PASS |

Macro gain per seed: +0.74 / +1.07 / +0.28 / +0.43 / -0.40 pp; seed-level paired t p = 0.160.

| Dataset | DSC gain pp | Small-target DSC gain pp |
|---|---:|---:|
| Kvasir | +0.06 | -0.28 |
| ColonDB | +0.13 | +0.28 |
| ClinicDB | +0.12 | -0.29 |
| CVC300 | +0.73 | +0.85 |
| BKAI | +1.09 | +1.06 |

Final-model rule: see formal_detail_mix_busi_v1

## Cohort notes

| Dataset | Records | Unique images | Exact overlap with source Train/Val | Shared with other test sets |
|---|---:|---:|---:|---|
| Kvasir | 100 | 100 | 0 | none |
| ColonDB | 360 | 329 | 0 | {'CVC300': 52} |
| ClinicDB | 61 | 61 | 0 | none |
| CVC300 | 60 | 58 | 0 | {'ColonDB': 52} |
| BKAI | 100 | 100 | 0 | none |

## New seeds only

### Val (source Val (development))

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 88.69 ± 0.57 | 85.25 ± 0.14 | 88.69 ± 0.57 | 90.65 ± 0.55 | 90.22 ± 0.95 | 87.78 ± 0.37 | 3/3 | 0/0 of 0 |
| global_mix | 88.61 ± 1.03 | 85.21 ± 0.65 | 88.61 ± 1.03 | 90.55 ± 1.00 | 89.67 ± 2.03 | 88.04 ± 0.88 | 2/3 | 0/0 of 0 |
| detail | 89.05 ± 0.49 | 85.88 ± 0.11 | 89.05 ± 0.49 | 90.92 ± 0.46 | 90.68 ± 1.03 | 87.62 ± 0.22 | 4/3 | 0/0 of 0 |
| detail_global_mix | 89.05 ± 0.71 | 86.25 ± 0.14 | 89.05 ± 0.71 | 90.94 ± 0.72 | 90.23 ± 1.69 | 88.29 ± 0.28 | 3/3 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +0.36 | [-0.19, +0.91] | +0.31 / +0.42 | 2/2 | 67/32/1 | +0.36 | 0.196 | 0.008 | +0.19 |
| global_mix - baseline | 100 | -0.08 | [-0.70, +0.49] | +0.24 / -0.41 | 1/2 | 43/55/2 | -0.06 | 0.788 | 0.562 | -0.31 |
| detail_global_mix - baseline | 100 | +0.36 | [-0.27, +0.99] | +0.46 / +0.26 | 2/2 | 58/42/0 | +0.33 | 0.266 | 0.036 | +0.13 |
| detail_global_mix - detail | 100 | -0.00 | [-0.48, +0.51] | +0.15 / -0.15 | 1/2 | 41/59/0 | -0.16 | 0.989 | 0.389 | -0.21 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 25 | +0.63 | [-0.05, +1.22] | +0.46 / +0.80 | 2/2 | 18/6/1 | +0.49 | 0.069 | 0.005 | +0.37 |
| global_mix - baseline | 25 | -0.04 | [-0.41, +0.32] | +0.32 / -0.41 | 1/2 | 12/12/1 | +0.00 | 0.822 | 0.989 | -0.22 |
| detail_global_mix - baseline | 25 | +1.00 | [+0.30, +1.63] | +1.00 / +1.00 | 2/2 | 17/8/0 | +0.89 | 0.009 | 0.007 | +0.75 |
| detail_global_mix - detail | 25 | +0.37 | [+0.02, +0.74] | +0.54 / +0.19 | 2/2 | 14/11/0 | +0.30 | 0.068 | 0.125 | +0.17 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.31 / +0.42 | +0.52 / +0.41 | -0.27 / -0.05 | +0 / +0 | +1 / +0 |
| global_mix - baseline | +0.24 / -0.41 | +0.22 / -1.31 | +0.63 / -0.10 | +0 / +0 | -1 / +0 |
| detail_global_mix - baseline | +0.46 / +0.26 | +0.53 / -0.51 | +0.45 / +0.58 | +0 / +0 | +0 / +0 |
| detail_global_mix - detail | +0.15 / -0.15 | +0.01 / -0.92 | +0.71 / +0.63 | +0 / +0 | -1 / +0 |

### Kvasir (Test, source domain)

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 88.65 ± 0.00 | 87.67 ± 0.47 | 88.65 ± 0.00 | 90.76 ± 0.03 | 91.54 ± 0.18 | 88.75 ± 0.44 | 3/2 | 0/0 of 0 |
| global_mix | 88.60 ± 0.14 | 86.99 ± 1.53 | 88.60 ± 0.14 | 90.66 ± 0.18 | 91.02 ± 0.45 | 88.81 ± 0.23 | 3/2 | 0/0 of 0 |
| detail | 89.66 ± 0.22 | 88.63 ± 0.38 | 89.66 ± 0.22 | 91.68 ± 0.27 | 93.13 ± 0.60 | 89.21 ± 0.23 | 3/3 | 0/0 of 0 |
| detail_global_mix | 89.43 ± 0.40 | 88.56 ± 0.60 | 89.43 ± 0.40 | 91.42 ± 0.36 | 92.61 ± 1.15 | 88.74 ± 0.79 | 3/2 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +1.01 | [+0.53, +1.55] | +0.86 / +1.17 | 2/2 | 68/32/0 | +0.47 | <0.001 | <0.001 | +0.77 |
| global_mix - baseline | 100 | -0.05 | [-0.73, +0.64] | -0.14 / +0.05 | 1/2 | 53/47/0 | +0.04 | 0.899 | 0.927 | -0.33 |
| detail_global_mix - baseline | 100 | +0.78 | [-0.19, +1.68] | +1.07 / +0.50 | 2/2 | 72/28/0 | +0.62 | 0.103 | <0.001 | +0.47 |
| detail_global_mix - detail | 100 | -0.23 | [-1.11, +0.57] | +0.21 / -0.67 | 1/2 | 54/45/1 | +0.07 | 0.602 | 0.501 | -0.50 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 22 | +0.97 | [+0.21, +1.76] | +0.37 / +1.56 | 2/2 | 15/7/0 | +0.80 | 0.024 | 0.055 | +0.58 |
| global_mix - baseline | 22 | -0.67 | [-2.34, +0.98] | -2.08 / +0.74 | 1/2 | 11/11/0 | -0.25 | 0.450 | 0.363 | -1.44 |
| detail_global_mix - baseline | 22 | +0.89 | [-1.85, +3.43] | +0.99 / +0.80 | 2/2 | 16/6/0 | +1.10 | 0.526 | 0.055 | -0.16 |
| detail_global_mix - detail | 22 | -0.07 | [-2.80, +2.38] | +0.62 / -0.77 | 1/2 | 12/10/0 | +0.27 | 0.929 | 0.770 | -1.11 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.86 / +1.17 | +1.03 / +2.14 | +0.93 / -0.01 | +0 / +0 | +0 / +1 |
| global_mix - baseline | -0.14 / +0.05 | -0.97 / -0.07 | +0.53 / -0.41 | +0 / +0 | +0 / +0 |
| detail_global_mix - baseline | +1.07 / +0.50 | +0.12 / +2.01 | +0.86 / -0.88 | +0 / +0 | +0 / +0 |
| detail_global_mix - detail | +0.21 / -0.67 | -0.91 / -0.13 | -0.07 / -0.86 | +0 / +0 | +0 / -1 |

### ColonDB (Test, OOD)

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 69.26 ± 0.79 | 66.85 ± 0.68 | 69.26 ± 0.79 | 72.30 ± 0.83 | 75.76 ± 2.37 | 66.34 ± 5.48 | 74/88 | 0/0 of 0 |
| global_mix | 69.50 ± 0.04 | 66.86 ± 1.05 | 69.50 ± 0.04 | 72.51 ± 0.14 | 73.33 ± 3.29 | 67.65 ± 4.69 | 83/80 | 0/0 of 0 |
| detail | 71.38 ± 0.08 | 68.94 ± 1.43 | 71.38 ± 0.08 | 74.44 ± 0.05 | 76.84 ± 5.10 | 68.07 ± 5.09 | 72/72 | 0/0 of 0 |
| detail_global_mix | 71.18 ± 1.45 | 69.43 ± 1.07 | 71.18 ± 1.45 | 74.30 ± 1.52 | 76.91 ± 1.50 | 67.58 ± 4.94 | 58/80 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 360 | +2.12 | [+1.39, +2.89] | +1.61 / +2.62 | 2/2 | 206/138/16 | +0.72 | <0.001 | <0.001 | +1.88 |
| global_mix - baseline | 360 | +0.23 | [-0.49, +0.99] | -0.35 / +0.82 | 1/2 | 156/187/17 | -0.09 | 0.543 | 0.338 | +0.02 |
| detail_global_mix - baseline | 360 | +1.92 | [+0.98, +2.92] | +2.38 / +1.46 | 2/2 | 181/162/17 | +0.10 | <0.001 | 0.036 | +1.64 |
| detail_global_mix - detail | 360 | -0.20 | [-1.08, +0.71] | +0.77 / -1.17 | 1/2 | 159/185/16 | -0.04 | 0.668 | 0.063 | -0.43 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 241 | +2.08 | [+1.16, +3.07] | +1.55 / +2.62 | 2/2 | 137/90/14 | +0.77 | <0.001 | <0.001 | +1.73 |
| global_mix - baseline | 241 | +0.01 | [-0.90, +0.94] | -0.26 / +0.27 | 1/2 | 100/125/16 | -0.11 | 0.991 | 0.198 | -0.32 |
| detail_global_mix - baseline | 241 | +2.58 | [+1.34, +3.91] | +3.82 / +1.34 | 2/2 | 123/103/15 | +0.17 | <0.001 | 0.012 | +2.16 |
| detail_global_mix - detail | 241 | +0.49 | [-0.57, +1.62] | +2.26 / -1.28 | 1/2 | 109/118/14 | +0.00 | 0.377 | 0.451 | +0.14 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +1.61 / +2.62 | -0.85 / +3.01 | +1.46 / +2.01 | +0 / +0 | -2 / -16 |
| global_mix - baseline | -0.35 / +0.82 | -3.08 / -1.78 | +0.76 / +1.86 | +0 / +0 | +9 / -8 |
| detail_global_mix - baseline | +2.38 / +1.46 | +1.76 / +0.53 | +0.86 / +1.62 | +0 / +0 | -16 / -8 |
| detail_global_mix - detail | +0.77 / -1.17 | +2.61 / -2.48 | -0.60 / -0.39 | +0 / +0 | -14 / +8 |

### ClinicDB (Test, OOD)

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 80.84 ± 0.63 | 79.82 ± 0.13 | 80.84 ± 0.63 | 84.78 ± 0.60 | 86.94 ± 1.51 | 73.93 ± 2.05 | 6/7 | 0/0 of 0 |
| global_mix | 81.36 ± 0.07 | 79.65 ± 0.66 | 81.36 ± 0.07 | 85.31 ± 0.24 | 85.98 ± 4.93 | 75.05 ± 2.64 | 6/6 | 0/0 of 0 |
| detail | 83.64 ± 1.21 | 83.01 ± 2.95 | 83.64 ± 1.21 | 87.42 ± 1.27 | 89.92 ± 0.74 | 76.11 ± 0.22 | 6/4 | 0/0 of 0 |
| detail_global_mix | 82.88 ± 0.89 | 81.61 ± 0.24 | 82.88 ± 0.89 | 86.61 ± 0.91 | 88.41 ± 2.57 | 75.64 ± 2.13 | 5/7 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 61 | +2.79 | [+1.77, +3.95] | +1.49 / +4.10 | 2/2 | 49/12/0 | +1.45 | <0.001 | <0.001 | +2.32 |
| global_mix - baseline | 61 | +0.51 | [-0.61, +1.64] | +0.01 / +1.01 | 2/2 | 34/27/0 | +0.14 | 0.381 | 0.365 | +0.09 |
| detail_global_mix - baseline | 61 | +2.04 | [+0.93, +3.21] | +2.22 / +1.86 | 2/2 | 43/18/0 | +0.71 | <0.001 | <0.001 | +1.58 |
| detail_global_mix - detail | 61 | -0.75 | [-2.11, +0.47] | +0.73 / -2.24 | 1/2 | 28/33/0 | -0.15 | 0.271 | 0.504 | -1.16 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 24 | +3.19 | [+1.43, +5.19] | +1.20 / +5.18 | 2/2 | 20/4/0 | +2.58 | <0.001 | 0.001 | +2.04 |
| global_mix - baseline | 24 | -0.16 | [-2.25, +1.79] | -0.54 / +0.21 | 1/2 | 12/12/0 | +0.02 | 0.879 | 1.000 | -0.98 |
| detail_global_mix - baseline | 24 | +1.79 | [+0.24, +3.42] | +2.06 / +1.53 | 2/2 | 17/7/0 | +1.39 | 0.042 | 0.036 | +1.06 |
| detail_global_mix - detail | 24 | -1.40 | [-4.17, +0.82] | +0.86 / -3.66 | 1/2 | 12/12/0 | +0.07 | 0.338 | 0.721 | -2.21 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +1.49 / +4.10 | +3.52 / +2.43 | +0.57 / +3.77 | +0 / +0 | +0 / -3 |
| global_mix - baseline | +0.01 / +1.01 | -3.38 / +1.46 | +1.53 / +0.69 | +0 / +0 | +0 / -1 |
| detail_global_mix - baseline | +2.22 / +1.86 | +0.72 / +2.22 | +1.76 / +1.65 | +0 / +0 | -1 / +0 |
| detail_global_mix - detail | +0.73 / -2.24 | -2.80 / -0.21 | +1.19 / -2.13 | +0 / +0 | -1 / +3 |

### CVC300 (Test, OOD)

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 80.26 ± 1.40 | 79.48 ± 1.36 | 80.26 ± 1.40 | 83.93 ± 1.49 | 75.06 ± 2.26 | 92.25 ± 0.40 | 7/6 | 0/0 of 0 |
| global_mix | 80.35 ± 0.55 | 79.64 ± 0.48 | 80.35 ± 0.55 | 83.90 ± 0.42 | 72.75 ± 2.54 | 92.82 ± 2.22 | 8/7 | 0/0 of 0 |
| detail | 80.97 ± 0.65 | 80.24 ± 0.72 | 80.97 ± 0.65 | 84.62 ± 0.79 | 76.64 ± 3.12 | 90.73 ± 2.16 | 6/7 | 0/0 of 0 |
| detail_global_mix | 80.84 ± 0.14 | 80.09 ± 0.07 | 80.84 ± 0.14 | 84.43 ± 0.07 | 73.28 ± 2.84 | 92.77 ± 1.29 | 6/6 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 60 | +0.71 | [-0.47, +2.01] | +1.24 / +0.18 | 2/2 | 34/26/0 | +0.29 | 0.275 | 0.268 | +0.08 |
| global_mix - baseline | 60 | +0.09 | [-1.15, +1.25] | +1.47 / -1.30 | 1/2 | 32/28/0 | +0.25 | 0.894 | 0.337 | -0.40 |
| detail_global_mix - baseline | 60 | +0.58 | [-0.77, +1.90] | +1.67 / -0.51 | 1/2 | 36/24/0 | +0.79 | 0.419 | 0.172 | -0.01 |
| detail_global_mix - detail | 60 | -0.13 | [-1.63, +1.36] | +0.43 / -0.69 | 1/2 | 35/24/1 | +0.36 | 0.871 | 0.637 | -0.71 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 56 | +0.76 | [-0.47, +2.10] | +1.21 / +0.31 | 2/2 | 33/23/0 | +0.41 | 0.273 | 0.223 | +0.09 |
| global_mix - baseline | 56 | +0.15 | [-1.13, +1.37] | +1.45 / -1.14 | 1/2 | 30/26/0 | +0.33 | 0.818 | 0.247 | -0.36 |
| detail_global_mix - baseline | 56 | +0.60 | [-0.83, +1.99] | +1.61 / -0.41 | 1/2 | 33/23/0 | +0.70 | 0.425 | 0.196 | -0.03 |
| detail_global_mix - detail | 56 | -0.16 | [-1.76, +1.45] | +0.40 / -0.71 | 1/2 | 32/23/1 | +0.36 | 0.856 | 0.734 | -0.78 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +1.24 / +0.18 | +0.97 / +2.19 | -0.28 / -2.76 | +0 / +0 | -1 / +1 |
| global_mix - baseline | +1.47 / -1.30 | -2.51 / -2.11 | +1.85 / -0.72 | +0 / +0 | +1 / +1 |
| detail_global_mix - baseline | +1.67 / -0.51 | -2.18 / -1.37 | +1.15 / -0.10 | +0 / +0 | -1 / +0 |
| detail_global_mix - detail | +0.43 / -0.69 | -3.16 / -3.56 | +1.43 / +2.66 | +0 / +0 | +0 / -1 |

### BKAI (Test, OOD)

Seeds 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 75.33 ± 1.84 | 69.63 ± 2.36 | 75.33 ± 1.84 | 76.92 ± 1.90 | 69.31 ± 4.20 | 92.05 ± 0.25 | 19/18 | 0/0 of 0 |
| global_mix | 76.25 ± 1.98 | 70.83 ± 2.59 | 76.25 ± 1.98 | 77.87 ± 2.08 | 69.10 ± 4.49 | 92.40 ± 0.19 | 17/16 | 0/0 of 0 |
| detail | 75.94 ± 1.31 | 70.03 ± 2.06 | 75.94 ± 1.31 | 77.45 ± 1.41 | 66.76 ± 2.16 | 92.82 ± 0.51 | 19/15 | 0/0 of 0 |
| detail_global_mix | 77.33 ± 3.25 | 71.28 ± 4.17 | 77.33 ± 3.25 | 78.89 ± 3.35 | 71.85 ± 7.77 | 93.71 ± 0.99 | 20/12 | 0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +0.60 | [-0.65, +1.80] | +0.97 / +0.23 | 2/2 | 67/33/0 | +0.82 | 0.350 | 0.012 | +0.25 |
| global_mix - baseline | 100 | +0.92 | [-0.64, +2.46] | +0.82 / +1.02 | 2/2 | 48/52/0 | -0.04 | 0.264 | 0.342 | +0.26 |
| detail_global_mix - baseline | 100 | +1.99 | [+0.49, +3.42] | +0.99 / +3.00 | 2/2 | 78/22/0 | +1.29 | 0.009 | <0.001 | +1.56 |
| detail_global_mix - detail | 100 | +1.39 | [-0.02, +2.79] | +0.02 / +2.76 | 2/2 | 63/37/0 | +0.33 | 0.056 | 0.012 | +0.92 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 71 | +0.41 | [-1.29, +2.05] | +0.62 / +0.20 | 2/2 | 45/26/0 | +0.95 | 0.640 | 0.120 | -0.10 |
| global_mix - baseline | 71 | +1.21 | [-0.92, +3.40] | +1.04 / +1.37 | 2/2 | 33/38/0 | -0.05 | 0.289 | 0.356 | +0.28 |
| detail_global_mix - baseline | 71 | +1.65 | [-0.38, +3.54] | +0.37 / +2.93 | 2/2 | 50/21/0 | +1.41 | 0.105 | 0.004 | +1.02 |
| detail_global_mix - detail | 71 | +1.24 | [-0.67, +3.03] | -0.25 / +2.73 | 1/2 | 43/28/0 | +0.46 | 0.200 | 0.082 | +0.57 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.97 / +0.23 | -1.10 / -4.00 | +1.31 / +0.23 | +0 / +0 | +0 / -3 |
| global_mix - baseline | +0.82 / +1.02 | -0.42 / -0.02 | +0.40 / +0.31 | +0 / +0 | -2 / -2 |
| detail_global_mix - baseline | +0.99 / +3.00 | +0.01 / +5.06 | +1.14 / +2.19 | +0 / +0 | +1 / -6 |
| detail_global_mix - detail | +0.02 / +2.76 | +1.11 / +9.05 | -0.17 / +1.95 | +0 / +0 | +1 / -3 |

## All seeds combined

### Val (source Val (development))

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 88.50 ± 0.40 | 85.15 ± 0.14 | 88.50 ± 0.40 | 90.46 ± 0.38 | 90.05 ± 0.59 | 87.34 ± 0.69 | 3/3/2/3/3 | 0/0/0/0/0 of 0 |
| global_mix | 88.45 ± 0.60 | 85.23 ± 0.33 | 88.45 ± 0.60 | 90.41 ± 0.57 | 89.90 ± 1.07 | 87.25 ± 1.26 | 3/3/2/2/3 | 0/0/0/0/0 of 0 |
| detail | 88.94 ± 0.44 | 86.14 ± 0.68 | 88.94 ± 0.44 | 90.81 ± 0.42 | 90.89 ± 0.55 | 87.08 ± 0.61 | 3/3/2/4/3 | 0/0/0/0/0 of 0 |
| detail_global_mix | 88.88 ± 0.51 | 86.45 ± 0.21 | 88.88 ± 0.51 | 90.76 ± 0.51 | 90.32 ± 0.93 | 87.35 ± 1.27 | 4/3/3/3/3 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +0.44 | [+0.09, +0.79] | +0.73 / +0.49 / +0.26 / +0.31 / +0.42 | 5/5 | 73/27/0 | +0.41 | 0.015 | <0.001 | +0.34 |
| global_mix - baseline | 100 | -0.05 | [-0.41, +0.29] | +0.00 / +0.03 / -0.12 / +0.24 / -0.41 | 3/5 | 50/50/0 | -0.01 | 0.786 | 0.897 | -0.17 |
| detail_global_mix - baseline | 100 | +0.38 | [-0.08, +0.83] | +0.59 / +0.40 / +0.21 / +0.46 / +0.26 | 5/5 | 64/36/0 | +0.37 | 0.102 | 0.006 | +0.25 |
| detail_global_mix - detail | 100 | -0.06 | [-0.48, +0.36] | -0.14 / -0.09 / -0.05 / +0.15 / -0.15 | 1/5 | 44/56/0 | -0.08 | 0.793 | 0.450 | -0.23 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 25 | +0.99 | [+0.49, +1.48] | +1.61 / +1.80 / +0.29 / +0.46 / +0.80 | 5/5 | 19/6/0 | +1.09 | <0.001 | 0.002 | +0.77 |
| global_mix - baseline | 25 | +0.09 | [-0.13, +0.31] | +0.22 / +0.02 / +0.28 / +0.32 / -0.41 | 4/5 | 11/14/0 | -0.11 | 0.463 | 0.788 | -0.02 |
| detail_global_mix - baseline | 25 | +1.30 | [+0.78, +1.84] | +1.61 / +1.43 / +1.44 / +1.00 / +1.00 | 5/5 | 21/4/0 | +1.31 | <0.001 | <0.001 | +1.04 |
| detail_global_mix - detail | 25 | +0.30 | [-0.07, +0.80] | +0.00 / -0.37 / +1.16 / +0.54 / +0.19 | 4/5 | 13/12/0 | +0.00 | 0.217 | 0.435 | +0.03 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.73 / +0.49 / +0.26 / +0.31 / +0.42 | +0.99 / +1.59 / +0.66 / +0.52 / +0.41 | -0.12 / -0.73 / -0.13 / -0.27 / -0.05 | +0 / +0 / +0 / +0 / +0 | +0 / +0 / +0 / +1 / +0 |
| global_mix - baseline | +0.00 / +0.03 / -0.12 / +0.24 / -0.41 | -0.15 / +0.42 / +0.06 / +0.22 / -1.31 | +0.08 / -0.01 / -1.05 / +0.63 / -0.10 | +0 / +0 / +0 / +0 / +0 | +0 / +0 / +0 / -1 / +0 |
| detail_global_mix - baseline | +0.59 / +0.40 / +0.21 / +0.46 / +0.26 | -0.12 / +1.34 / +0.09 / +0.53 / -0.51 | +1.04 / -1.23 / -0.79 / +0.45 / +0.58 | +0 / +0 / +0 / +0 / +0 | +1 / +0 / +1 / +0 / +0 |
| detail_global_mix - detail | -0.14 / -0.09 / -0.05 / +0.15 / -0.15 | -1.11 / -0.25 / -0.57 / +0.01 / -0.92 | +1.16 / -0.50 / -0.67 / +0.71 / +0.63 | +0 / +0 / +0 / +0 / +0 | +1 / +0 / +1 / -1 / +0 |

### Kvasir (Test, source domain)

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 88.82 ± 0.18 | 87.93 ± 1.12 | 88.82 ± 0.18 | 90.93 ± 0.16 | 91.15 ± 1.38 | 89.09 ± 0.84 | 2/2/2/3/2 | 0/0/0/0/0 of 0 |
| global_mix | 88.75 ± 0.15 | 87.06 ± 1.42 | 88.75 ± 0.15 | 90.81 ± 0.18 | 91.52 ± 0.76 | 88.96 ± 0.52 | 3/3/2/3/2 | 0/0/0/0/0 of 0 |
| detail | 89.54 ± 0.30 | 88.90 ± 1.00 | 89.54 ± 0.30 | 91.55 ± 0.32 | 92.37 ± 1.33 | 88.99 ± 1.04 | 3/2/2/3/3 | 0/0/0/0/0 of 0 |
| detail_global_mix | 89.60 ± 0.32 | 88.62 ± 0.52 | 89.60 ± 0.32 | 91.61 ± 0.30 | 92.23 ± 0.98 | 88.96 ± 0.61 | 1/2/1/3/2 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +0.73 | [+0.37, +1.09] | +0.81 / +0.19 / +0.59 / +0.86 / +1.17 | 5/5 | 71/29/0 | +0.47 | <0.001 | <0.001 | +0.58 |
| global_mix - baseline | 100 | -0.07 | [-0.39, +0.25] | -0.22 / -0.11 / +0.06 / -0.14 / +0.05 | 2/5 | 52/48/0 | +0.02 | 0.664 | 0.945 | -0.20 |
| detail_global_mix - baseline | 100 | +0.78 | [+0.23, +1.33] | +0.95 / +0.78 / +0.62 / +1.07 / +0.50 | 5/5 | 71/29/0 | +0.59 | 0.005 | <0.001 | +0.59 |
| detail_global_mix - detail | 100 | +0.06 | [-0.44, +0.52] | +0.14 / +0.58 / +0.03 / +0.21 / -0.67 | 4/5 | 60/40/0 | +0.12 | 0.816 | 0.127 | -0.07 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 22 | +0.97 | [+0.44, +1.55] | +2.49 / -0.90 / +1.34 / +0.37 / +1.56 | 4/5 | 16/6/0 | +0.52 | 0.002 | 0.008 | +0.72 |
| global_mix - baseline | 22 | -0.87 | [-1.69, -0.11] | -0.67 / -2.13 / -0.21 / -2.08 / +0.74 | 1/5 | 9/13/0 | -0.49 | 0.046 | 0.127 | -1.11 |
| detail_global_mix - baseline | 22 | +0.69 | [-1.00, +2.29] | +2.95 / -0.60 / -0.68 / +0.99 / +0.80 | 3/5 | 14/8/0 | +0.83 | 0.433 | 0.205 | +0.00 |
| detail_global_mix - detail | 22 | -0.28 | [-2.03, +1.22] | +0.46 / +0.30 / -2.02 / +0.62 / -0.77 | 3/5 | 13/9/0 | +0.23 | 0.761 | 0.745 | -0.96 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.81 / +0.19 / +0.59 / +0.86 / +1.17 | +0.73 / +1.35 / +0.83 / +1.03 / +2.14 | -0.77 / +0.11 / -0.77 / +0.93 / -0.01 | +0 / +0 / +0 / +0 / +0 | +1 / +0 / +0 / +0 / +1 |
| global_mix - baseline | -0.22 / -0.11 / +0.06 / -0.14 / +0.05 | +0.13 / +2.24 / +0.52 / -0.97 / -0.07 | -0.48 / -0.47 / +0.17 / +0.53 / -0.41 | +0 / +0 / +0 / +0 / +0 | +1 / +1 / +0 / +0 / +0 |
| detail_global_mix - baseline | +0.95 / +0.78 / +0.62 / +1.07 / +0.50 | +0.19 / +2.15 / +0.91 / +0.12 / +2.01 | -0.21 / -0.77 / +0.32 / +0.86 / -0.88 | +0 / +0 / +0 / +0 / +0 | -1 / +0 / -1 / +0 / +0 |
| detail_global_mix - detail | +0.14 / +0.58 / +0.03 / +0.21 / -0.67 | -0.55 / +0.80 / +0.08 / -0.91 / -0.13 | +0.56 / -0.88 / +1.09 / -0.07 / -0.86 | +0 / +0 / +0 / +0 / +0 | -2 / +0 / -1 / +0 / -1 |

### ColonDB (Test, OOD)

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 69.54 ± 0.51 | 66.81 ± 0.36 | 69.54 ± 0.51 | 72.58 ± 0.53 | 73.82 ± 2.79 | 67.90 ± 3.10 | 77/74/71/74/88 | 0/0/0/0/0 of 0 |
| global_mix | 69.47 ± 0.59 | 66.59 ± 1.03 | 69.47 ± 0.59 | 72.51 ± 0.63 | 73.42 ± 2.36 | 68.37 ± 2.83 | 82/72/68/83/80 | 0/0/0/0/0 of 0 |
| detail | 70.71 ± 0.66 | 68.29 ± 1.09 | 70.71 ± 0.66 | 73.72 ± 0.70 | 74.93 ± 3.23 | 67.81 ± 3.06 | 76/78/76/72/72 | 0/0/0/0/0 of 0 |
| detail_global_mix | 70.84 ± 0.81 | 68.57 ± 1.01 | 70.84 ± 0.81 | 73.89 ± 0.88 | 75.81 ± 1.46 | 67.96 ± 2.68 | 73/65/69/58/80 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 360 | +1.17 | [+0.62, +1.75] | +0.76 / -0.05 / +0.92 / +1.61 / +2.62 | 4/5 | 196/152/12 | +0.50 | <0.001 | <0.001 | +1.05 |
| global_mix - baseline | 360 | -0.07 | [-0.58, +0.42] | -0.87 / -0.49 / +0.55 / -0.35 / +0.82 | 2/5 | 148/201/11 | -0.23 | 0.797 | 0.119 | -0.17 |
| detail_global_mix - baseline | 360 | +1.30 | [+0.65, +1.97] | +1.11 / +0.88 / +0.66 / +2.38 / +1.46 | 5/5 | 187/160/13 | +0.07 | <0.001 | 0.009 | +1.14 |
| detail_global_mix - detail | 360 | +0.13 | [-0.42, +0.69] | +0.35 / +0.93 / -0.26 / +0.77 / -1.17 | 3/5 | 154/193/13 | -0.20 | 0.654 | 0.222 | -0.03 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 241 | +1.49 | [+0.79, +2.19] | +0.24 / +1.50 / +1.52 / +1.55 / +2.62 | 5/5 | 134/95/12 | +0.63 | <0.001 | <0.001 | +1.30 |
| global_mix - baseline | 241 | -0.22 | [-0.91, +0.41] | -1.62 / +0.48 / +0.06 / -0.26 / +0.27 | 3/5 | 94/136/11 | -0.32 | 0.535 | 0.099 | -0.37 |
| detail_global_mix - baseline | 241 | +1.77 | [+0.91, +2.66] | +1.49 / +1.46 / +0.73 / +3.82 / +1.34 | 5/5 | 129/99/13 | +0.07 | <0.001 | 0.002 | +1.54 |
| detail_global_mix - detail | 241 | +0.28 | [-0.38, +0.97] | +1.26 / -0.04 / -0.79 / +2.26 / -1.28 | 2/5 | 107/121/13 | -0.06 | 0.414 | 0.717 | +0.06 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.76 / -0.05 / +0.92 / +1.61 / +2.62 | +4.99 / +0.29 / -1.91 / -0.85 / +3.01 | +0.54 / -3.94 / -0.48 / +1.46 / +2.01 | +0 / +0 / +0 / +0 / +0 | -1 / +4 / +5 / -2 / -16 |
| global_mix - baseline | -0.87 / -0.49 / +0.55 / -0.35 / +0.82 | +1.16 / +0.73 / +1.00 / -3.08 / -1.78 | -0.32 / -2.28 / +2.33 / +0.76 / +1.86 | +0 / +0 / +0 / +0 / +0 | +5 / -2 / -3 / +9 / -8 |
| detail_global_mix - baseline | +1.11 / +0.88 / +0.66 / +2.38 / +1.46 | +4.65 / +0.77 / +2.25 / +1.76 / +0.53 | -2.60 / +0.35 / +0.09 / +0.86 / +1.62 | +0 / +0 / +0 / +0 / +0 | -4 / -9 / -2 / -16 / -8 |
| detail_global_mix - detail | +0.35 / +0.93 / -0.26 / +0.77 / -1.17 | -0.34 / +0.49 / +4.16 / +2.61 / -2.48 | -3.14 / +4.30 / +0.57 / -0.60 / -0.39 | +0 / +0 / +0 / +0 / +0 | -3 / -13 / -7 / -14 / +8 |

### ClinicDB (Test, OOD)

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 81.45 ± 0.64 | 79.87 ± 0.92 | 81.45 ± 0.64 | 85.42 ± 0.66 | 87.21 ± 1.25 | 75.01 ± 1.60 | 4/5/5/6/7 | 0/0/0/0/0 of 0 |
| global_mix | 81.98 ± 0.76 | 80.97 ± 1.85 | 81.98 ± 0.76 | 85.99 ± 0.76 | 87.33 ± 2.97 | 75.03 ± 1.40 | 5/4/4/6/6 | 0/0/0/0/0 of 0 |
| detail | 83.08 ± 1.03 | 82.42 ± 1.95 | 83.08 ± 1.03 | 86.96 ± 0.97 | 88.71 ± 1.18 | 75.96 ± 1.00 | 8/5/3/6/4 | 0/0/0/0/0 of 0 |
| detail_global_mix | 83.20 ± 0.62 | 82.13 ± 1.08 | 83.20 ± 0.62 | 87.04 ± 0.68 | 88.87 ± 1.52 | 76.17 ± 1.21 | 4/6/4/5/7 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 61 | +1.62 | [+0.73, +2.58] | +0.06 / +0.62 / +1.85 / +1.49 / +4.10 | 5/5 | 46/15/0 | +0.94 | <0.001 | <0.001 | +1.19 |
| global_mix - baseline | 61 | +0.53 | [-0.23, +1.34] | +0.08 / +0.20 / +1.32 / +0.01 / +1.01 | 5/5 | 32/29/0 | +0.13 | 0.204 | 0.446 | +0.17 |
| detail_global_mix - baseline | 61 | +1.75 | [+1.09, +2.44] | +2.02 / +1.44 / +1.20 / +2.22 / +1.86 | 5/5 | 49/12/0 | +0.89 | <0.001 | <0.001 | +1.50 |
| detail_global_mix - detail | 61 | +0.12 | [-0.80, +1.04] | +1.96 / +0.82 / -0.65 / +0.73 / -2.24 | 3/5 | 33/28/0 | +0.15 | 0.797 | 0.714 | -0.16 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 24 | +2.54 | [+0.83, +4.49] | +0.66 / +3.23 / +2.44 / +1.20 / +5.18 | 5/5 | 18/6/0 | +1.55 | 0.009 | 0.011 | +1.46 |
| global_mix - baseline | 24 | +1.09 | [-0.38, +2.77] | +1.93 / +1.29 / +2.57 / -0.54 / +0.21 | 4/5 | 12/12/0 | +0.11 | 0.208 | 0.384 | +0.20 |
| detail_global_mix - baseline | 24 | +2.26 | [+1.06, +3.50] | +4.09 / +2.47 / +1.13 / +2.06 / +1.53 | 5/5 | 20/4/0 | +1.58 | 0.001 | 0.002 | +1.74 |
| detail_global_mix - detail | 24 | -0.29 | [-2.20, +1.44] | +3.43 / -0.75 / -1.31 / +0.86 / -3.66 | 2/5 | 13/11/0 | +0.16 | 0.778 | 0.786 | -0.99 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.06 / +0.62 / +1.85 / +1.49 / +4.10 | +1.89 / -0.01 / -0.33 / +3.52 / +2.43 | -1.57 / -0.84 / +2.81 / +0.57 / +3.77 | +0 / +0 / +0 / +0 / +0 | +4 / +0 / -2 / +0 / -3 |
| global_mix - baseline | +0.08 / +0.20 / +1.32 / +0.01 / +1.01 | +0.65 / +1.25 / +0.63 / -3.38 / +1.46 | -2.01 / -1.23 / +1.14 / +1.53 / +0.69 | +0 / +0 / +0 / +0 / +0 | +1 / -1 / -1 / +0 / -1 |
| detail_global_mix - baseline | +2.02 / +1.44 / +1.20 / +2.22 / +1.86 | +3.23 / +2.00 / +0.11 / +0.72 / +2.22 | -0.01 / +0.23 / +2.17 / +1.76 / +1.65 | +0 / +0 / +0 / +0 / +0 | +0 / +1 / -1 / -1 / +0 |
| detail_global_mix - detail | +1.96 / +0.82 / -0.65 / +0.73 / -2.24 | +1.34 / +2.01 / +0.44 / -2.80 / -0.21 | +1.56 / +1.08 / -0.64 / +1.19 / -2.13 | +0 / +0 / +0 / +0 / +0 | -4 / +1 / +1 / -1 / +3 |

### CVC300 (Test, OOD)

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 79.61 ± 0.92 | 78.78 ± 0.94 | 79.61 ± 0.92 | 83.22 ± 0.99 | 73.93 ± 2.34 | 92.38 ± 1.73 | 6/6/7/7/6 | 0/0/0/0/0 of 0 |
| global_mix | 79.58 ± 0.80 | 78.88 ± 0.81 | 79.58 ± 0.80 | 83.12 ± 0.80 | 73.58 ± 2.21 | 92.32 ± 1.48 | 7/6/6/8/7 | 0/0/0/0/0 of 0 |
| detail | 79.72 ± 1.36 | 78.89 ± 1.53 | 79.72 ± 1.36 | 83.25 ± 1.47 | 74.80 ± 2.64 | 91.49 ± 1.74 | 7/7/7/6/7 | 0/0/0/0/0 of 0 |
| detail_global_mix | 80.45 ± 0.77 | 79.73 ± 0.94 | 80.45 ± 0.77 | 83.99 ± 0.76 | 73.36 ± 1.65 | 92.89 ± 0.87 | 6/4/7/6/6 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 60 | +0.11 | [-1.06, +1.25] | -1.35 / +0.10 / +0.39 / +1.24 / +0.18 | 4/5 | 27/33/0 | -0.14 | 0.852 | 0.950 | -0.34 |
| global_mix - baseline | 60 | -0.03 | [-0.86, +0.75] | -0.33 / -0.20 / +0.22 / +1.47 / -1.30 | 2/5 | 30/30/0 | +0.01 | 0.953 | 0.482 | -0.30 |
| detail_global_mix - baseline | 60 | +0.84 | [-0.29, +2.14] | +0.03 / +2.07 / +0.95 / +1.67 / -0.51 | 4/5 | 27/33/0 | -0.11 | 0.192 | 0.716 | +0.23 |
| detail_global_mix - detail | 60 | +0.73 | [-0.47, +2.29] | +1.37 / +1.96 / +0.56 / +0.43 / -0.69 | 4/5 | 32/28/0 | +0.26 | 0.374 | 0.648 | -0.14 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 56 | +0.11 | [-1.15, +1.32] | -1.60 / +0.32 / +0.31 / +1.21 / +0.31 | 4/5 | 25/31/0 | -0.14 | 0.866 | 0.951 | -0.37 |
| global_mix - baseline | 56 | +0.10 | [-0.77, +0.88] | -0.21 / -0.06 / +0.47 / +1.45 / -1.14 | 2/5 | 29/27/0 | +0.04 | 0.820 | 0.302 | -0.19 |
| detail_global_mix - baseline | 56 | +0.96 | [-0.24, +2.26] | -0.06 / +2.44 / +1.20 / +1.61 / -0.41 | 3/5 | 24/32/0 | -0.17 | 0.156 | 0.704 | +0.30 |
| detail_global_mix - detail | 56 | +0.85 | [-0.43, +2.51] | +1.54 / +2.13 / +0.89 / +0.40 / -0.71 | 4/5 | 31/25/0 | +0.32 | 0.319 | 0.463 | -0.08 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | -1.35 / +0.10 / +0.39 / +1.24 / +0.18 | +1.23 / -0.13 / +0.09 / +0.97 / +2.19 | -1.76 / -0.30 / +0.63 / -0.28 / -2.76 | +0 / +0 / +0 / +0 / +0 | +1 / +1 / +0 / -1 / +1 |
| global_mix - baseline | -0.33 / -0.20 / +0.22 / +1.47 / -1.30 | +4.16 / +0.37 / -1.66 / -2.51 / -2.11 | -2.41 / +0.13 / +0.81 / +1.85 / -0.72 | +0 / +0 / +0 / +0 / +0 | +1 / +0 / -1 / +1 / +1 |
| detail_global_mix - baseline | +0.03 / +2.07 / +0.95 / +1.67 / -0.51 | +3.85 / -2.08 / -1.05 / -2.18 / -1.37 | -1.36 / +2.49 / +0.33 / +1.15 / -0.10 | +0 / +0 / +0 / +0 / +0 | +0 / -2 / +0 / -1 / +0 |
| detail_global_mix - detail | +1.37 / +1.96 / +0.56 / +0.43 / -0.69 | +2.62 / -1.95 / -1.14 / -3.16 / -3.56 | +0.40 / +2.79 / -0.29 / +1.43 / +2.66 | +0 / +0 / +0 / +0 / +0 | -1 / -3 / +0 / +0 / -1 |

### BKAI (Test, OOD)

Seeds 666, 667, 668, 669, 670; mean ± sample SD over seeds.

| Arm | DSC % | Small DSC % | Positive DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Normal images with FP (per seed) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 74.77 ± 1.19 | 68.65 ± 1.60 | 74.77 ± 1.19 | 76.33 ± 1.23 | 67.87 ± 3.89 | 92.72 ± 0.78 | 20/21/19/19/18 | 0/0/0/0/0 of 0 |
| global_mix | 75.84 ± 1.07 | 70.06 ± 1.49 | 75.84 ± 1.07 | 77.43 ± 1.13 | 68.93 ± 2.43 | 92.99 ± 0.81 | 16/17/16/17/16 | 0/0/0/0/0 of 0 |
| detail | 75.92 ± 1.13 | 69.97 ± 1.65 | 75.92 ± 1.13 | 77.43 ± 1.16 | 66.55 ± 1.34 | 93.63 ± 0.89 | 14/17/20/19/15 | 0/0/0/0/0 of 0 |
| detail_global_mix | 77.01 ± 1.76 | 71.03 ± 2.28 | 77.01 ± 1.76 | 78.55 ± 1.81 | 71.36 ± 4.47 | 93.83 ± 0.69 | 16/19/16/20/12 | 0/0/0/0/0 of 0 |

**Paired DSC differences, all images** (seed-averaged per image).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 100 | +1.15 | [+0.35, +2.00] | +3.49 / +0.68 / +0.39 / +0.97 / +0.23 | 5/5 | 70/30/0 | +0.66 | 0.007 | <0.001 | +0.85 |
| global_mix - baseline | 100 | +1.07 | [+0.10, +2.12] | +1.81 / +1.20 / +0.52 / +0.82 / +1.02 | 5/5 | 57/43/0 | +0.08 | 0.042 | 0.103 | +0.61 |
| detail_global_mix - baseline | 100 | +2.25 | [+1.23, +3.28] | +3.38 / +1.75 / +2.13 / +0.99 / +3.00 | 5/5 | 80/20/0 | +1.26 | <0.001 | <0.001 | +1.84 |
| detail_global_mix - detail | 100 | +1.09 | [+0.04, +2.16] | -0.12 / +1.07 / +1.73 / +0.02 / +2.76 | 4/5 | 63/37/0 | +0.37 | 0.044 | 0.013 | +0.72 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| detail - baseline | 71 | +1.32 | [+0.21, +2.52] | +4.73 / +0.44 / +0.62 / +0.62 / +0.20 | 5/5 | 46/25/0 | +0.85 | 0.027 | 0.011 | +0.89 |
| global_mix - baseline | 71 | +1.41 | [+0.06, +2.90] | +2.71 / +1.32 / +0.62 / +1.04 / +1.37 | 5/5 | 41/30/0 | +0.21 | 0.054 | 0.077 | +0.77 |
| detail_global_mix - baseline | 71 | +2.38 | [+1.05, +3.78] | +4.14 / +1.51 / +2.97 / +0.37 / +2.93 | 5/5 | 53/18/0 | +1.40 | <0.001 | <0.001 | +1.81 |
| detail_global_mix - detail | 71 | +1.06 | [-0.33, +2.39] | -0.60 / +1.07 / +2.35 / -0.25 / +2.73 | 3/5 | 44/27/0 | +0.63 | 0.136 | 0.065 | +0.59 |

**Other changes.**

| Contrast | Positive-only DSC pp (per seed) | Precision pp (per seed) | Recall pp (per seed) | Normal-image FP count change | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +3.49 / +0.68 / +0.39 / +0.97 / +0.23 | +3.78 / -0.00 / -5.30 / -1.10 / -4.00 | +0.87 / +0.74 / +1.39 / +1.31 / +0.23 | +0 / +0 / +0 / +0 / +0 | -6 / -4 / +1 / +0 / -3 |
| global_mix - baseline | +1.81 / +1.20 / +0.52 / +0.82 / +1.02 | +5.04 / +2.41 / -1.74 / -0.42 / -0.02 | +0.31 / -0.29 / +0.64 / +0.40 / +0.31 | +0 / +0 / +0 / +0 / +0 | -4 / -4 / -3 / -2 / -2 |
| detail_global_mix - baseline | +3.38 / +1.75 / +2.13 / +0.99 / +3.00 | +9.31 / +0.12 / +2.95 / +0.01 / +5.06 | +0.56 / +1.20 / +0.46 / +1.14 / +2.19 | +0 / +0 / +0 / +0 / +0 | -4 / -2 / -3 / +1 / -6 |
| detail_global_mix - detail | -0.12 / +1.07 / +1.73 / +0.02 / +2.76 | +5.53 / +0.12 / +8.25 / +1.11 / +9.05 | -0.30 / +0.45 / -0.93 / -0.17 / +1.95 | +0 / +0 / +0 / +0 / +0 | +2 / +2 / -4 / +1 / -3 |

## Limitations

- Image-level tests are conditional on the trained seeds, not patient clustered, and not corrected for multiple comparisons.
- The five test sets were used before; seeds 666-668 results on them are already known.
- ColonDB and CVC300 share 52 exact image hashes and are not independent cohorts.
