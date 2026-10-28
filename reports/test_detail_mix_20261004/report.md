# Test/OOD evaluation of the detail and global_mix arms (generic prompt)

Suite `test_detail_mix_kvasir_v1`; twelve accepted epoch-100 checkpoints; MC30, batch 32, threshold 0.5; fixed text `a polyp in a colonoscopy image`.
Evaluation only: no training, no checkpoint, threshold or prompt selection. The gate below was fixed before any of these results existed.

## Gate (detail - baseline): **PASS**

| Criterion | Value | Result |
|---|---|---|
| G1 Kvasir Test DSC gain > 0 pp | +0.53 pp | PASS |
| G2 positive DSC gain on >= 3 of 4 OOD sets | 3 (ColonDB, ClinicDB, BKAI) | PASS |
| G3 macro small-target DSC gain > 0 pp | +1.16 pp | PASS |
| Guard: no set worse than -1.0 pp | worst -0.28 pp | PASS |

| Dataset | n | n small | detail - baseline DSC pp | Small DSC pp | Unique-image DSC pp |
|---|---:|---:|---:|---:|---:|
| Kvasir | 100 | 22 | +0.53 | +0.97 | +0.53 (100 images) |
| ColonDB | 360 | 241 | +0.54 | +1.09 | +0.31 (329 images) |
| ClinicDB | 61 | 24 | +0.84 | +2.11 | +0.84 (61 images) |
| CVC300 | 60 | 56 | -0.28 | -0.32 | -0.25 (58 images) |
| BKAI | 100 | 71 | +1.52 | +1.93 | +1.52 (100 images) |
| Macro mean, 5 sets | | | +0.63 | +1.16 | |
| Macro mean, 4 OOD sets | | | +0.66 | | |

## Kvasir Test (source domain)

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 88.93 ± 0.12 | 88.10 ± 1.52 | 91.04 ± 0.07 | 90.90 ± 1.89 | 89.32 ± 1.06 | 2/2/2 | 0/0/0 |
| global_mix | 88.85 ± 0.03 | 87.10 ± 1.68 | 90.91 ± 0.09 | 91.86 ± 0.80 | 89.06 ± 0.69 | 3/3/2 | 0/0/0 |
| detail | 89.47 ± 0.37 | 89.08 ± 1.34 | 91.46 ± 0.37 | 91.87 ± 1.56 | 88.84 ± 1.43 | 3/2/2 | 0/0/0 |
| detail_global_mix | 89.72 ± 0.28 | 88.66 ± 0.59 | 91.74 ± 0.23 | 91.98 ± 1.02 | 89.10 ± 0.59 | 1/2/1 | 0/0/0 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +0.53 | [+0.14, +0.91] | +0.81 / +0.19 / +0.59 | 3/3 | 67/33/0 | +0.56 | 0.008 | <0.001 | +0.12 | +0.42 |
| global_mix - baseline | 100 | -0.09 | [-0.52, +0.27] | -0.22 / -0.11 / +0.06 | 1/3 | 49/51/0 | -0.03 | 0.703 | 0.959 | +0.08 | -0.18 |
| detail_global_mix - baseline | 100 | +0.78 | [+0.31, +1.28] | +0.95 / +0.78 / +0.62 | 3/3 | 67/33/0 | +0.56 | 0.002 | <0.001 | +0.18 | +0.62 |
| detail_global_mix - detail | 100 | +0.25 | [-0.18, +0.71] | +0.14 / +0.58 / +0.03 | 3/3 | 58/42/0 | +0.15 | 0.279 | 0.108 | +0.18 | +0.07 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 22 | +0.97 | [+0.31, +1.67] | +2.49 / -0.90 / +1.34 | 2/3 | 17/5/0 | +0.69 | 0.011 | 0.027 | +0.40 | +0.63 |
| global_mix - baseline | 22 | -1.00 | [-2.58, +0.13] | -0.67 / -2.13 / -0.21 | 0/3 | 9/13/0 | -0.46 | 0.190 | 0.346 | +0.16 | -1.28 |
| detail_global_mix - baseline | 22 | +0.55 | [-0.79, +1.91] | +2.95 / -0.60 / -0.68 | 1/3 | 14/8/0 | +0.47 | 0.443 | 0.436 | +0.60 | -0.05 |
| detail_global_mix - detail | 22 | -0.42 | [-1.83, +1.00] | +0.46 / +0.30 / -2.02 | 2/3 | 9/13/0 | -0.20 | 0.578 | 0.516 | +0.58 | -1.10 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +0.42 | [+0.02, +0.81] | +0.71 / +0.06 / +0.50 | 3/3 | 65/35/0 | +0.50 | 0.036 | 0.002 | +0.12 | +0.31 |
| global_mix - baseline | 100 | -0.13 | [-0.58, +0.23] | -0.28 / -0.15 / +0.04 | 1/3 | 47/53/0 | -0.05 | 0.576 | 0.769 | +0.08 | -0.22 |
| detail_global_mix - baseline | 100 | +0.69 | [+0.20, +1.18] | +0.86 / +0.68 / +0.54 | 3/3 | 67/33/0 | +0.50 | 0.005 | <0.001 | +0.19 | +0.51 |
| detail_global_mix - detail | 100 | +0.27 | [-0.18, +0.73] | +0.14 / +0.63 / +0.04 | 3/3 | 61/39/0 | +0.14 | 0.246 | 0.078 | +0.19 | +0.08 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.73 / +1.35 / +0.83 | -0.77 / +0.11 / -0.77 | -10.7 / -13.2 / -11.9 | +7.5 / -1.1 / +6.5 | +1 / +0 / +0 |
| global_mix - baseline | +0.13 / +2.24 / +0.52 | -0.48 / -0.47 / +0.17 | -2.2 / -22.2 / -6.8 | +4.7 / +4.8 / -1.4 | +1 / +1 / +0 |
| detail_global_mix - baseline | +0.19 / +2.15 / +0.91 | -0.21 / -0.77 / +0.32 | -2.8 / -21.7 / -11.9 | +2.0 / +7.8 / -2.7 | -1 / +0 / -1 |
| detail_global_mix - detail | -0.55 / +0.80 / +0.08 | +0.56 / -0.88 / +1.09 | +8.9 / -9.7 / +0.0 | -5.1 / +9.0 / -8.6 | -2 / +0 / -1 |

## ColonDB Test (OOD)

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 69.72 ± 0.28 | 66.78 ± 0.14 | 72.76 ± 0.31 | 72.52 ± 2.55 | 68.94 ± 0.39 | 77/74/71 | 18/24/27 |
| global_mix | 69.45 ± 0.84 | 66.41 ± 1.21 | 72.51 ± 0.89 | 73.48 ± 2.38 | 68.85 ± 2.04 | 82/72/68 | 24/21/23 |
| detail | 70.26 ± 0.34 | 67.86 ± 0.82 | 73.24 ± 0.32 | 73.65 ± 1.32 | 67.64 ± 2.38 | 76/78/76 | 22/26/24 |
| detail_global_mix | 70.61 ± 0.27 | 68.01 ± 0.51 | 73.61 ± 0.32 | 75.08 ± 1.07 | 68.22 ± 1.40 | 73/65/69 | 23/23/29 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 360 | +0.54 | [-0.18, +1.25] | +0.76 / -0.05 / +0.92 | 2/3 | 197/151/12 | +0.28 | 0.137 | 0.010 | +0.15 | +0.39 |
| global_mix - baseline | 360 | -0.27 | [-1.01, +0.44] | -0.87 / -0.49 / +0.55 | 1/3 | 156/189/15 | -0.11 | 0.477 | 0.191 | +0.15 | -0.42 |
| detail_global_mix - baseline | 360 | +0.89 | [+0.11, +1.68] | +1.11 / +0.88 / +0.66 | 3/3 | 187/160/13 | +0.09 | 0.032 | 0.046 | +0.20 | +0.69 |
| detail_global_mix - detail | 360 | +0.34 | [-0.38, +1.08] | +0.35 / +0.93 / -0.26 | 2/3 | 163/184/13 | -0.03 | 0.358 | 0.656 | +0.19 | +0.15 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 241 | +1.09 | [+0.23, +1.91] | +0.24 / +1.50 / +1.52 | 3/3 | 133/96/12 | +0.33 | 0.011 | 0.004 | +0.23 | +0.87 |
| global_mix - baseline | 241 | -0.36 | [-1.32, +0.53] | -1.62 / +0.48 / +0.06 | 2/3 | 110/117/14 | +0.00 | 0.448 | 0.793 | +0.22 | -0.58 |
| detail_global_mix - baseline | 241 | +1.23 | [+0.22, +2.27] | +1.49 / +1.46 / +0.73 | 3/3 | 130/98/13 | +0.28 | 0.021 | 0.016 | +0.30 | +0.94 |
| detail_global_mix - detail | 241 | +0.14 | [-0.76, +1.05] | +1.26 / -0.04 / -0.79 | 1/3 | 112/116/13 | +0.00 | 0.761 | 0.822 | +0.28 | -0.14 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 360 | +0.48 | [-0.25, +1.21] | +0.67 / -0.12 / +0.89 | 2/3 | 197/151/12 | +0.26 | 0.205 | 0.020 | +0.16 | +0.32 |
| global_mix - baseline | 360 | -0.25 | [-1.04, +0.50] | -0.94 / -0.41 / +0.59 | 1/3 | 162/185/13 | -0.16 | 0.521 | 0.181 | +0.16 | -0.41 |
| detail_global_mix - baseline | 360 | +0.85 | [+0.04, +1.67] | +1.07 / +0.86 / +0.62 | 3/3 | 185/162/13 | +0.09 | 0.046 | 0.077 | +0.22 | +0.63 |
| detail_global_mix - detail | 360 | +0.37 | [-0.38, +1.13] | +0.40 / +0.98 / -0.27 | 2/3 | 164/183/13 | -0.02 | 0.340 | 0.682 | +0.20 | +0.17 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +4.99 / +0.29 / -1.91 | +0.54 / -3.94 / -0.48 | -21.4 / -7.1 / +9.4 | -1.8 / +12.7 / +1.5 | -1 / +4 / +5 |
| global_mix - baseline | +1.16 / +0.73 / +1.00 | -0.32 / -2.28 / +2.33 | -5.8 / -6.9 / -1.9 | +1.0 / +7.4 / -7.4 | +5 / -2 / -3 |
| detail_global_mix - baseline | +4.65 / +0.77 / +2.25 | -2.60 / +0.35 / +0.09 | -23.6 / -3.5 / -11.3 | +8.4 / -1.1 / -0.3 | -4 / -9 / -2 |
| detail_global_mix - detail | -0.34 / +0.49 / +4.16 | -3.14 / +4.30 / +0.57 | -2.8 / +3.9 / -18.9 | +10.4 / -12.3 / -1.8 | -3 / -13 / -7 |

## ClinicDB Test (OOD)

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 81.86 ± 0.09 | 79.91 ± 1.29 | 85.84 ± 0.17 | 87.39 ± 1.37 | 75.72 ± 1.04 | 4/5/5 | 0/0/0 |
| global_mix | 82.39 ± 0.72 | 81.85 ± 1.93 | 86.44 ± 0.60 | 88.23 ± 1.58 | 75.03 ± 0.66 | 5/4/4 | 0/0/0 |
| detail | 82.70 ± 0.93 | 82.02 ± 1.63 | 86.65 ± 0.84 | 87.91 ± 0.28 | 75.86 ± 1.39 | 8/5/3 | 0/0/0 |
| detail_global_mix | 83.41 ± 0.46 | 82.48 ± 1.36 | 87.33 ± 0.44 | 89.17 ± 1.00 | 76.52 ± 0.43 | 4/6/4 | 0/0/0 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 61 | +0.84 | [-0.25, +1.98] | +0.06 / +0.62 / +1.85 | 3/3 | 39/22/0 | +0.54 | 0.149 | 0.062 | +0.53 | +0.33 |
| global_mix - baseline | 61 | +0.53 | [-0.61, +1.74] | +0.08 / +0.20 / +1.32 | 3/3 | 34/27/0 | +0.20 | 0.395 | 0.393 | +0.52 | +0.02 |
| detail_global_mix - baseline | 61 | +1.55 | [+0.81, +2.40] | +2.02 / +1.44 / +1.20 | 3/3 | 42/19/0 | +0.79 | <0.001 | <0.001 | +0.41 | +1.18 |
| detail_global_mix - detail | 61 | +0.71 | [-0.43, +1.81] | +1.96 / +0.82 / -0.65 | 2/3 | 35/26/0 | +0.16 | 0.230 | 0.112 | +0.34 | +0.38 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 24 | +2.11 | [+0.17, +4.36] | +0.66 / +3.23 / +2.44 | 3/3 | 14/10/0 | +0.53 | 0.060 | 0.126 | +1.34 | +0.84 |
| global_mix - baseline | 24 | +1.93 | [-0.02, +4.36] | +1.93 / +1.29 / +2.57 | 3/3 | 16/8/0 | +0.55 | 0.091 | 0.126 | +1.31 | +0.68 |
| detail_global_mix - baseline | 24 | +2.56 | [+0.93, +4.41] | +4.09 / +2.47 / +1.13 | 3/3 | 16/8/0 | +1.14 | 0.006 | 0.016 | +1.05 | +1.66 |
| detail_global_mix - detail | 24 | +0.46 | [-2.11, +2.66] | +3.43 / -0.75 / -1.31 | 1/3 | 14/10/0 | +0.32 | 0.728 | 0.284 | +0.86 | -0.44 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 61 | +0.81 | [-0.23, +1.93] | +0.02 / +0.81 / +1.59 | 3/3 | 39/22/0 | +0.37 | 0.160 | 0.085 | +0.54 | +0.28 |
| global_mix - baseline | 61 | +0.60 | [-0.55, +1.83] | +0.21 / +0.42 / +1.16 | 3/3 | 36/25/0 | +0.16 | 0.341 | 0.325 | +0.55 | +0.05 |
| detail_global_mix - baseline | 61 | +1.48 | [+0.72, +2.35] | +1.90 / +1.58 / +0.97 | 3/3 | 39/22/0 | +0.66 | <0.001 | 0.001 | +0.46 | +1.06 |
| detail_global_mix - detail | 61 | +0.68 | [-0.53, +1.81] | +1.88 / +0.77 / -0.62 | 2/3 | 37/24/0 | +0.27 | 0.267 | 0.067 | +0.35 | +0.34 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +1.89 / -0.01 / -0.33 | -1.57 / -0.84 / +2.81 | -16.9 / -1.1 / +7.0 | +6.7 / +3.5 / -11.1 | +4 / +0 / -2 |
| global_mix - baseline | +0.65 / +1.25 / +0.63 | -2.01 / -1.23 / +1.14 | -7.7 / -13.3 / -4.5 | +8.6 / +5.1 / -4.5 | +1 / -1 / -1 |
| detail_global_mix - baseline | +3.23 / +2.00 / +0.11 | -0.01 / +0.23 / +2.17 | -25.6 / -18.6 / +1.8 | +0.0 / -1.0 / -8.6 | +0 / +1 / -1 |
| detail_global_mix - detail | +1.34 / +2.01 / +0.44 | +1.56 / +1.08 / -0.64 | -10.4 / -17.7 / -4.9 | -6.3 / -4.3 / +2.8 | -4 / +1 / +1 |

## CVC300 Test (OOD)

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 79.17 ± 0.12 | 78.30 ± 0.14 | 82.75 ± 0.13 | 73.17 ± 2.49 | 92.48 ± 2.42 | 6/6/7 | 1/0/0 |
| global_mix | 79.07 ± 0.38 | 78.37 ± 0.49 | 82.60 ± 0.40 | 74.13 ± 2.33 | 91.99 ± 1.22 | 7/6/6 | 2/2/0 |
| detail | 78.89 ± 0.95 | 77.98 ± 1.17 | 82.33 ± 0.94 | 73.57 ± 1.83 | 92.00 ± 1.67 | 7/7/7 | 1/2/0 |
| detail_global_mix | 80.19 ± 0.96 | 79.50 ± 1.25 | 83.71 ± 0.92 | 73.41 ± 1.19 | 92.96 ± 0.82 | 6/4/7 | 1/1/0 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 60 | -0.28 | [-1.68, +0.97] | -1.35 / +0.10 / +0.39 | 2/3 | 27/33/0 | -0.34 | 0.700 | 0.588 | +0.43 | -0.73 |
| global_mix - baseline | 60 | -0.10 | [-1.32, +0.96] | -0.33 / -0.20 / +0.22 | 1/3 | 30/30/0 | -0.00 | 0.881 | 0.688 | +0.33 | -0.45 |
| detail_global_mix - baseline | 60 | +1.01 | [-0.57, +2.62] | +0.03 / +2.07 / +0.95 | 3/3 | 33/27/0 | +0.15 | 0.225 | 0.361 | +0.72 | +0.30 |
| detail_global_mix - detail | 60 | +1.30 | [-0.13, +3.16] | +1.37 / +1.96 / +0.56 | 3/3 | 37/23/0 | +0.57 | 0.142 | 0.215 | +1.08 | +0.23 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 56 | -0.32 | [-1.81, +1.01] | -1.60 / +0.32 / +0.31 | 2/3 | 25/31/0 | -0.34 | 0.679 | 0.533 | +0.46 | -0.81 |
| global_mix - baseline | 56 | +0.06 | [-1.24, +1.16] | -0.21 / -0.06 / +0.47 | 1/3 | 29/27/0 | +0.13 | 0.926 | 0.417 | +0.36 | -0.30 |
| detail_global_mix - baseline | 56 | +1.19 | [-0.47, +2.82] | -0.06 / +2.44 / +1.20 | 2/3 | 32/24/0 | +0.24 | 0.176 | 0.289 | +0.77 | +0.44 |
| detail_global_mix - detail | 56 | +1.52 | [+0.01, +3.46] | +1.54 / +2.13 / +0.89 | 3/3 | 36/20/0 | +0.60 | 0.088 | 0.087 | +1.15 | +0.38 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 60 | -0.43 | [-1.86, +0.88] | -1.34 / -0.17 / +0.23 | 1/3 | 28/32/0 | -0.38 | 0.572 | 0.437 | +0.44 | -0.90 |
| global_mix - baseline | 60 | -0.16 | [-1.57, +1.01] | -0.32 / -0.33 / +0.18 | 1/3 | 30/30/0 | +0.00 | 0.836 | 0.688 | +0.36 | -0.53 |
| detail_global_mix - baseline | 60 | +0.95 | [-0.72, +2.59] | +0.17 / +1.86 / +0.82 | 3/3 | 31/29/0 | +0.08 | 0.278 | 0.529 | +0.74 | +0.21 |
| detail_global_mix - detail | 60 | +1.38 | [-0.08, +3.26] | +1.50 / +2.03 / +0.60 | 3/3 | 38/22/0 | +0.53 | 0.122 | 0.199 | +1.10 | +0.29 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +1.23 / -0.13 / +0.09 | -1.76 / -0.30 / +0.63 | -7.6 / +0.4 / +0.2 | +36.2 / +3.1 / -7.6 | +1 / +1 / +0 |
| global_mix - baseline | +4.16 / +0.37 / -1.66 | -2.41 / +0.13 / +0.81 | -21.0 / -1.9 / +9.6 | +49.7 / -1.4 / -9.9 | +1 / +0 / -1 |
| detail_global_mix - baseline | +3.85 / -2.08 / -1.05 | -1.36 / +2.49 / +0.33 | -18.8 / +14.7 / +5.8 | +28.1 / -26.1 / -4.1 | +0 / -2 / +0 |
| detail_global_mix - detail | +2.62 / -1.95 / -1.14 | +0.40 / +2.79 / -0.29 | -12.1 / +14.3 / +5.6 | -6.0 / -28.3 / +3.9 | -1 / -3 / +0 |

## BKAI Test (OOD)

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 74.39 ± 0.77 | 68.00 ± 0.84 | 75.94 ± 0.81 | 66.91 ± 4.23 | 93.17 ± 0.67 | 20/21/19 | 2/1/2 |
| global_mix | 75.56 ± 0.27 | 69.55 ± 0.33 | 77.14 ± 0.24 | 68.82 ± 1.31 | 93.39 ± 0.85 | 16/17/16 | 2/2/2 |
| detail | 75.91 ± 1.30 | 69.93 ± 1.82 | 77.42 ± 1.30 | 66.41 ± 1.08 | 94.17 ± 0.61 | 14/17/20 | 1/4/1 |
| detail_global_mix | 76.81 ± 0.86 | 70.87 ± 1.27 | 78.32 ± 0.86 | 71.04 ± 3.06 | 93.91 ± 0.66 | 16/19/16 | 1/3/3 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +1.52 | [+0.56, +2.51] | +3.49 / +0.68 / +0.39 | 3/3 | 63/37/0 | +0.85 | 0.003 | 0.002 | +0.38 | +1.16 |
| global_mix - baseline | 100 | +1.18 | [+0.08, +2.52] | +1.81 / +1.20 / +0.52 | 3/3 | 54/46/0 | +0.08 | 0.053 | 0.247 | +0.69 | +0.50 |
| detail_global_mix - baseline | 100 | +2.42 | [+1.29, +3.69] | +3.38 / +1.75 / +2.13 | 3/3 | 76/24/0 | +1.12 | <0.001 | <0.001 | +0.57 | +1.88 |
| detail_global_mix - detail | 100 | +0.90 | [-0.19, +2.04] | -0.12 / +1.07 / +1.73 | 2/3 | 58/41/1 | +0.32 | 0.126 | 0.075 | +0.39 | +0.51 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 71 | +1.93 | [+0.60, +3.34] | +4.73 / +0.44 / +0.62 | 3/3 | 44/27/0 | +1.31 | 0.006 | 0.005 | +0.54 | +1.43 |
| global_mix - baseline | 71 | +1.55 | [+0.08, +3.37] | +2.71 / +1.32 / +0.62 | 3/3 | 39/32/0 | +0.16 | 0.074 | 0.230 | +0.96 | +0.60 |
| detail_global_mix - baseline | 71 | +2.87 | [+1.42, +4.55] | +4.14 / +1.51 / +2.97 | 3/3 | 51/20/0 | +1.91 | <0.001 | <0.001 | +0.80 | +2.13 |
| detail_global_mix - detail | 71 | +0.94 | [-0.53, +2.43] | -0.60 / +1.07 / +2.35 | 2/3 | 42/29/0 | +0.48 | 0.220 | 0.234 | +0.55 | +0.40 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +1.47 | [+0.51, +2.52] | +3.51 / +0.59 / +0.31 | 3/3 | 63/37/0 | +0.73 | 0.005 | 0.003 | +0.39 | +1.11 |
| global_mix - baseline | 100 | +1.20 | [+0.10, +2.53] | +1.88 / +1.22 / +0.49 | 3/3 | 53/47/0 | +0.06 | 0.056 | 0.231 | +0.70 | +0.51 |
| detail_global_mix - baseline | 100 | +2.38 | [+1.23, +3.70] | +3.41 / +1.67 / +2.06 | 3/3 | 78/22/0 | +1.06 | <0.001 | <0.001 | +0.59 | +1.83 |
| detail_global_mix - detail | 100 | +0.91 | [-0.21, +2.10] | -0.10 / +1.08 / +1.74 | 2/3 | 57/42/1 | +0.32 | 0.131 | 0.074 | +0.41 | +0.51 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +3.78 / -0.00 / -5.30 | +0.87 / +0.74 / +1.39 | -14.4 / +0.8 / +29.6 | -14.3 / -10.3 / -19.2 | -6 / -4 / +1 |
| global_mix - baseline | +5.04 / +2.41 / -1.74 | +0.31 / -0.29 / +0.64 | -19.6 / -10.9 / +9.4 | -5.0 / +4.1 / -8.9 | -4 / -4 / -3 |
| detail_global_mix - baseline | +9.31 / +0.12 / +2.95 | +0.56 / +1.20 / +0.46 | -34.1 / +0.8 / -13.3 | -9.3 / -16.7 / -6.3 | -4 / -2 / -3 |
| detail_global_mix - detail | +5.53 / +0.12 / +8.25 | -0.30 / +0.45 / -0.93 | -23.0 / -0.1 / -33.1 | +5.9 / -7.1 / +15.9 | +2 / +2 / -4 |

## Val reproduction check on this machine

| Run | Accepted Val DSC % | Local Val DSC % | Difference pp |
|---|---:|---:|---:|
| baseline_seed666 | 88.4621 | 88.4586 | -0.0035 |
| global_mix_seed666 | 88.4660 | 88.4644 | -0.0016 |
| detail_seed666 | 89.1884 | 89.1971 | +0.0087 |
| detail_global_mix_seed666 | 89.0510 | 89.0569 | +0.0059 |
| baseline_seed667 | 88.6115 | 88.6159 | +0.0044 |
| global_mix_seed667 | 88.6450 | 88.6473 | +0.0023 |
| detail_seed667 | 89.0982 | 89.1085 | +0.0103 |
| detail_global_mix_seed667 | 89.0087 | 89.0146 | +0.0059 |
| baseline_seed668 | 88.0234 | 88.0148 | -0.0086 |
| global_mix_seed668 | 87.9078 | 87.9082 | +0.0004 |
| detail_seed668 | 88.2842 | 88.2750 | -0.0092 |
| detail_global_mix_seed668 | 88.2306 | 88.2369 | +0.0063 |

## Limitations

- The five test sets were already used with earlier attribute-prompt models; they are not untouched holdouts.
- ColonDB and CVC300 share 52 exact image hashes and are not independent cohorts.
- Evaluation hardware (local RTX 5060 Ti) differs from training hardware (RTX 4090/5090).
- Arms for seeds 667/668 were trained on different GPUs (B/G/D on 4090, combination on 5090).
- Three seeds; image-level tests are conditional on those seeds, not patient/video clustered, and not corrected for multiple comparisons.
