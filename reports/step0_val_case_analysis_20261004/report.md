# Step 0: image-level analysis of the accepted Val predictions

Source Kvasir Val (100 images, 25 small targets), four arms x three seeds, epoch 100, generic prompt.
Recomputed means match the three acceptance audits. This is the repeatedly used development set, not confirmation data.

## Kvasir Val

Three-seed mean ± sample SD.

| Arm | DSC % | Small DSC % | NSD % | Precision % | Recall % | DSC<50% (per seed) | Zero overlap (per seed) |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 88.37 ± 0.31 | 85.08 ± 0.11 | 90.33 ± 0.27 | 89.94 ± 0.44 | 87.05 ± 0.75 | 3/3/2 | 0/1/0 |
| global_mix | 88.34 ± 0.38 | 85.25 ± 0.03 | 90.31 ± 0.35 | 90.05 ± 0.36 | 86.73 ± 1.32 | 3/3/2 | 0/1/0 |
| detail | 88.86 ± 0.50 | 86.31 ± 0.90 | 90.74 ± 0.47 | 91.02 ± 0.10 | 86.73 ± 0.49 | 3/3/2 | 1/1/0 |
| detail_global_mix | 88.76 ± 0.46 | 86.58 ± 0.11 | 90.63 ± 0.45 | 90.38 ± 0.54 | 86.73 ± 1.31 | 4/3/3 | 0/1/0 |

**Paired DSC differences, all images** (seed-averaged per image; tests are image-level and conditional on the three seeds).

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +0.49 | [+0.08, +0.88] | +0.73 / +0.49 / +0.26 | 3/3 | 73/27/0 | +0.60 | 0.017 | <0.001 | +0.10 | +0.40 |
| global_mix - baseline | 100 | -0.03 | [-0.36, +0.28] | +0.00 / +0.03 / -0.12 | 2/3 | 55/45/0 | +0.11 | 0.876 | 0.364 | +0.08 | -0.11 |
| detail_global_mix - baseline | 100 | +0.40 | [-0.21, +0.95] | +0.59 / +0.40 / +0.21 | 3/3 | 70/30/0 | +0.37 | 0.185 | <0.001 | +0.14 | +0.26 |
| detail_global_mix - detail | 100 | -0.09 | [-0.63, +0.42] | -0.14 / -0.09 / -0.05 | 0/3 | 52/48/0 | +0.03 | 0.736 | 0.824 | +0.20 | -0.30 |

**Paired DSC differences, small targets only.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 25 | +1.23 | [+0.70, +1.78] | +1.61 / +1.80 / +0.29 | 3/3 | 20/5/0 | +1.04 | <0.001 | <0.001 | +0.29 | +1.03 |
| global_mix - baseline | 25 | +0.17 | [-0.18, +0.53] | +0.22 / +0.02 / +0.28 | 3/3 | 15/10/0 | +0.31 | 0.373 | 0.236 | +0.15 | +0.03 |
| detail_global_mix - baseline | 25 | +1.49 | [+0.95, +2.08] | +1.61 / +1.43 / +1.44 | 3/3 | 22/3/0 | +1.48 | <0.001 | <0.001 | +0.37 | +1.22 |
| detail_global_mix - detail | 25 | +0.26 | [-0.21, +0.93] | +0.00 / -0.37 / +1.16 | 2/3 | 14/11/0 | +0.05 | 0.549 | 0.706 | +0.32 | -0.06 |

**Paired NSD differences, all images.**

| Contrast | n | Mean pp | 95% bootstrap CI | Per seed pp | Seeds >0 | Better/worse/tied | Median pp | Permutation p | Wilcoxon p | Top-2 gains add | Others without top-2 gains |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| detail - baseline | 100 | +0.40 | [+0.01, +0.78] | +0.65 / +0.41 / +0.16 | 3/3 | 73/27/0 | +0.49 | 0.044 | <0.001 | +0.10 | +0.31 |
| global_mix - baseline | 100 | -0.02 | [-0.35, +0.28] | -0.00 / +0.05 / -0.11 | 1/3 | 54/45/1 | +0.09 | 0.900 | 0.347 | +0.09 | -0.11 |
| detail_global_mix - baseline | 100 | +0.30 | [-0.29, +0.83] | +0.47 / +0.34 / +0.08 | 3/3 | 69/31/0 | +0.33 | 0.302 | <0.001 | +0.14 | +0.16 |
| detail_global_mix - detail | 100 | -0.10 | [-0.64, +0.41] | -0.17 / -0.07 / -0.07 | 0/3 | 49/51/0 | -0.01 | 0.706 | 0.955 | +0.20 | -0.31 |

**Pixel-level and failure changes.**

| Contrast | Precision pp (per seed) | Recall pp (per seed) | FP pixels % (per seed) | FN pixels % (per seed) | DSC<50% count change |
|---|---|---|---|---|---|
| detail - baseline | +0.99 / +1.59 / +0.66 | -0.12 / -0.73 / -0.13 | -11.0 / -17.3 / -7.7 | +0.9 / +6.0 / +0.9 | +0 / +0 / +0 |
| global_mix - baseline | -0.15 / +0.42 / +0.06 | +0.08 / -0.01 / -1.05 | +1.7 / -4.4 / -2.0 | -0.6 / +0.1 / +7.7 | +0 / +0 / +0 |
| detail_global_mix - baseline | -0.12 / +1.34 / +0.09 | +1.04 / -1.23 / -0.79 | +2.5 / -15.2 / -2.0 | -8.1 / +10.1 / +5.8 | +1 / +0 / +1 |
| detail_global_mix - detail | -1.11 / -0.25 / -0.57 | +1.16 / -0.50 / -0.67 | +15.1 / +2.5 / +6.2 | -8.9 / +3.9 / +4.8 | +1 / +0 / +1 |
