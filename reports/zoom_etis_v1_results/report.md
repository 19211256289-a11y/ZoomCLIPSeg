# zoom_etis_v1: one-time evaluation on ETIS-Larib

One-time evaluation on a polyp dataset that was never used in this project (ETIS-LaribPolypDB, 196 images). The accepted Kvasir-trained checkpoints (baseline and detail branch, five seeds) are evaluated with single-pass inference and with the locked two-pass zoom configuration. Nothing is trained, selected or tuned.

196 images, every image with a polyp; median lesion area 1.98% of the image. Seeds 666, 667, 668, 669, 670. Fixed sentence: "a polyp in a colonoscopy image".

**Verdict: pass** (detail arm, two-pass minus single-pass): +2.84 pp (per seed +3.16, +1.87, +3.11, +4.34, +1.72)

## DSC (%)

| arm | inference | mean | SD over seeds | per seed | NSD | small-lesion DSC | failures per seed |
|---|---|---|---|---|---|---|---|
| baseline | single pass | 55.10 | 1.15 | 56.02, 53.68, 55.12, 54.28, 56.43 | 56.83 | 46.29 | 76, 81, 76, 78, 80 |
| baseline | two-pass | 59.50 | 0.97 | 60.34, 58.95, 58.98, 58.52, 60.72 | 61.21 | 51.68 | 73, 76, 73, 70, 70 |
| detail | single pass | 57.46 | 1.10 | 58.60, 57.71, 57.94, 55.67, 57.38 | 59.11 | 49.45 | 67, 73, 70, 78, 76 |
| detail | two-pass | 60.30 | 1.09 | 61.77, 59.59, 61.05, 60.01, 59.10 | 61.98 | 53.09 | 64, 72, 67, 71, 75 |

## Paired differences (pp)

| contrast | mean | per seed | seeds > 0 | 95% CI over images | permutation p | images better / worse |
|---|---|---|---|---|---|---|
| two_pass_on_detail | +2.84 | +3.16, +1.87, +3.11, +4.34, +1.72 | 5/5 | +0.82 to +4.82 | 0.0052 | 102 / 42 |
| two_pass_on_baseline | +4.40 | +4.32, +5.26, +3.86, +4.24, +4.29 | 5/5 | +2.56 to +6.38 | 0.0000 | 98 / 37 |
| detail_minus_baseline_single | +2.36 | +2.58, +4.03, +2.82, +1.39, +0.95 | 5/5 | +1.19 to +3.59 | 0.0001 | 120 / 62 |
| detail_minus_baseline_two_pass | +0.80 | +1.42, +0.64, +2.07, +1.49, -1.62 | 4/5 | -0.49 to +2.09 | 0.2326 | 109 / 75 |
| full_method_minus_baseline_single | +5.20 | +5.75, +5.91, +5.93, +5.74, +2.67 | 5/5 | +3.04 to +7.47 | 0.0000 | 134 / 49 |

## By lesion area (share of the image)

| area | n | baseline single | baseline two-pass | detail single | detail two-pass | second pass (detail) |
|---|---|---|---|---|---|---|
| <1% | 62 | 34.74 | 43.56 | 36.07 | 43.93 | 72% |
| 1-3% | 54 | 50.64 | 57.84 | 56.45 | 60.75 | 85% |
| 3-5.9% | 35 | 60.05 | 56.54 | 62.36 | 57.50 | 70% |
| 5.9-15% | 27 | 78.00 | 79.81 | 77.26 | 77.53 | 61% |
| >15% | 18 | 94.67 | 94.67 | 94.96 | 94.96 | 0% |

## Gate (fixed in advance)

- g1 two-pass gain of the detail arm +2.84 pp (need >= +1.0): True
- g2 positive for every seed (5/5): True

Second pass applied to 67% of the images (detail arm), 61% (baseline).
Exact duplicates of images used before: none found.
Disclosures: ETIS-Larib was not used for training, model selection or any analysis in this project before this evaluation. Patient-level independence from the training data and absence from the pre-training data of the frozen encoders cannot be established from the archive. The dataset has one acquisition source; every image contains a polyp.
