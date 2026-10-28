# zoom_probe_v1: evaluation-only review analyses

Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.

## A. Polyps: second-pass variants (detail-branch model)

Gain = two-pass minus single-pass DSC, percentage points; OOD = equal-weight mean of the four OOD sets.

| variant | group | seeds | OOD gain [95% CI], p | per seed | in-domain gain | second pass (OOD) |
|---|---|---|---|---|---|---|
| t15 | reference | 5 | +2.13 [+1.33, +2.91], p=0.0000 | +2.37, +2.05, +2.25, +2.57, +1.41 | +0.38 | 75% |
| in224 | resolution | 5 | +1.73 [+0.95, +2.49], p=0.0001 | +1.68, +1.67, +2.05, +2.01, +1.25 | +0.23 | 75% |
| in224_out224 | resolution | 5 | +1.53 [+0.75, +2.29], p=0.0001 | +1.45, +1.51, +1.84, +1.85, +1.01 | +0.18 | 75% |
| margin100 | sensitivity | 3 | +2.27 [+1.34, +3.20], p=0.0000 | +2.02, +2.11, +2.69 | +0.30 | 79% |
| margin150 | sensitivity | 3 | +2.13 [+1.32, +2.95], p=0.0000 | +2.01, +2.70, +1.68 | +0.39 | 71% |
| minwin010 | sensitivity | 3 | +2.22 [+1.33, +3.09], p=0.0000 | +2.37, +2.05, +2.25 | +0.47 | 75% |
| minwin030 | sensitivity | 3 | +2.21 [+1.31, +3.12], p=0.0000 | +2.38, +2.04, +2.23 | +0.44 | 75% |
| trigger070 | sensitivity | 3 | +2.14 [+1.32, +2.97], p=0.0000 | +2.41, +2.20, +1.80 | +0.47 | 65% |
| trigger090 | sensitivity | 3 | +2.18 [+1.26, +3.11], p=0.0000 | +2.43, +2.05, +2.07 | +0.51 | 83% |
| tau010 | sensitivity | 3 | +2.13 [+1.43, +2.87], p=0.0000 | +2.27, +2.01, +2.11 | +0.50 | 62% |
| tau020 | sensitivity | 3 | +2.11 [+1.22, +3.02], p=0.0000 | +2.60, +1.81, +1.93 | +0.58 | 82% |

Reference gain on the first three seeds only (for comparison with the sensitivity rows): +2.22 pp.

**Scale or original pixels?**

- in224: gain +1.73 pp against +2.13 pp for t15 (ratio 0.81); t15 minus in224: +0.40 [+0.15, +0.66], p=0.0012
- in224_out224: gain +1.53 pp against +2.13 pp for t15 (ratio 0.72); t15 minus in224_out224: +0.60 [+0.34, +0.87], p=0.0000

Verdict by the rule fixed in advance (ratio of in224): **scale_explains_most**.

**Sensitivity:** robust by the rule fixed in advance: **True**.

## B. Breast ultrasound: detail branch at matched image-level specificity

theta = image-score threshold of the detail-branch model; the baseline always uses 0.5.

| seed | gate | theta | Val spec. | Test spec. | Test sens. | OOD empty | OOD macro DSC | OOD macro gain [95% CI], p |
|---|---|---|---|---|---|---|---|---|
| 666 | baseline | 0.500 | 76.9% | 92.3% | 98.5% | 41 | 70.44 | |
| 666 | detail, ungated | 0.500 | 61.5% | 76.9% | 98.5% | 26 | 74.82 | +4.38 [+1.06, +7.99], p=0.0126 |
| 666 | detail, val_gate | 0.994 | 76.9% | 84.6% | 98.5% | 51 | 74.22 | +3.78 [+0.47, +7.45], p=0.0351 |
| 666 | detail, test_gate | 1.000 | 100.0% | 92.3% | 93.8% | 101 | 69.34 | -1.09 [-4.94, +2.56], p=0.5943 |
| 667 | baseline | 0.500 | 84.6% | 92.3% | 98.5% | 86 | 56.12 | |
| 667 | detail, ungated | 0.500 | 69.2% | 69.2% | 100.0% | 23 | 74.98 | +18.87 [+15.41, +22.70], p=0.0000 |
| 667 | detail, val_gate | 0.998 | 84.6% | 84.6% | 100.0% | 52 | 74.39 | +18.27 [+14.84, +21.99], p=0.0000 |
| 667 | detail, test_gate | 0.999 | 84.6% | 92.3% | 98.5% | 59 | 74.13 | +18.02 [+14.58, +21.81], p=0.0000 |
| 668 | baseline | 0.500 | 69.2% | 84.6% | 98.5% | 55 | 66.49 | |
| 668 | detail, ungated | 0.500 | 53.8% | 76.9% | 100.0% | 12 | 75.27 | +8.78 [+5.90, +12.06], p=0.0000 |
| 668 | detail, val_gate | 0.983 | 69.2% | 76.9% | 100.0% | 25 | 75.23 | +8.74 [+5.89, +11.99], p=0.0000 |
| 668 | detail, test_gate | 0.990 | 69.2% | 84.6% | 100.0% | 27 | 75.19 | +8.70 [+5.82, +11.98], p=0.0000 |

Lesion-free images: 13 in the validation split, 13 in the test set.

Seed-averaged OOD macro gain of the detail branch:

- ungated: +10.68 [+8.32, +13.34], p=0.0000; per seed +4.38, +18.87, +8.78
- val_gate: +10.26 [+7.92, +13.00], p=0.0000; per seed +3.78, +18.27, +8.74
- test_gate: +8.54 [+6.37, +10.92], p=0.0000; per seed -1.09, +18.02, +8.70

Image-level AUC on the BUSI test set (lesion against lesion-free): seed 666: baseline 0.937, detail 0.987; seed 667: baseline 0.964, detail 0.978; seed 668: baseline 0.942, detail 0.995.

Verdict by the rule fixed in advance (validation-calibrated gate): **retained**.

## C. Inference time (batch size 1, 30 stochastic passes per run)

| model | device | first pass (ms) | second pass when run (ms) | second pass share | two-pass total (ms) | relative cost |
|---|---|---|---|---|---|---|
| baseline_seed666 | NVIDIA GeForce RTX 5060 Ti | 747 +/- 10 | 752 +/- 7 | 48% | 1108 | 1.48x |
| detail_seed666 | NVIDIA GeForce RTX 5060 Ti | 760 +/- 6 | 767 +/- 9 | 48% | 1128 | 1.48x |

Timing images: 50 Kvasir-SEG test images after 10 warm-up images.

## Disclosures

- The five polyp test sets and the ultrasound test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- ETIS-Larib is not touched; its evaluation was run once and stays closed.
- The validation split of BUSI contains few lesion-free images; the calibrated gate is coarse.
