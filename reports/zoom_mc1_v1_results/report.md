# zoom_mc1_v1: SCZ with one stochastic pass per run (detail-branch model, five seeds)

Nothing was trained. The locked method (t15) is unchanged whatever these numbers show. ETIS-Larib is not touched.

| configuration | second pass (OOD) | Kvasir DSC | OOD DSC | OOD per seed |
|---|---|---|---|---|
| single pass, 30 stochastic passes (accepted) | 0% | 89.54 | 77.36 | 76.83, 76.56, 77.43, 77.43, 78.53 |
| single pass, 1 stochastic pass | 0% | 89.54 | 77.35 | 76.83, 76.56, 77.44, 77.38, 78.55 |
| SCZ, 1 stochastic pass per run | 75% | 89.98 | 79.53 | 78.88, 78.79, 79.69, 80.11, 80.19 |
| SCZ, 30 stochastic passes per run (locked method) | 75% | 89.92 | 79.48 | 79.20, 78.58, 79.68, 80.00, 79.92 |

## Contrasts fixed in advance (pp)

| contrast | Kvasir | OOD mean [95% CI], p | per seed |
|---|---|---|---|
| scz01_vs_t15: scz01 minus t15 | +0.07 [-0.02, +0.16], p=0.1848 | +0.06 [-0.10, +0.21], p=0.4811 | -0.32, +0.21, +0.01, +0.11, +0.27 |
| scz01_vs_single01: scz01 minus single01 | +0.44 [+0.00, +0.94], p=0.0632 | +2.18 [+1.37, +2.99], p=0.0000 | +2.06, +2.24, +2.25, +2.73, +1.64 |
| scz01_vs_single30: scz01 minus single30 | +0.44 [+0.01, +0.91], p=0.0572 | +2.18 [+1.34, +2.95], p=0.0000 | +2.05, +2.23, +2.26, +2.68, +1.66 |
| single01_vs_single30: single01 minus single30 | +0.00 [-0.04, +0.04], p=0.9956 | -0.01 [-0.03, +0.02], p=0.6930 | -0.00, -0.01, +0.01, -0.05, +0.02 |

## Verdicts by the rules fixed in advance

- SCZ with one stochastic pass is non-inferior to the locked method if the lower bound of the 95% interval of scz01 minus t15 (OOD mean) is above -0.5 pp: **True**
- SCZ keeps its gain with one stochastic pass if scz01 minus single01 (OOD mean) is positive for every seed and its 95% interval excludes zero: **True**

## Time per image at batch size 1 (NVIDIA GeForce RTX 5060 Ti, ColonDB, detail_seed666, 150 images after 10 warm-up images)

| setting | mean ms (over repetitions) | range of repetition means | second pass share |
|---|---|---|---|
| single pass, 1 stochastic pass | 61.4 | 54.5-66.3 | 0% |
| SCZ, 1 stochastic pass per run | 104.7 | 95.3-114.9 | 81% |
| single pass, 30 stochastic passes (accepted) | 831.5 | 746.9-885.8 | 0% |
| SCZ, 30 stochastic passes per run (locked method) | 1467.4 | 1397.4-1555.8 | 81% |

Note: One GPU (RTX 5060 Ti), batch size 1, torch.cuda.synchronize before and after each image; images are the first images of the CVC-ColonDB test list. Accuracy is not taken from the timing run.

Per test set, DSC (%):

| configuration | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single pass, 30 stochastic passes (accepted) | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| single pass, 1 stochastic pass | 89.54 | 70.68 | 83.06 | 79.74 | 75.91 |
| SCZ, 1 stochastic pass per run | 89.98 | 71.94 | 84.31 | 82.20 | 79.68 |
| SCZ, 30 stochastic passes per run (locked method) | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |

## Definitions

- **single30**: mean probability map of 30 stochastic passes of the whole image, computed exactly as in the accepted evaluation (reproduction check)
- **single01**: the first of these 30 stochastic passes (one stochastic pass of the whole image)
- **scz01**: the locked two-pass rule applied to single01; the second run on the window uses one newly drawn stochastic pass
- **t15**: the locked method (SCZ, 30 stochastic passes in each run); read from the accepted results
- **time**: wall-clock time per image at batch size 1 on one GPU, from the preprocessed input tensors to the final probability map at image resolution, including the window rule, reading and cropping the original image, the second run and pasting; data loading of the first input is not included. Measured for single01, scz01, single30 and t15 in one session, after warm-up images, in three repetitions.

Consequence fixed in advance: The manuscript reports both verdicts and the times as they are. If both rules are met, it states that SCZ with one stochastic pass per run gives the accuracy of the locked method at the measured time per image; otherwise it states which rule failed. No configuration of this suite replaces the locked method.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated.
- ETIS-Larib is not touched.
- single01 is the first of the 30 passes of the accepted first run and is therefore correlated with single30.
