# polyp_pvt_kvasir_v1: Polyp-PVT trained on the 800 Kvasir-SEG training images

Three seeds (666, 667, 668), last-epoch weights, no selection on validation or test data. Rows of our models: accepted results of the same three seeds. OOD = equal-weight mean of the four OOD sets. ETIS is the held-out set.

## DSC (%), mean +/- SD over seeds

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean | ETIS |
|---|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 91.59 +/- 0.33 | 76.20 +/- 1.59 | 86.40 +/- 2.56 | 87.72 +/- 0.84 | 85.12 +/- 3.03 | 83.86 +/- 1.85 | 74.42 +/- 3.26 |
| MedCLIPSeg baseline, single pass | 88.93 +/- 0.12 | 69.72 +/- 0.28 | 81.86 +/- 0.09 | 79.17 +/- 0.12 | 74.39 +/- 0.77 | 76.28 +/- 0.23 | 54.94 +/- 1.18 |
| ZoomCLIPSeg (detail branch + SCZ) | 89.92 +/- 0.35 | 71.53 +/- 0.88 | 83.57 +/- 0.92 | 81.94 +/- 0.93 | 79.58 +/- 0.75 | 79.15 +/- 0.55 | 60.80 +/- 1.11 |

## NSD (%), mean +/- SD over seeds

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean | ETIS |
|---|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 93.31 +/- 0.29 | 79.13 +/- 1.64 | 89.79 +/- 2.73 | 91.08 +/- 0.90 | 86.65 +/- 3.13 | 86.66 +/- 1.94 | 76.43 +/- 3.23 |
| MedCLIPSeg baseline, single pass | 91.04 +/- 0.07 | 72.76 +/- 0.31 | 85.84 +/- 0.17 | 82.75 +/- 0.13 | 75.94 +/- 0.81 | 79.33 +/- 0.26 | 56.59 +/- 1.13 |
| ZoomCLIPSeg (detail branch + SCZ) | 91.84 +/- 0.32 | 74.55 +/- 0.89 | 87.25 +/- 0.91 | 85.36 +/- 1.02 | 81.00 +/- 0.77 | 82.04 +/- 0.54 | 62.48 +/- 1.13 |

## Polyp-PVT minus our models, DSC in percentage points [95% CI], p

| against | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | ETIS | OOD mean | OOD mean per seed |
|---|---|---|---|---|---|---|---|---|
| MedCLIPSeg baseline, single pass | +2.66 [+1.21, +4.17], p=0.0002 | +6.48 [+4.76, +8.24], p=0.0000 | +4.54 [+2.15, +6.96], p=0.0004 | +8.55 [+6.06, +11.37], p=0.0000 | +10.73 [+7.10, +14.56], p=0.0000 | +19.48 [+16.49, +22.50], p=0.0000 | +7.58 [+6.25, +8.94], p=0.0000 | +5.65, +8.47, +8.61 |
| ZoomCLIPSeg (detail branch + SCZ) | +1.67 [+0.02, +3.26], p=0.0427 | +4.68 [+2.98, +6.42], p=0.0000 | +2.83 [+0.52, +5.33], p=0.0231 | +5.78 [+2.96, +9.20], p=0.0000 | +5.54 [+1.99, +9.22], p=0.0028 | +13.62 [+10.48, +16.89], p=0.0000 | +4.71 [+3.33, +6.13], p=0.0000 | +2.53, +6.11, +5.48 |

## Small lesions (area fraction at most 0.0590), DSC (%)

| model | OOD pooled (n=392) | ETIS (n=151) |
|---|---|---|
| Polyp-PVT (this run) | 77.30 | 69.62 |
| MedCLIPSeg baseline, single pass | 69.45 | 46.02 |
| ZoomCLIPSeg (detail branch + SCZ) | 73.31 | 54.14 |

- Polyp-PVT minus MedCLIPSeg baseline, single pass: OOD pooled +7.85 [+6.15, +9.56], p=0.0000; ETIS +23.60 [+20.12, +27.26], p=0.0000
- Polyp-PVT minus ZoomCLIPSeg (detail branch + SCZ): OOD pooled +3.99 [+2.40, +5.63], p=0.0000; ETIS +15.49 [+11.70, +19.45], p=0.0000

## Images without any predicted lesion / with DSC below 10% (lesion images, per seed)

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | ETIS |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 0/0/0 ; 0/1/0 | 7/2/2 ; 29/34/27 | 0/0/0 ; 0/0/0 | 0/0/0 ; 0/0/0 | 0/0/0 ; 3/1/2 | 6/4/2 ; 20/19/16 |
| MedCLIPSeg baseline, single pass | 0/0/0 ; 0/0/0 | 0/4/2 ; 33/34/33 | 0/0/0 ; 0/0/0 | 0/0/0 ; 1/0/0 | 1/1/0 ; 6/4/4 | 3/1/0 ; 41/45/42 |
| ZoomCLIPSeg (detail branch + SCZ) | 0/0/0 ; 0/0/1 | 1/1/0 ; 32/35/29 | 0/0/0 ; 0/1/0 | 0/0/0 ; 2/2/0 | 0/1/0 ; 4/5/5 | 1/6/0 ; 42/44/40 |

## Cost of Polyp-PVT

- Parameters: 25.11 M, all trainable (25.11 M).
- Peak GPU memory in training (batch 16, largest size 448): 11.7 GB.
- Training time per seed: 1.00 h, 1.00 h, 1.02 h on NVIDIA GeForce RTX 5060 Ti.
- Final training loss per seed: 0.2070, 0.0819, 0.0815.
- Inference: 11.9 ms per image (batch size 1, one deterministic pass, forward plus upsampling to the original size; Kvasir, after 10 warm-up images).

## Differences from the released training and test scripts

- Training data are the 800 Kvasir-SEG images of our split, not the merged Kvasir-SEG and CVC-ClinicDB set of 1450 images.
- The weights of the last epoch are evaluated. The released script keeps the epoch with the best test Dice; we do not select on test data.
- The sigmoid output is thresholded at 0.5 without the per-image min-max normalisation of the released test script, and DSC is computed by our metric code at the original resolution.
- The mini-batch order is drawn from a seeded generator; images are decoded once and kept in memory.
- Training masks are binarised at 127 before resizing (the definition of the ground truth in our metric code); for binary mask files this equals the released loader.
- Files are decoded with OpenCV, as in all our evaluations, and then resized with PIL as in the released loader.
- The validation split is not used.

## Disclosures

- Polyp-PVT uses a 352 x 352 input and an ImageNet-pretrained backbone that is trained end to end (all weights trainable); our models use a 224 x 224 input and frozen encoders.
- The five polyp test sets were used in earlier stages of this project; ETIS-Larib was evaluated once for our method and is new for this model.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
