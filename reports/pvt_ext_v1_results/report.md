# pvt_ext_v1: Polyp-PVT architecture as a text-free reference (ultrasound; reduced training sets)

Three seeds (666, 667, 668), released recipe unchanged, last-epoch weights, no selection on validation or test data. OOD = equal-weight mean of the four OOD sets.

## busi_full: trained on BUSI / Train_text.xlsx (624 images)

DSC (%), mean +/- SD over seeds:

| model | BUSI | BUSBRA | BUSUC | BUID | UDIAT | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 83.21 +/- 1.45 | 68.32 +/- 3.46 | 77.74 +/- 1.12 | 73.27 +/- 2.55 | 71.75 +/- 2.73 | 72.77 +/- 2.24 |
| MedCLIPSeg baseline, single pass | 79.61 +/- 2.62 | 57.71 +/- 10.13 | 76.56 +/- 5.73 | 66.34 +/- 6.23 | 56.78 +/- 8.11 | 64.35 +/- 7.40 |
| Detail branch, single pass | 81.04 +/- 1.17 | 68.94 +/- 0.57 | 83.58 +/- 0.84 | 74.63 +/- 0.87 | 72.95 +/- 0.76 | 75.03 +/- 0.23 |
| ZoomCLIPSeg (detail branch + SCZ) | 81.24 +/- 1.47 | 70.01 +/- 0.75 | 83.73 +/- 0.93 | 75.24 +/- 1.24 | 75.36 +/- 1.77 | 76.09 +/- 0.65 |

NSD (%), mean +/- SD over seeds:

| model | BUSI | BUSBRA | BUSUC | BUID | UDIAT | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 85.62 +/- 1.53 | 72.02 +/- 3.65 | 81.65 +/- 1.21 | 75.20 +/- 2.66 | 75.35 +/- 2.83 | 76.05 +/- 2.33 |
| MedCLIPSeg baseline, single pass | 82.19 +/- 2.68 | 61.16 +/- 10.39 | 80.96 +/- 5.69 | 68.23 +/- 6.26 | 60.00 +/- 8.10 | 67.59 +/- 7.44 |
| Detail branch, single pass | 83.65 +/- 1.18 | 72.88 +/- 0.60 | 88.07 +/- 0.84 | 76.65 +/- 0.91 | 76.72 +/- 0.76 | 78.58 +/- 0.26 |
| ZoomCLIPSeg (detail branch + SCZ) | 83.84 +/- 1.46 | 73.87 +/- 0.79 | 88.21 +/- 0.91 | 77.30 +/- 1.32 | 78.97 +/- 1.95 | 79.59 +/- 0.72 |

Polyp-PVT minus our models, DSC in percentage points [95% CI], p:

| against | BUSI | BUSBRA | BUSUC | BUID | UDIAT | OOD mean | OOD mean per seed |
|---|---|---|---|---|---|---|---|
| MedCLIPSeg baseline, single pass | +3.60 [-0.67, +8.12], p=0.1172 | +10.62 [+7.89, +13.36], p=0.0000 | +1.18 [-1.52, +3.77], p=0.3859 | +6.93 [+3.45, +10.90], p=0.0002 | +14.97 [+6.18, +24.66], p=0.0037 | +8.42 [+5.83, +11.13], p=0.0000 | -0.23, +17.64, +7.87 |
| Detail branch, single pass | +2.17 [-2.32, +7.42], p=0.4080 | -0.62 [-3.17, +1.93], p=0.6382 | -5.84 [-9.11, -2.95], p=0.0001 | -1.36 [-5.91, +2.08], p=0.6024 | -1.20 [-9.78, +6.88], p=0.7898 | -2.25 [-4.81, +0.20], p=0.0848 | -4.62, -1.23, -0.92 |
| ZoomCLIPSeg (detail branch + SCZ) | +1.97 [-2.68, +7.09], p=0.4565 | -1.68 [-4.14, +0.81], p=0.1861 | -6.00 [-9.49, -2.97], p=0.0001 | -1.98 [-6.41, +1.49], p=0.4138 | -3.61 [-11.73, +3.21], p=0.3807 | -3.32 [-5.80, -1.09], p=0.0046 | -5.99, -1.64, -2.32 |

Detection (per seed):

| model | missed lesions on the four OOD sets | lesion-free source test images without a prediction |
|---|---|---|
| Polyp-PVT (this run) | 74/29/49 | 12/12/13 of 13 |
| MedCLIPSeg baseline, single pass | 41/86/55 | 12/12/11 of 13 |
| Detail branch, single pass | 26/23/12 | 10/9/10 of 13 |
| ZoomCLIPSeg (detail branch + SCZ) | 26/23/12 | 10/9/10 of 13 |

Training time per seed: 0.78 h, 0.77 h, 0.78 h. Final training loss: 0.1256, 0.1104, 0.1347 (range over the last ten epochs: 0.125-0.141, 0.110-0.131, 0.123-0.327).

## kvasir_p10: trained on Kvasir / Train_text_10.xlsx (80 images)

DSC (%), mean +/- SD over seeds:

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 87.70 +/- 0.78 | 66.80 +/- 2.94 | 81.59 +/- 1.70 | 83.61 +/- 1.05 | 75.82 +/- 1.81 | 76.95 +/- 1.32 |

NSD (%), mean +/- SD over seeds:

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 89.55 +/- 0.76 | 69.52 +/- 2.98 | 85.21 +/- 1.76 | 87.10 +/- 1.06 | 77.28 +/- 1.92 | 79.78 +/- 1.36 |

Detection (per seed):

| model | missed lesions on the four OOD sets | lesion-free source test images without a prediction |
|---|---|---|
| Polyp-PVT (this run) | 4/19/63 | none in this set |

Training time per seed: 0.10 h, 0.10 h, 0.10 h. Final training loss: 0.3454, 0.3258, 0.3523 (range over the last ten epochs: 0.345-0.439, 0.313-0.467, 0.328-0.398).

## kvasir_p25: trained on Kvasir / Train_text_25.xlsx (200 images)

DSC (%), mean +/- SD over seeds:

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 90.66 +/- 0.30 | 71.11 +/- 0.34 | 84.24 +/- 1.00 | 86.87 +/- 0.67 | 81.76 +/- 2.97 | 80.99 +/- 0.55 |

NSD (%), mean +/- SD over seeds:

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean |
|---|---|---|---|---|---|---|
| Polyp-PVT (this run) | 92.41 +/- 0.34 | 73.87 +/- 0.54 | 87.73 +/- 1.22 | 90.26 +/- 0.74 | 83.30 +/- 3.04 | 83.79 +/- 0.64 |

Detection (per seed):

| model | missed lesions on the four OOD sets | lesion-free source test images without a prediction |
|---|---|---|
| Polyp-PVT (this run) | 5/16/11 | none in this set |

Training time per seed: 0.26 h, 0.26 h, 0.25 h. Final training loss: 0.1406, 0.1147, 0.1209 (range over the last ten epochs: 0.141-0.465, 0.112-0.129, 0.121-0.135).

## Notes

- The number of epochs of the released recipe (99) is kept for every group, as the number of epochs of our models is kept (100). With a reduced training set both families therefore make proportionally fewer optimisation steps.
- Conditional follow-up fixed in advance: If, on a reduced training set, this network is not ahead of our full method on the mean of the four OOD sets, it is trained again on that subset with the number of optimisation steps of the full-data run (epochs multiplied by the inverse of the fraction) in a separate suite, and both results are reported. This guards against an advantage of our models that would only reflect an under-trained reference.

## Disclosures

- The network was designed for polyp images; its recipe is applied to ultrasound without any change or tuning.
- It uses a 352 x 352 input and an ImageNet-pretrained backbone trained end to end; our models use a 224 x 224 input and frozen encoders.
- The ultrasound and polyp test sets were used in earlier stages of this project.
- BUID (35 images) and UDIAT (25 images) are small.
