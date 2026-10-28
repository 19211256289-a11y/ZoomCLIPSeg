# pvt_zoom_v1: the locked SCZ applied, unchanged, to Polyp-PVT (three seeds)

Nothing was trained. DSC in %, mean over seeds and images; OOD = equal-weight mean of the four OOD sets.

| model | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean | ETIS |
|---|---|---|---|---|---|---|---|
| Polyp-PVT, single pass | 91.59 | 76.20 | 86.40 | 87.72 | 85.12 | 83.86 | 74.42 |
| Polyp-PVT + SCZ | 91.69 | 75.10 | 86.09 | 86.45 | 82.43 | 82.52 | 73.31 |
| MedCLIPSeg baseline, single pass | 88.93 | 69.72 | 81.86 | 79.17 | 74.39 | 76.28 | 54.94 |
| ZoomCLIPSeg (detail branch + SCZ) | 89.92 | 71.53 | 83.57 | 81.94 | 79.58 | 79.15 | 60.80 |

## Gain of SCZ on Polyp-PVT (two-pass minus single-pass, pp) [95% CI], p

| | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | ETIS | OOD mean |
|---|---|---|---|---|---|---|---|
| gain | +0.09 [-0.09, +0.29], p=0.3439 | -1.10 [-1.63, -0.60], p=0.0001 | -0.31 [-2.13, +1.39], p=0.7554 | -1.27 [-3.04, +0.07], p=0.1320 | -2.69 [-4.36, -1.15], p=0.0008 | -1.11 [-2.60, +0.37], p=0.1419 | -1.34 [-2.09, -0.63], p=0.0001 |
| second pass | 40% | 71% | 71% | 92% | 70% | 74% | |

OOD gain per seed: -1.82, -0.94, -1.28

**Verdict by the rule fixed in advance: scz_harms.**

Consequence fixed in advance: scz_transfers: SCZ is reported as a test-time procedure that also helps a dedicated network. no_effect or scz_harms: the manuscript states that the scale dependence addressed by SCZ is specific to the frozen 224 x 224 model and that a network with multi-scale training does not benefit.

## DSC by lesion area as a share of the image

OOD pooled:

| lesion area | n | Polyp-PVT, single pass | Polyp-PVT + SCZ | MedCLIPSeg baseline, single pass | ZoomCLIPSeg (detail branch + SCZ) |
|---|---|---|---|---|---|
| <1% | 55 | 43.09 | 38.91 | 22.89 | 33.34 |
| 1-3% | 179 | 80.19 | 79.01 | 72.18 | 76.47 |
| 3-5.9% | 158 | 85.93 | 84.71 | 82.56 | 83.63 |
| 5.9-15% | 129 | 88.73 | 87.52 | 85.00 | 85.69 |
| >=15% | 60 | 78.86 | 79.32 | 68.21 | 65.52 |

ETIS:

| lesion area | n | Polyp-PVT, single pass | Polyp-PVT + SCZ | MedCLIPSeg baseline, single pass | ZoomCLIPSeg (detail branch + SCZ) |
|---|---|---|---|---|---|
| <1% | 62 | 55.68 | 58.08 | 34.13 | 45.10 |
| 1-3% | 54 | 80.16 | 77.58 | 49.59 | 61.56 |
| 3-5.9% | 35 | 78.06 | 72.46 | 61.57 | 58.69 |
| 5.9-15% | 27 | 87.31 | 86.18 | 78.42 | 75.30 |
| >=15% | 18 | 95.28 | 95.28 | 94.54 | 94.95 |

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- ETIS-Larib is evaluated here once for the combination of Polyp-PVT with SCZ; the evaluation of our own method on it stays closed.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- No constant of SCZ is adapted to Polyp-PVT; in particular the target fraction is not tuned.
- This protocol and its code were written after the single-pass results of seed 666 of polyp_pvt_kvasir_v1 had been inspected (a preview at the request of the first author) and before any two-pass result of any Polyp-PVT model existed.
