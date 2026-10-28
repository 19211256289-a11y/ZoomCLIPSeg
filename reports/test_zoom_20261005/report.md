# Two-pass zoom: one-time Test confirmation (detail arm, locked config t15)

Evaluation only. Three-seed mean. Contrast: two-pass minus single-pass on the same checkpoints.

**Verdict: pass** | macro OOD DSC gain +2.21 pp (per seed +2.37, +2.02, +2.25)

## DSC and NSD

| set | n | single DSC | two-pass DSC | gain (pp) | per seed | 95% CI | perm. p | NSD gain | better / worse images |
|---|---|---|---|---|---|---|---|---|---|
| Kvasir | 100 | 89.47 | 89.92 | +0.46 | +0.31, +0.95, +0.11 | +0.02 to +0.97 | 0.0569 | +0.38 | 27 / 18 |
| ColonDB | 360 | 70.26 | 71.53 | +1.26 | +1.06, +0.84, +1.89 | +0.42 to +2.15 | 0.0041 | +1.31 | 166 / 97 |
| ClinicDB | 61 | 82.70 | 83.57 | +0.87 | +2.35, +0.16, +0.08 | -0.70 to +2.42 | 0.2955 | +0.60 | 29 / 20 |
| CVC300 | 60 | 78.89 | 81.94 | +3.05 | +3.08, +3.03, +3.06 | +0.44 to +5.36 | 0.0143 | +3.04 | 47 / 11 |
| BKAI | 100 | 75.91 | 79.58 | +3.67 | +2.99, +4.06, +3.97 | +1.87 to +5.87 | 0.0001 | +3.58 | 55 / 14 |

## Small targets, failures and cost

| set | small n | small DSC single | small DSC two-pass | failures single -> two-pass | share of images with a second pass | second pass empty |
|---|---|---|---|---|---|---|
| Kvasir | 22 | 89.07 | 90.93 | 2.3 -> 2.0 | 41% | 0.0 |
| ColonDB | 241 | 67.86 | 70.03 | 76.7 -> 71.3 | 68% | 0.0 |
| ClinicDB | 24 | 82.02 | 82.41 | 5.3 -> 5.7 | 75% | 0.0 |
| CVC300 | 56 | 77.98 | 81.44 | 7.0 -> 7.3 | 92% | 0.0 |
| BKAI | 71 | 69.93 | 74.94 | 17.0 -> 15.3 | 63% | 0.0 |

## OOD by lesion size (four OOD sets pooled, means over seeds)

| lesion share of image | n | single DSC | two-pass DSC | zoomed | failures single -> two-pass |
|---|---|---|---|---|---|
| <1% | 55 | 25.54 | 33.34 | 69% | 41.0 -> 35.7 |
| 1-3% | 179 | 73.14 | 76.47 | 85% | 31.7 -> 27.7 |
| 3-5.9% | 158 | 83.28 | 83.63 | 90% | 11.3 -> 13.0 |
| 5.9-15% | 129 | 85.75 | 85.69 | 55% | 4.3 -> 4.7 |
| >15% | 60 | 65.77 | 65.52 | 13% | 17.7 -> 18.7 |

Unique-image sensitivity (equal weight per distinct image): ColonDB +1.49 pp over 329 images; CVC300 +3.01 pp over 58 images

## Gate (fixed before any two-pass Test number was computed)

- g1 macro OOD gain +2.21 pp (need >= +1.0): True
- g2 OOD sets with a positive gain 4/4 (need >= 3): True
- g3 Kvasir Test change +0.46 pp (need >= -0.2): True
- g4 largest regression on any set +0.46 pp (need >= -0.5): True
- g5 every seed positive on the macro OOD gain: True

Single-pass results reproduce the accepted Test evaluation within 0.0018 pp.
Disclosures: the weakness was found on these Test predictions; the configuration was selected on rescaled source Val
images only; the test sets were used before; ColonDB and CVC300 share 52 images; the second pass costs one extra
MC30 inference per zoomed image.
