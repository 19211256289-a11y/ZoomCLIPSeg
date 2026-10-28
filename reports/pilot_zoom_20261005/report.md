# Pilot C: scale-canonical two-pass inference (Kvasir Val, synthetic scale shift)

Evaluation only. Three-seed mean DSC (%) of the detail arm. No Test or OOD image was read.

**Verdict: pass** | selected config: `t15`

## Cohorts

| condition | n | positive | median lesion fraction | lesions <1% |
|---|---|---|---|---|
| s025 | 100 | 100 | 0.77% | 62 |
| s035 | 100 | 100 | 1.53% | 32 |
| s050 | 100 | 100 | 3.12% | 12 |
| s100 | 100 | 100 | 12.43% | 1 |
| s200 | 100 | 100 | 44.17% | 0 |

## DSC by condition and config

| config | s025 | s035 | s050 | s100 | s200 | mean |
|---|---|---|---|---|---|---|
| single | 32.25 | 44.48 | 73.64 | 88.86 | 84.82 | 64.81 |
| t08 | 39.27 | 56.14 | 80.60 | 89.15 | 84.79 | 69.99 |
| t15 | 65.77 | 73.55 | 84.87 | 89.20 | 84.73 | 79.63 |
| t08_out | 39.27 | 56.14 | 80.60 | 87.53 | 84.69 | 69.65 |
| t15_out | 65.77 | 73.55 | 84.87 | 87.57 | 84.62 | 79.28 |
| oracle_t08 | 87.38 | 88.02 | 88.86 | 88.79 | 85.86 | 87.78 |
| oracle_t15 | 89.65 | 90.25 | 90.47 | 89.35 | 86.20 | 89.18 |

## Selected config `t15` versus single-pass

| condition | gain (pp) | zoom-in | zoom-out | fallback | improved >1pp | worsened >1pp | failures single -> two-pass |
|---|---|---|---|---|---|---|---|
| s025 | +33.52 | 100.0 | 0.0 | 0.0 | 91.3 | 6.3 | 79.0 -> 23.3 |
| s035 | +29.07 | 88.7 | 0.0 | 0.0 | 82.0 | 3.7 | 55.3 -> 19.0 |
| s050 | +11.23 | 88.3 | 0.0 | 0.0 | 73.0 | 7.0 | 16.0 -> 6.3 |
| s100 | +0.34 | 36.7 | 0.0 | 0.0 | 13.7 | 8.3 | 2.7 -> 2.7 |
| s200 | -0.09 | 5.3 | 0.0 | 0.0 | 0.7 | 2.3 | 8.3 -> 8.3 |

(counts are means over the three seeds; failures are positive images with DSC < 50%)

## Gate

- p0 proxy valid (single-pass drop at the smallest scale 56.61 pp, need >= 5.0): True
- g1 mean shrink gain +24.61 pp (need >= +2.0): True
- g2 unshifted change +0.34 pp (need >= -0.2): True
- g3 every seed positive on the shrink mean (+26.72, +23.30, +23.80): True

Oracle rows take the window from the ground truth and are an upper bound for perfect localisation, not a method.
Selection and gate use the same development data, so this is not a Test result.
Part of the gain on shrunken images comes from the finer output grid of the zoomed pass (single-pass masks are
produced at 224 and upsampled); the failure counts separate recovered lesions from sharper boundaries.
