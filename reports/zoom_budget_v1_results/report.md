# zoom_budget_v1: accuracy against inference budget (detail-branch model, five seeds)

Nothing was trained. The locked method (t15 with 30 passes) is unchanged whatever these numbers show.

OOD = equal-weight mean of the four OOD polyp sets. Forward passes = mean number of stochastic forward passes per OOD image; the standard single pass uses 30.

| configuration | forward passes | second pass | Kvasir DSC | OOD DSC | minus the standard single pass [95% CI], p |
|---|---|---|---|---|---|
| single pass, 5 passes | 5.0 | 0% | 89.54 | 77.35 | -0.00 [-0.02, +0.01], p=0.5213 |
| single pass, 10 passes | 10.0 | 0% | 89.54 | 77.35 | -0.00 [-0.01, +0.01], p=0.7316 |
| single pass, 15 passes | 15.0 | 0% | 89.54 | 77.35 | -0.00 [-0.01, +0.00], p=0.3817 |
| SCZ, 10 passes | 17.5 | 75% | 89.93 | 79.41 | +2.05 [+1.25, +2.88], p=0.0000 |
| reflection averaging, 10 passes | 20.0 | 0% | 90.31 | 78.74 | +1.39 [+0.96, +1.86], p=0.0000 |
| SCZ, 15 passes | 26.3 | 75% | 89.95 | 79.51 | +2.16 [+1.34, +2.95], p=0.0000 |
| single pass, 30 passes | 30.0 | 0% | 89.54 | 77.36 |  |
| reflection averaging, 15 passes | 30.0 | 0% | 90.31 | 78.75 | +1.39 [+0.96, +1.85], p=0.0000 |
| SCZ + reflection averaging, 10 passes | 35.5 | 77% | 90.46 | 80.45 | +3.10 [+2.15, +4.04], p=0.0000 |
| SCZ (locked method) | 52.6 | 75% | 89.92 | 79.48 | +2.12 [+1.31, +2.91], p=0.0000 |
| SCZ + reflection averaging, 15 passes | 53.2 | 77% | 90.52 | 80.43 | +3.07 [+2.16, +4.00], p=0.0000 |
| reflection averaging, 30 passes | 60.0 | 0% | 90.31 | 78.75 | +1.39 [+0.96, +1.86], p=0.0000 |
| SCZ + reflection averaging, 30 passes | 106.3 | 77% | 90.53 | 80.42 | +3.07 [+2.14, +3.99], p=0.0000 |

## Contrasts fixed in advance (OOD mean, pp)

| contrast | difference [95% CI], p | per seed |
|---|---|---|
| scz_adds_to_reflection: sczflip30 minus hflip30 | +1.67 [+0.86, +2.50], p=0.0000 | +2.07, +2.09, +1.81, +1.35, +1.04 |
| reflection_adds_to_scz: sczflip30 minus t15 | +0.95 [+0.44, +1.46], p=0.0003 | +1.12, +1.76, +0.88, +0.15, +0.81 |
| scz_at_standard_budget: scz15 minus single30 | +2.16 [+1.34, +2.95], p=0.0000 | +2.43, +2.22, +2.20, +2.50, +1.43 |
| hflip_at_standard_budget: hflip15 minus single30 | +1.39 [+0.95, +1.87], p=0.0000 | +1.42, +1.69, +1.32, +1.38, +1.15 |
| sczflip_near_standard_budget: sczflip10 minus single30 | +3.10 [+2.18, +4.04], p=0.0000 | +3.61, +3.81, +3.14, +2.68, +2.25 |
| scz_half_passes: scz15 minus t15 | +0.04 [-0.06, +0.12], p=0.4827 | +0.06, +0.20, -0.05, -0.07, +0.04 |
| passes_05: single05 minus single30 | -0.00 [-0.02, +0.01], p=0.5179 | -0.01, -0.00, +0.00, -0.02, +0.01 |
| passes_10: single10 minus single30 | -0.00 [-0.01, +0.01], p=0.7329 | -0.00, -0.00, +0.00, -0.01, -0.00 |
| passes_15: single15 minus single30 | -0.00 [-0.01, +0.00], p=0.3829 | -0.01, -0.00, +0.01, -0.00, -0.01 |

## Verdicts by the rules fixed in advance

- SCZ adds to reflection averaging: **True**; reflection averaging adds to SCZ: **True**; complementary (both): **True**
- SCZ with 15 passes per run is better than the standard single pass with 30 (at a lower number of forward passes): **True**
- Smallest number of passes within 0.2 pp of 30 passes (single pass, OOD mean): **5**

Per test set, DSC (%):

| configuration | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single pass, 5 passes | 89.54 | 70.71 | 83.07 | 79.71 | 75.91 |
| single pass, 10 passes | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| single pass, 15 passes | 89.54 | 70.72 | 83.07 | 79.72 | 75.91 |
| SCZ, 10 passes | 89.93 | 71.88 | 84.09 | 81.98 | 79.67 |
| reflection averaging, 10 passes | 90.31 | 71.67 | 83.81 | 81.97 | 77.54 |
| SCZ, 15 passes | 89.95 | 71.99 | 84.11 | 82.13 | 79.81 |
| single pass, 30 passes | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| reflection averaging, 15 passes | 90.31 | 71.67 | 83.81 | 81.97 | 77.54 |
| SCZ + reflection averaging, 10 passes | 90.46 | 72.71 | 84.51 | 83.77 | 80.83 |
| SCZ (locked method) | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |
| SCZ + reflection averaging, 15 passes | 90.52 | 72.72 | 84.48 | 83.85 | 80.67 |
| reflection averaging, 30 passes | 90.31 | 71.66 | 83.81 | 81.97 | 77.55 |
| SCZ + reflection averaging, 30 passes | 90.53 | 72.72 | 84.50 | 83.80 | 80.66 |

## Definitions

- **pass_prefix**: the first pass is run once with 30 stochastic forward passes exactly as in the accepted evaluation; the estimate with k passes is the mean of the first k of these 30 independent passes
- **single**: mean probability map of k passes of the whole image (k = 5, 10, 15, 30; k = 30 is the accepted single pass)
- **hflip**: mean of single with k passes and of the reflected image with k passes, reflected back (k = 10, 15, 30)
- **scz**: the locked two-pass rule applied to the single map with k passes; the second pass uses k newly drawn passes (k = 10, 15; k = 30 is the accepted configuration t15 and is read from the accepted results)
- **sczflip**: the locked two-pass rule applied to the hflip map with k passes; the second pass averages k passes of the window and k passes of the reflected window (k = 10, 15, 30)
- **cost**: mean number of stochastic forward passes per image on the OOD sets: k for single, 2k for hflip, k(1 + s) for scz and 2k(1 + s) for sczflip, with s the share of images that receive a second pass. Every model call makes one additional deterministic forward pass that is not counted.

Consequence fixed in advance: The manuscript reports the outcome of each rule as it is. If complementary fails, it states that SCZ and reflection averaging do not add up. No configuration of this suite replaces the locked method; a cheaper configuration may be described as an operating point that was identified after the fact on test data that had been used before.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated.
- ETIS-Larib is not touched.
- Estimates with fewer passes are prefixes of the 30 passes of one run and are therefore correlated with the 30-pass estimate.
