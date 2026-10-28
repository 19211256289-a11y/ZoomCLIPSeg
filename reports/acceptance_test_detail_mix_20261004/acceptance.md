# Acceptance: Step 0 (Val image-level analysis) and Step 1 (Test/OOD evaluation)

Date: 2026-10-04 (Hong Kong time). Suite `test_detail_mix_kvasir_v1`, evaluation only. Run 17:34:39 to 18:41:45, one uninterrupted process, 74/74 jobs, no error, no pause.

## Integrity: passed

| Check | Result |
|---|---|
| Suite code on disk equals the code prepared before the run (7 files, SHA256) | identical |
| Protocol file SHA256 `a43c0685...69a7` equals the hash stored in the locked plan; plan digest equals the queue's | yes |
| Twelve checkpoints matched the hashes in the three acceptance audits (enforced by the runner before loading) | yes |
| Five test cohorts equal the files sealed for the earlier 45-test study (enforced by the runner) | yes |
| Val reproduction on the local RTX 5060 Ti versus accepted server values | max abs difference 0.0103 pp DSC (limit 0.25) |
| Cohort sizes 100/360/61/60/100, small targets 22/241/24/56/71, 12 runs per image, identical ground truth across runs | yes |
| Per-image DSC recomputed from stored pixel counts | max difference 0 |
| All arm means and paired contrasts recomputed with independent code (pandas/scipy) | agree with report to 3e-14 |

Not re-verified in this acceptance: saved mask/probability files were not re-read, and metrics were not recomputed from masks and ground truth; the recomputation used the stored per-image pixel counts.

## Results (three-seed mean DSC %, generic prompt)

| Dataset | baseline | global_mix | detail | detail_global_mix |
|---|---:|---:|---:|---:|
| Kvasir Test | 88.93 | 88.85 | 89.47 | 89.72 |
| ColonDB | 69.72 | 69.45 | 70.26 | 70.61 |
| ClinicDB | 81.86 | 82.39 | 82.70 | 83.41 |
| CVC300 | 79.17 | 79.07 | 78.89 | 80.19 |
| BKAI | 74.39 | 75.56 | 75.91 | 76.81 |

## Pre-specified gate (detail minus baseline): passed

G1 Kvasir Test +0.53 pp; G2 positive on 3 of 4 OOD sets (ColonDB +0.54, ClinicDB +0.84, BKAI +1.52; CVC300 -0.28); G3 macro small-target gain +1.16 pp; guard worst set -0.28 pp.

## Assessment

- Val to Test consistency: Val +0.49 overall / +1.23 small; Test macro +0.63 / +1.16. The effect did not shrink on data not used for development of this module.
- Step 0: the Val gain is broad (73 of 100 images better, 20 of 25 small; top two gains contribute 0.10 of 0.49 pp).
- Pooled OOD, 496 unique images: +0.67 pp (95% CI +0.09 to +1.24); small targets (317) +1.27 pp, positive in all three seeds.
- Seed level: 13 of 15 seed x dataset cells positive; macro per seed +0.76/+0.31/+0.83, paired t p = 0.06 (three seeds). BKAI's +1.52 is mainly seed 666 (+3.49; others +0.68, +0.39).
- CVC300 is not independent: 52 of its 58 unique images are also in ColonDB. Independent OOD evidence is ColonDB, ClinicDB and BKAI. ColonDB's gain is +0.31 pp when each unique image is weighted equally.
- The hard tail is unchanged: about 74 of 360 ColonDB records stay below 50% DSC in every arm.
- Secondary, not pre-specified as primary: the combination is the best arm on all five sets. Versus baseline it is positive in 15 of 15 cells, macro +1.33 pp (seed-level p = 0.007). Versus detail alone it is +0.70 pp macro, 12 of 15 cells, p = 0.09, while on Val it was -0.09. Treat as a hypothesis to confirm on new data, not as a selected final model.
- Generic-prompt results remain below the earlier attribute-prompt models on these sets (for example BKAI 74.4 versus 82.1, ColonDB 69.7 versus 74.7 for the matched image-reflection baseline).

## Decision

Integrity accepted. The detail branch passes the gate with a modest, small-target-led gain. No model selection is made from these Test results.
