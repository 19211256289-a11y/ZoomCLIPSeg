# pvt_seeds_v1: Polyp-PVT reference, training seeds [669, 670] (full Kvasir-SEG training set)

Released recipe unchanged, last-epoch weights, no selection on validation or test data. OOD = equal-weight mean of the four OOD polyp sets.

| seed | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean | ETIS |
|---|---|---|---|---|---|---|---|
| 669 | 91.45 | 75.51 | 85.89 | 87.97 | 85.59 | 83.74 | 75.47 |
| 670 | 91.87 | 75.64 | 85.70 | 85.95 | 83.80 | 82.77 | 74.45 |

Missed lesions on the four OOD sets (per seed): [2, 0]. Training time per seed: 1.01 h, 1.05 h. Final training loss: 0.0842, 0.1207.

## Reading rule fixed in advance

The manuscript replaces the three-seed values of the reference by the five-seed values (seeds 666-670) wherever it compares with five-seed results of our models, whatever the direction of the change, and keeps the three-seed values only in analyses that are defined on the seeds 666-668.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- ETIS-Larib was evaluated once for our method and three times before for this reference (seeds 666-668).
