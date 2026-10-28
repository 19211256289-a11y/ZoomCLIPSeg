# clipseg_scz_v1: scale-canonical zoom on CLIPSeg (frozen CLIP encoders, fixed sentence, three seeds)

Decoder fine-tuned on the Kvasir-SEG training split, last-epoch weights, no selection on validation or test data. OOD = equal-weight mean of the four OOD polyp sets.

| inference | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean (SD over seeds) |
|---|---|---|---|---|---|---|
| single pass | 87.80 | 70.52 | 79.56 | 85.80 | 78.60 | 78.62 (0.25) |
| SCZ (locked rule) | 87.70 | 68.86 | 79.69 | 86.59 | 79.15 | 78.57 (0.72) |
| SCZ minus single pass (pp) | -0.10 [-1.90, +1.90], p=0.9291 | -1.66 [-2.79, -0.54], p=0.0042 | +0.13 [-1.73, +2.35], p=0.9216 | +0.79 [-1.67, +3.02], p=0.5282 | +0.55 [-2.44, +3.42], p=0.7224 | -0.05 [-1.16, +1.06], p=0.9395 |

OOD gain per seed: +0.41, -0.56, +0.00. Second pass on 82% of the OOD images.

OOD images pooled, by lesion area (DSC %, mean over seeds):

| lesion area | n | single pass | SCZ |
|---|---|---|---|
| <1% | 55 | 31.4 | 37.0 |
| 1-3% | 179 | 76.0 | 74.9 |
| 3-5.9% | 158 | 83.4 | 81.3 |
| 5.9-15% | 129 | 84.3 | 82.9 |
| >=15% | 60 | 64.6 | 62.9 |

## Verdicts by the rules fixed in advance

- SCZ transfers to CLIPSeg (gain positive for every seed, interval excludes zero): **False**
- SCZ does not transfer (interval includes zero or lies below it): **True**

Training time per seed: 20 min, 20 min, 20 min. Final training loss: 0.0586, 0.0579, 0.0580.

Consequence fixed in advance: If transfers holds, the manuscript states that SCZ also helps a second frozen-encoder, fixed-resolution text-prompted model and gives the gain. If does_not_transfer holds, it states that SCZ did not help CLIPSeg and keeps the scope "MedCLIPSeg variants only". A positive gain that is not positive for every seed is reported as such and counts as neither. The accuracy of CLIPSeg itself is reported for context and is not compared with our models as a ranking, because its recipe was not tuned.

## Disclosures

- The five polyp test sets were used in earlier stages of this project; ETIS-Larib is not touched.
- The training recipe of CLIPSeg was fixed here without tuning and differs from the one used for the values reported in the MedCLIPSeg paper.
- The decoder starts from released weights, so that the seeds differ in mini-batch order and augmentation only.
- The target fraction and the zoom constants are those locked for MedCLIPSeg at 224 x 224; nothing is re-derived for this model.
