# clipseg224_scz_v1: scale-canonical zoom on CLIPSeg at 224 x 224 (frozen CLIP encoders, fixed sentence, three seeds)

Decoder fine-tuned on the Kvasir-SEG training split, last-epoch weights, no selection on validation or test data. OOD = equal-weight mean of the four OOD polyp sets.

| inference | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI | OOD mean (SD over seeds) |
|---|---|---|---|---|---|---|
| single pass | 87.47 | 71.36 | 79.59 | 85.81 | 77.88 | 78.66 (0.37) |
| SCZ (locked rule) | 88.10 | 71.32 | 80.18 | 85.73 | 82.54 | 79.94 (0.54) |
| SCZ minus single pass (pp) | +0.63 [-0.47, +1.96], p=0.3475 | -0.04 [-1.15, +1.07], p=0.9458 | +0.60 [-1.40, +2.66], p=0.5881 | -0.08 [-1.76, +1.61], p=0.9320 | +4.66 [+2.37, +7.04], p=0.0002 | +1.29 [+0.35, +2.22], p=0.0073 |

OOD gain per seed: +0.57, +2.27, +1.02. Second pass on 82% of the OOD images.

OOD images pooled, by lesion area (DSC %, mean over seeds):

| lesion area | n | single pass | SCZ |
|---|---|---|---|
| <1% | 55 | 34.8 | 39.9 |
| 1-3% | 179 | 77.0 | 79.5 |
| 3-5.9% | 158 | 83.1 | 82.4 |
| 5.9-15% | 129 | 84.0 | 84.0 |
| >=15% | 60 | 63.8 | 61.4 |

## Verdicts by the rules fixed in advance

- SCZ transfers to CLIPSeg (gain positive for every seed, interval excludes zero): **True**
- SCZ does not transfer (interval includes zero or lies below it): **False**
- The gain is larger at 224 x 224 than at 352 x 352: **True** (difference of the OOD gains: +1.33 [+0.25, +2.46], p=0.0202; per seed +0.16, +2.83, +1.01; at 352 x 352: single pass 78.62, SCZ 78.57)

Training time per seed: 8 min, 8 min, 8 min. Final training loss: 0.0623, 0.0646, 0.0625.

Consequence fixed in advance: The manuscript reports the gain of SCZ at both resolutions as measured. If transfers holds here, it states that SCZ helps CLIPSeg at the resolution of our models; if larger_at_low_resolution also holds, it states that the gain shrinks when the encoder sees more pixels, in line with the explanation by a fixed low-resolution view, and names a higher input resolution as an alternative remedy where the model allows it. If neither holds, it states that SCZ did not help CLIPSeg and keeps the scope "MedCLIPSeg variants only". The accuracy of CLIPSeg itself is reported for context and not as a ranking.

## Disclosures

- The five polyp test sets were used in earlier stages of this project; ETIS-Larib is not touched.
- The training recipe of CLIPSeg was fixed here without tuning and differs from the one used for the values reported in the MedCLIPSeg paper.
- The decoder starts from released weights, so that the seeds differ in mini-batch order and augmentation only.
- The target fraction and the zoom constants are those locked for MedCLIPSeg at 224 x 224; nothing is re-derived for this model.
- The released CLIPSeg was trained at 352 x 352; running it at 224 x 224 is outside its native setting.
- This suite was designed after one seed of the companion suite had been inspected.
