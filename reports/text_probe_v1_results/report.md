# text_probe_v1: text at test time, one stochastic pass, and timing (detail-branch model, seeds 666-668)

Nothing was trained. Single pass of the whole image, without SCZ. OOD = equal-weight mean of the four OOD polyp sets.

| condition | text | Kvasir DSC | OOD DSC (SD over seeds) | minus the training sentence, OOD [95% CI], p | per seed |
|---|---|---|---|---|---|
| training sentence (accepted single pass) | `a polyp in a colonoscopy image` | 89.47 | 76.94 (0.44) |  |  |
| empty text | `` | 89.51 | 76.84 (0.48) | -0.10 [-0.23, -0.00], p=0.0723 | -0.09, -0.15, -0.08 |
| class sentence of the other modality | `a breast tumor in an ultrasound image` | 89.50 | 76.92 (0.37) | -0.02 [-0.09, +0.04], p=0.5018 | -0.10, +0.12, -0.09 |
| other wording A | `polyp` | 89.50 | 76.88 (0.48) | -0.06 [-0.15, +0.01], p=0.1299 | -0.08, -0.09, -0.02 |
| other wording B | `a colorectal polyp seen during endoscopy` | 89.47 | 76.91 (0.40) | -0.03 [-0.07, +0.01], p=0.1985 | -0.03, +0.02, -0.08 |
| training sentence, one stochastic pass | `a polyp in a colonoscopy image` | 89.47 | 76.94 (0.46) | +0.00 [-0.03, +0.04], p=0.8151 | +0.05, -0.04, -0.00 |

Empty text: empty string.

Per test set, DSC (%):

| condition | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| training sentence (accepted single pass) | 89.47 | 70.26 | 82.70 | 78.89 | 75.91 |
| empty text | 89.51 | 70.17 | 82.29 | 78.98 | 75.90 |
| class sentence of the other modality | 89.50 | 70.24 | 82.58 | 78.81 | 76.04 |
| other wording A | 89.50 | 70.18 | 82.45 | 79.00 | 75.87 |
| other wording B | 89.47 | 70.22 | 82.58 | 78.92 | 75.92 |
| training sentence, one stochastic pass | 89.47 | 70.23 | 82.73 | 78.88 | 75.95 |

Difference to the training sentence per test set (pp):

| condition | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| empty text | +0.04 [-0.04, +0.13], p=0.3421 | -0.09 [-0.21, +0.02], p=0.1345 | -0.41 [-0.81, -0.10], p=0.0106 | +0.09 [-0.07, +0.23], p=0.2613 | -0.01 [-0.21, +0.18], p=0.9574 |
| class sentence of the other modality | +0.03 [-0.03, +0.09], p=0.3197 | -0.02 [-0.10, +0.06], p=0.6453 | -0.13 [-0.32, +0.04], p=0.1922 | -0.08 [-0.21, +0.04], p=0.2136 | +0.13 [-0.01, +0.28], p=0.0865 |
| other wording A | +0.03 [-0.03, +0.10], p=0.3009 | -0.09 [-0.17, -0.00], p=0.0384 | -0.25 [-0.53, -0.03], p=0.0496 | +0.12 [+0.02, +0.22], p=0.0259 | -0.04 [-0.21, +0.12], p=0.6455 |
| other wording B | +0.00 [-0.03, +0.04], p=0.8860 | -0.04 [-0.10, +0.01], p=0.1637 | -0.12 [-0.24, -0.02], p=0.0285 | +0.03 [-0.06, +0.11], p=0.4787 | +0.01 [-0.07, +0.10], p=0.7673 |
| training sentence, one stochastic pass | -0.00 [-0.04, +0.03], p=0.9403 | -0.04 [-0.08, +0.01], p=0.0968 | +0.03 [-0.06, +0.10], p=0.5346 | -0.01 [-0.09, +0.07], p=0.8228 | +0.04 [-0.03, +0.11], p=0.3145 |

## Verdicts by the rules fixed in advance

- The content of the sentence matters (other-modality sentence lowers the OOD mean): **False**
- The sentence acts as a constant input (empty and other-modality text both within 0.5 pp): **False**
- Sensitive to the wording (at least one other wording changes the OOD mean): **False**

## Time of one inference run

detail_seed666 on Kvasir, NVIDIA GeForce RTX 5060 Ti, batch size 1. batch size 1; one inference run of the whole image; GPU synchronised before and after; data loading excluded.

| stochastic passes | mean (ms) | SD (ms) | median (ms) | images |
|---|---|---|---|---|
| 1 | 66 | 3 | 66 | 50 |
| 5 | 195 | 7 | 195 | 50 |
| 15 | 505 | 14 | 504 | 50 |
| 30 | 942 | 19 | 939 | 50 |

## Definitions

- **inference**: single pass of the whole image, mean of 30 stochastic forward passes, as in the accepted evaluation
- **noise**: the random stream is reset before every text condition, so that the conditions share their random draws
- **single01**: the training sentence with one stochastic forward pass (an actual run, not a prefix of the 30)

Consequence fixed in advance: The manuscript reports each outcome as it is. If acts_as_constant holds, it states that with one sentence per dataset the text acts as a constant input of the trained model and that "text-guided" names the model family. If text_content_matters holds, it states that the trained model still depends on the content of its sentence. The sensitivity to the wording is reported with its size in either case.

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- Only the detail-branch model and the seeds 666-668 are evaluated; SCZ is not applied.
- ETIS-Larib is not touched.
- The models were never trained with the alternative texts; the analysis measures sensitivity at test time only.
