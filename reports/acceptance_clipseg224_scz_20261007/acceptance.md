# Acceptance: clipseg224_scz_v1 (2026-10-07, 21:30 JST)

Accepted. Plan digest matches the report; code and protocol on the PC equal the deployed copies; OOD means per seed recomputed from the 30 case files agree with results.json to 1e-13. Evaluation and training manifests were listed but not re-hashed.

Result (three seeds, CLIPSeg at 224 x 224): single pass OOD 78.66, SCZ 79.94; difference +1.29 pp [+0.35, +2.22], p=0.007; per seed +0.57, +2.27, +1.02. By set: ColonDB -0.04, ClinicDB +0.60, CVC-300 -0.08, BKAI +4.66 [+2.37, +7.04]; the OOD gain is carried by BKAI. Gain at 224 minus gain at 352: +1.33 [+0.25, +2.46], p=0.020.

Verdicts: transfers True; does_not_transfer False; larger_at_low_resolution True.

Notes: follow-up suite, designed after seed 666 of clipseg_scz_v1 had been inspected (disclosed in the protocol). The manuscript does not name a higher input resolution as a remedy, because the single-pass OOD mean at 352 (78.62) does not exceed that at 224 (78.66); it states this instead.
