# Acceptance: clipseg_scz_v1 (2026-10-07, 21:30 JST)

Accepted. Plan digest matches the report; code and protocol on the PC equal the deployed copies (hashes in audit.json; __init__.py compared from the local copy only); OOD means per seed recomputed from the 24 case files agree with results.json to 1e-13. Evaluation and training manifests were listed but not re-hashed.

Result (three seeds, CLIPSeg at 352 x 352): single pass OOD 78.62, SCZ 78.57; difference -0.05 pp [-1.16, +1.06], p=0.94; per seed +0.41, -0.56, +0.00. ColonDB -1.66 [-2.79, -0.54]. Lesions <1%: 31.4 -> 37.0; larger strata lose 1-2 pp.

Verdicts by the rules fixed in advance: transfers False; does_not_transfer True.
