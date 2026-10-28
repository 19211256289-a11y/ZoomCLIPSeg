# Acceptance: retime_v1 (2026-10-09, 16:40 JST)

Accepted. Runner code on the PC equals the deployed copy (sha256 f5b49d15bcdd...). Report and results.json agree.
Checkpoint detail_seed666 (sha256 3e06610d8c4c...), RTX 5060 Ti, torch 2.9.0+cu128, batch size 1, whole image,
GPU synchronised, data loading excluded.

| passes | repeats (ms) | mean (ms) |
|---|---|---|
| 1 | 66, 62, 64 | 64 |
| 5 | 178, 180, 186 | 181 |
| 15 | 484, 498, 521 | 501 |
| 30 | 933, 965, 935 | 945 |

Reading: the 30-pass time agrees with text_probe_v1 (942 ms) and not with zoom_probe_v1 (760 ms); the earlier 0.76 s
is treated as the outlier. Ratio 30/1 pass about 14.7. Used in the manuscript (Section 5.8) with the qualification
that one pass was only shown to match 30 passes without SCZ.
