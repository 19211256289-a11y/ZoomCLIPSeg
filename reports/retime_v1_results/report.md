# retime_v1: time of one inference run (detail_seed666, NVIDIA GeForce RTX 5060 Ti, batch size 1), three repetitions

| stochastic passes | repeat 1 (ms) | repeat 2 (ms) | repeat 3 (ms) | mean (ms) |
|---|---|---|---|---|
| 1 | 66 | 62 | 64 | 64 |
| 5 | 178 | 180 | 186 | 181 |
| 15 | 484 | 498 | 521 | 501 |
| 30 | 933 | 965 | 935 | 945 |

Earlier measurements of the 30-pass run: 760 ms (zoom_probe_v1, 2026-10-06) and 942 ms (text_probe_v1, 2026-10-07).
