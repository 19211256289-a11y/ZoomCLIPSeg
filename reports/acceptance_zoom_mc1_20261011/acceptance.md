# Acceptance: zoom_mc1_v1 (11 Oct 2026)

Verdict: **accepted**; both reading rules fixed in advance are met.

- SCZ with one stochastic pass per run vs the locked method (OOD mean): +0.06 pp [-0.10, +0.21]; non-inferiority margin -0.5 pp: met.
- SCZ gain with one pass (scz01 - single01, OOD mean): +2.18 pp [+1.37, +2.99]; positive for all five seeds (+2.06, +2.24, +2.25, +2.73, +1.64): met.
- OOD DSC: single30 77.36, single01 77.35, scz01 79.53, t15 79.48. Kvasir: 89.54, 89.54, 89.98, 89.92.
- Time per image (RTX 5060 Ti, batch 1, 150 CVC-ColonDB images, 3 repetitions): single01 61 ms, scz01 105 ms, single30 832 ms, t15 1467 ms; repetition means vary by up to about 20%.

Checks: plan digest in queue equals evaluation_plan.json digest; code SHA-256 of every file in the plan equals the deployed package; protocol SHA-256 equals the deployed suite YAML; all jobs completed, no error recorded; means of every configuration recomputed from the 25 per-image case files equal the report (max deviation 0.0000 pp); single-pass reproduction against the accepted evaluation passed for every run and cohort (runner gate, tolerance 0.25 pp); ETIS-Larib not evaluated.

Event: 07:56 JST first start; the console window was frozen (Windows QuickEdit) after the second batch of detail_seed666/Kvasir; restarted 08:46 JST with the same plan digest; the interrupted job was recomputed from scratch.
