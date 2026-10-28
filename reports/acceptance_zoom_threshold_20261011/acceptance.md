# Acceptance: zoom_threshold_v1 (11 Oct 2026)

Verdict: **accepted**; the reading rule fixed in advance is met.

- SCZ gain (OOD mean) at thresholds 0.3/0.4/0.5/0.6/0.7: +2.35, +2.31, +2.13, +1.90, +1.79 pp; every interval above zero, every seed positive.
- Single pass OOD: 76.97, 77.18, 77.36, 77.50, 77.62; SCZ OOD: 79.31, 79.49, 79.48, 79.40, 79.42.
- Reproduction: scz_t50 minus accepted t15 = +0.01 pp (tolerance 0.25 pp).

Checks: plan digest in queue equals evaluation_plan.json digest; code SHA-256 of every file in the plan equals the deployed package; protocol SHA-256 equals the deployed suite YAML; all jobs completed, no error recorded; means of every configuration recomputed from the 25 per-image case files equal the report (max deviation 0.0000 pp); single-pass reproduction against the accepted evaluation passed for every run and cohort (runner gate, tolerance 0.25 pp); ETIS-Larib not evaluated.
