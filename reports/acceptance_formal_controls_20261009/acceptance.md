# Acceptance: formal_controls_kvasir_v1 (2026-10-09, 23:00 JST)

Accepted. Queue state completed (21 training jobs, all evaluations), plan digest b84c6e4e... equals the plan file;
control code (research_controls/common.py, recipes.py, train_controls.py) and protocol yaml on the PC equal the hashes
in the plan. Five training jobs were interrupted and resumed from their checkpoints (detail 666-668, orig_fixed 667-668;
the 2026-10-08 power cut affected orig_fixed_seed667). Arm means and all ten macro OOD contrasts were recomputed from
the 105 per-image case files (seeds 666-668, five test sets) and agree with results.json to two decimals.

Reading rules fixed in advance:
- two_pass_after_scale_augmentation: retained (+1.48 pp; per seed +1.77, +1.62, +1.05).
- detail_information (real 448 vs upsampled 224): supported by the rule (mean over five sets +0.08 pp; per seed +0.10,
  +0.48, -0.33), but the difference is negligible and every per-set interval includes zero; the manuscript states that
  the control does not distinguish information from capacity.

Descriptive: Dice fix +0.92, ordinary flip +1.78, paired vs ordinary flip -0.29, recipe total +2.42, detail branch
+0.29, scale augmentation +0.16 (redistributes accuracy by lesion size: <1% 22.6 -> 34.7, 1-6% lose 1.5-2.2 pp),
SCZ on baseline +2.33 / detail +1.78 / after scale augmentation +1.48 (OOD macro, pp).
Note: single-view arms use half the forward/backward passes per step of the paired arms.
