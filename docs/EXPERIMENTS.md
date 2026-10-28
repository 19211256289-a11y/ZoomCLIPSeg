# From the paper to the code

Every experiment is a *suite*: a protocol `MedCLIPSeg/<package>/suites/<suite>.yaml`, written and locked before its
results existed; the records `MedCLIPSeg/outputs_research/<suite>/`; an acceptance audit `reports/acceptance_*`; and
a generated report `reports/<suite>_results/` (older suites: `reports/<name>_<date>/`).

Run a package with `python run.py <package> <action>`; `python run.py <package> --help` lists its actions.
Training packages are marked **train**; all others only evaluate existing weights.

## Text and baseline (Section 5.1, Table 1)

| Paper | Suite | Package | Records |
|---|---|---|---|
| Models with attribute captions: three training variants × three seeds (Table 1; reference row of Table 2) | `formal_kvasir_v1` | `research_formal` **train** | `reports/acceptance_formal_val_20261003`, `reports/formal_seed66{6,7}_20261002` |
| Test-time text interventions on these nine models (Table 1) | `validation_kvasir_v1` | `research_validation` | `reports/acceptance_validation_20261003` |
| Sentence substitution for fixed-sentence models: empty text, other modality, other wording (Section 5.1) | `text_probe_v1` | `research_text_probe` | `reports/text_probe_v1_results` |

## Polyps (Sections 5.2–5.3, Tables 2–4, A2, A4, A5, Fig. 3)

| Paper | Suite | Package | Records |
|---|---|---|---|
| Fixed-sentence models, four arms (baseline, global residual query, detail branch, both), seeds 666–668 | `formal_detail_mix_kvasir_v1` | `research_detail_formal` **train** | `reports/acceptance_4090_20261004`, `reports/acceptance_5090_*_20261004` |
| Their single-pass Kvasir-SEG test and OOD evaluation | `test_detail_mix_kvasir_v1` | `research_detail_test` | `reports/test_detail_mix_20261004`, `reports/step0_val_case_analysis_20261004` |
| Seeds 669 and 670, all four arms | `formal_detail_mix_kvasir_v2` | `research_detail_ext` **train** | `reports/formal_detail_mix_kvasir_v2_results` |
| Choice of the SCZ target fraction τ on source validation images (Section 3.3) | `pilot_zoom_kvasir_v1` | `research_zoom` | `reports/pilot_zoom_20261005` |
| Pre-specified SCZ test (Section 5.2) | `test_zoom_kvasir_v1` | `research_zoom_test` | `reports/test_zoom_20261005` |
| SCZ on seeds 669–670; on every training variant (Table 2, Table A4) | `zoom_kvasir_seeds_v1`, `zoom_ablation_kvasir_v1` | `research_zoom_ext` | `reports/zoom_kvasir_seeds_v1_results`, `reports/zoom_ablation_kvasir_v1_results` |
| Scale versus new pixels; constants of Eq. (2) (Section 5.3, Appendix B.2) | `zoom_probe_v1` | `research_zoom_probe` | `reports/zoom_probe_v1_results` |
| Crop factor (Table 3, polyps) | `zoom_box_v1`, `zoom_box_v2` | `research_zoom_box`, `research_zoom_box2` | `reports/zoom_box_v1_results`, `reports/zoom_box_v2_results` |
| Tiling, multi-scale and flip averaging (Table 4) | `zoom_compare_v1` | `research_zoom_compare` | `reports/zoom_compare_v1_results` |
| Inference budget; flip averaging combined with SCZ | `zoom_budget_v1` | `research_zoom_budget` | `reports/zoom_budget_v1_results` |
| One stochastic pass per run (Section 5.3, Appendix B.2) | `zoom_mc1_v1` | `research_zoom_mc1` | `reports/zoom_mc1_v1_results` |
| Binarisation threshold 0.3–0.7 (Appendix B.2) | `zoom_threshold_v1` | `research_zoom_threshold` | `reports/zoom_threshold_v1_results` |

## Breast ultrasound (Section 5.4, Tables 3, 5, A4, Fig. A1)

| Paper | Suite | Package | Records |
|---|---|---|---|
| Baseline, detail branch and both arms on BUSI, seeds 666–668 | `formal_detail_mix_busi_v1` | `research_detail_ext` **train** | `reports/formal_detail_mix_busi_v1_results`, `reports/acceptance_busi_20261005` |
| SCZ with the quantile-matched τ = 0.086 | `zoom_busi_v1` | `research_zoom_ext` | `reports/zoom_busi_v1_results` |
| Crop factor (Table 3, ultrasound) | `zoom_box_busi_v1` | `research_zoom_box_busi` | `reports/zoom_box_busi_v1_results` |

## Held-out set and external references (Sections 5.5–5.6, Tables 6–8, A6, A7)

| Paper | Suite | Package | Records |
|---|---|---|---|
| One-time ETIS-Larib evaluation (Table 6) | `zoom_etis_v1` | `research_zoom_etis` | `reports/zoom_etis_v1_results` |
| Polyp-PVT trained on the same 800 images, seeds 666–668 (Tables 7, 8) | `polyp_pvt_kvasir_v1` | `research_polyp_pvt` **train** | `reports/polyp_pvt_kvasir_v1_results` |
| Polyp-PVT, seeds 669–670 | `pvt_seeds_v1` | `research_pvt_seeds` **train** | `reports/pvt_seeds_v1_results` |
| SCZ applied to Polyp-PVT (Table 7) | `pvt_zoom_v1` | `research_pvt_zoom` | `reports/pvt_zoom_v1_results` |
| Polyp-PVT on BUSI (Table A6) | `pvt_ext_v1` | `research_pvt_ext` **train** | `reports/pvt_ext_v1_results` |
| CLIPSeg at 352 × 352 and 224 × 224 (Table A7) | `clipseg_scz_v1`, `clipseg224_scz_v1` | `research_clipseg`, `research_clipseg224` **train** | `reports/clipseg_scz_v1_results`, `reports/clipseg224_scz_v1_results` |

## Controls, cost and figures (Section 6, Appendix B, Tables A3, A8)

| Paper | Suite | Package | Records |
|---|---|---|---|
| Training recipe, detail input and scale augmentation, retrained on one GPU (Table A3) | `formal_controls_kvasir_v1` | `research_controls` **train** | `reports/formal_controls_kvasir_v1_results` |
| Time per image (Table A8) | `retime_v1` | `research_retime` | `reports/retime_v1_results` |
| Predicted masks for the image figures (Figs. 1, 2, 4) | `zoom_export_v1` | `research_zoom_export` | `MedCLIPSeg/outputs_research/zoom_export_v1` |

Tables 8 (gap to the dedicated network by lesion size), A2 (IoU), the deduplicated and BCa analyses are computed from
the per-image records by the scripts in `paper/scripts/` (`gap_analysis.py`, `iou_table.py`, `dedup_analysis.py`,
`bca_intervals.py`, `review_analyses.py`).

## Packages that are libraries for the ones above

| Package | Role |
|---|---|
| `research` | detail-branch encoder and decoder (`modules.py`), dual-scale 224/448 data pipeline (`data.py`), shared helpers |
| `research_v6` | paired-view objective, model builder and data loading that `research_formal` extends |
| `research_spatial` | model base class that the detail-branch model extends |
| `research_detail_mix` | the detail-branch model `DetailMixCLIP` (`model.py`) and its objective |

## Rebuilding reports and repeating runs

- `report` rebuilds a suite's report from its released per-image records; it needs no weights or images. For the
  released suites the rebuilt `report.md` is byte-identical to the one in `reports/`.
- `status` reads the local execution state in `control/`, which is not released; on a fresh clone it shows
  `not_started`.
- `run` refuses to overwrite released records whose plan no longer matches the local code, inputs or checkpoints
  (for example, different absolute paths on a new machine). To repeat a suite, move
  `MedCLIPSeg/outputs_research/<suite>/` and its `reports/` folder aside first; the runner then locks a new plan.
