# ZoomCLIPSeg

Code, locked protocols and per-image records for the paper

**Vision-Language Medical Image Segmentation Without Mask-Derived Prompts: Where It Fails and a Training-Free Remedy**
Yanan Liu, Hongmei Yang, Kefeng Sun, Min Liu, Na Li, Jiateng Liu, Rensong Liu — Qilu University of Technology
(Shandong Academy of Sciences) and Jinan Second Mental Health Center. Manuscript under review.

Text-guided segmenters such as [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) are prompted with lesion
attributes (location, size, number) that are read from the annotation mask. This repository studies the automatic
setting in which every image receives the same class sentence, on colonoscopy and breast-ultrasound data:

- **Fixed-sentence baseline.** A frozen UniMed-CLIP encoder with MedCLIPSeg's probabilistic adapters, trained with one
  class sentence (for example *"a polyp in a colonoscopy image"*), recovers most of the accuracy lost when mask-derived
  captions are removed, and is then insensitive to the wording of the sentence.
- **Scale-canonical zoom (SCZ).** A training-free second pass: the first prediction defines a window in which the
  lesion would cover a training-typical fraction of the view; the window is cropped from the original image, segmented
  by the same network, and pasted back. It raises the out-of-distribution Dice by 2.5 points on four polyp sets and by
  4.4 points on the held-out ETIS-Larib set, for every one of five training seeds. A square crop of twice the predicted
  box performs as well as the window rule.
- **Detail branch (optional).** A zero-initialised convolutional side branch (+0.21 M parameters) that reads the image
  at 448 × 448 and adds a residual to the logits; it stabilises training on breast ultrasound.

ZoomCLIPSeg denotes the full configuration (fixed sentence + detail branch + SCZ).

## Where the method is implemented

| Component | Code |
|---|---|
| SCZ window rule, crop and paste-back (Eq. 2, Algorithm in Fig. 2b) | `MedCLIPSeg/research_zoom/geometry.py` (`window_rule`, `crop`, `compose`) |
| Two-pass inference loop | `MedCLIPSeg/research_zoom/evaluate.py`, `MedCLIPSeg/research_zoom_ext/` |
| Detail branch: encoder and decoder | `MedCLIPSeg/research/modules.py` (`DetailEncoder`, `DetailDecoder`) |
| Detail branch attached to the network (`DetailMixCLIP`) | `MedCLIPSeg/research_detail_mix/model.py` |
| 224 / 448 dual-scale input pipeline | `MedCLIPSeg/research/data.py` |
| Training recipe: fixed sentence, paired reflection, per-image Dice | `MedCLIPSeg/research_formal/`, `MedCLIPSeg/train_detail_formal.py` |
| Corrected Dice term (Appendix A of the paper) | `MedCLIPSeg/train.py`; see `docs/upstream_changes.patch` |
| Metrics (DSC, NSD, size strata) | `MedCLIPSeg/research_formal/metrics.py` |

## Repository layout

```
run.py                      cross-platform launcher for every experiment package
MedCLIPSeg/                 MedCLIPSeg code base with the research packages of this paper
  research_*/               one package per experiment; suites/*.yaml are the locked protocols
  outputs_research/<suite>/ released records: locked plans, per-image results (cases.csv), summaries
  configs/, data/ETIS/      dataset configurations; ETIS-Larib split manifest (no images)
reports/                    acceptance audits and result reports of every experiment in the paper
paper/                      scripts that regenerate the paper's figures and tables from paper/data/
docs/                       SETUP.md, EXPERIMENTS.md, upstream_changes.patch
SOURCE_MANIFEST.csv         SHA-256 and original path of every file copied from the experiment workspace
```

## Getting started

0. On Windows, clone into a short path or run `git config --global core.longpaths true` first; some record paths are
   up to 134 characters long.
1. Set up the environment, pretrained models and datasets: [docs/SETUP.md](docs/SETUP.md).
2. Find the experiment behind a table or figure: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
3. Run it:

```bash
python run.py research_zoom_etis status   # progress of the ETIS-Larib evaluation
python run.py research_zoom_etis report   # rebuild its report from the per-image records
python run.py research_zoom_etis run      # evaluate (needs the trained weights)
```

Each experiment is a *suite* with a protocol (`MedCLIPSeg/research_*/suites/<suite>.yaml`) that was written and locked
before its results existed. Runners verify the SHA-256 digests of the protocol, the code, the input files and the
checkpoints against the locked plan and refuse to run if any of them differ, so the code is released exactly as it
ran. `.gitattributes` disables line-ending conversion for this reason. `report` works on the released records
without weights or images; to repeat a run on a new machine, move the suite's records aside first
(see the end of [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).

## Records

- `MedCLIPSeg/outputs_research/<suite>/`: the locked plan of each suite and, for every model, seed and test set, a
  `cases.csv` with one row per image (DSC, NSD, lesion area, predicted area, second-pass window, ...) plus a
  `summary.json`. Large binary outputs (probability maps, prediction images, checkpoints) are not included.
- `reports/acceptance_*`: the audit of each experiment before its results were used (some are written in Chinese).
- `reports/<suite>_results/`: the report generated by the suite's runner (`report.md`, `results.json`).

The records contain absolute paths of the machines on which they were produced (a Windows workstation and rented
Linux GPU servers); they are kept unchanged because the plans store digests of these files.

## Trained weights

The final-epoch weights of all models in the paper will be published as release assets of this repository. Their
SHA-256 digests are recorded in the locked plans, and evaluation runners check them before use.

## Regenerating figures and tables

```bash
cd paper
python scripts/make_figures.py && python scripts/make_tables.py
```

See [paper/README.md](paper/README.md).

## Scope of this release

The repository contains every package that a reported experiment imports, and the records of those experiments.
Exploratory pilots of the development phase (prompt, soft-contrast, query and spatial-module screens, early
training rounds) and an unpublished follow-up study are not included. The original Windows batch launchers are
replaced by `run.py`; multi-GPU orchestration scripts for the rented servers are not included, and every suite can be
run with a single process.

## License and acknowledgements

This project builds on [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (Apache-2.0) and is released under the
Apache License 2.0 (`LICENSE`). Bundled third-party components keep their own terms; in particular the vendored
Polyp-PVT network in `MedCLIPSeg/research_polyp_pvt/lib/` is for research and education use only. See
[NOTICE.md](NOTICE.md). Datasets are not redistributed; please follow the terms of each dataset.

## Citation

```bibtex
@misc{liu2026zoomclipseg,
  title  = {Vision-Language Medical Image Segmentation Without Mask-Derived Prompts:
            Where It Fails and a Training-Free Remedy},
  author = {Liu, Yanan and Yang, Hongmei and Sun, Kefeng and Liu, Min and Li, Na and Liu, Jiateng and Liu, Rensong},
  year   = {2026},
  note   = {Manuscript under review. Code: https://github.com/19211256289-a11y/ZoomCLIPSeg}
}
```

Please also cite MedCLIPSeg (Koleilat et al., CVPR 2026) and UniMed-CLIP (Khattak et al., 2024).
