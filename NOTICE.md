# Notices

This repository is released under the Apache License 2.0 (`LICENSE`). It contains or builds on the following
third-party work, which keeps its own copyright and terms.

| Component | Location | Origin | Terms |
|---|---|---|---|
| MedCLIPSeg | `MedCLIPSeg/` (training and test scripts, `trainers/`, `datasets/`, `utils/`, `configs/`, `assets/`) | [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg), commit `31d2220` | Apache-2.0, `MedCLIPSeg/LICENSE`. Our changes to upstream files are listed in `docs/upstream_changes.patch`. |
| CLIP | `MedCLIPSeg/clip/` | [openai/CLIP](https://github.com/openai/CLIP), as bundled by MedCLIPSeg | MIT |
| OpenCLIP | `MedCLIPSeg/open_clip_lib/` | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip), as bundled by MedCLIPSeg | MIT |
| timm layers | `MedCLIPSeg/biomedclip/layers/` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models), as bundled by MedCLIPSeg | Apache-2.0 |
| Polyp-PVT and PVTv2 | `MedCLIPSeg/research_polyp_pvt/lib/` | [DengPingFan/Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) (commit `cd71fbf`), [whai362/PVT](https://github.com/whai362/PVT) | Free for research and education use only, as stated by the source. Changes are described in each file's header. |

Pretrained weights (UniMed-CLIP, BiomedBERT, PVTv2-B2, CLIPSeg) and datasets are not redistributed; they are
downloaded from their sources (`docs/SETUP.md`) under their own licences. `MedCLIPSeg/data/ETIS/` contains only a
file manifest; the ETIS-Larib images are restricted to research and education use by their owners.
