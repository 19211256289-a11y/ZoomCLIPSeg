# Setup

The experiments ran with Python 3.10, PyTorch 2.9.0 (CUDA 12.8), MONAI 1.6.0 and transformers 4.56.0, on Windows 11
(RTX 5060 Ti) and Linux GPU servers (RTX 4090, RTX 5090). All paths below are relative to the repository root.

## 1. Environment

```bash
conda create -n zoomclipseg python=3.10 -y
conda activate zoomclipseg
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Pretrained models

`run.py` runs offline (`HF_HUB_OFFLINE=1`), so download these once:

| Model | Location | Source |
|---|---|---|
| UniMed-CLIP ViT-B/16 image encoder | `MedCLIPSeg/checkpoints/unimed_clip_vit_b16.pt` | `checkpoints/unimed_clip_vit_b16.pt` in the Hugging Face model repo [TahaKoleilat/MedCLIPSeg](https://huggingface.co/TahaKoleilat/MedCLIPSeg) |
| BiomedBERT text encoder | `.cache/huggingface/` (Hugging Face cache) | [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) |
| PVTv2-B2 backbone (Polyp-PVT reference only) | `MedCLIPSeg/checkpoints/pvt_v2_b2.pth` | <https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth> |
| CLIPSeg (Table A7 only) | `MedCLIPSeg/checkpoints/clipseg-rd64-refined/` | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download TahaKoleilat/MedCLIPSeg checkpoints/unimed_clip_vit_b16.pt --local-dir MedCLIPSeg
HF_HOME=.cache/huggingface huggingface-cli download microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
huggingface-cli download CIDAS/clipseg-rd64-refined --local-dir MedCLIPSeg/checkpoints/clipseg-rd64-refined
```

(On Windows PowerShell, set the cache with `$env:HF_HOME=".cache/huggingface"` before the second command.)

## 3. Datasets

All datasets except ETIS-Larib use the splits and prompt files released with MedCLIPSeg
([TahaKoleilat/MedCLIPSeg dataset](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg)); the folder layout is
described in `MedCLIPSeg/assets/DATASETS.md`. This paper uses:

| Role | Datasets (folder names under `MedCLIPSeg/data/`) |
|---|---|
| Polyps, source | `Kvasir` |
| Polyps, out-of-distribution | `ColonDB`, `ClinicDB`, `CVC300`, `BKAI` |
| Polyps, held out | `ETIS` (see below) |
| Breast ultrasound, source | `BUSI` |
| Breast ultrasound, out-of-distribution | `BUSBRA`, `BUSUC`, `BUID`, `UDIAT` |

```
MedCLIPSeg/data/<Dataset>/
  Prompts_Folder/   Train_text.xlsx, Val_text.xlsx, Test_text_original.xlsx, ...
  Train_Folder/{img,label}/
  Val_Folder/{img,label}/
  Test_Folder/{img,label}/
```

**ETIS-Larib.** Download `ETIS-LaribPolypDB.rar` from the link on the
[official page](https://polyp.grand-challenge.org/ETISLarib/) and place the 196 images as
`MedCLIPSeg/data/ETIS/images/1.tif ... 196.tif` and the masks as `MedCLIPSeg/data/ETIS/masks/p1.tif ... p196.tif`.
The evaluation checks every file against `MedCLIPSeg/data/ETIS/manifest.csv` (SHA-256 per file) before it runs;
`download_verification.json` records the archive hash.

Every runner seals the image and mask files it reads; a dataset that differs from the one used in the paper is
reported as a changed input.

## 4. Trained weights

Place the released checkpoints where the locked plan of a suite expects them; the plan lists each checkpoint with its
SHA-256 digest (`MedCLIPSeg/outputs_research/<suite>/`). See the release notes of the weights for the folder layout.
