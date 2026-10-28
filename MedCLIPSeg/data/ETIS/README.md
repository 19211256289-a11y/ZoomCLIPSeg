# ETIS-LaribPolypDB — reserved external test cohort

Downloaded from the original archive linked by the [official Grand Challenge dataset page](https://polyp.grand-challenge.org/ETISLarib/).

## Files

- `images/`: 196 original TIFF images, `1.tif` through `196.tif`.
- `masks/`: 196 original TIFF masks, `p1.tif` through `p196.tif`.
- `manifest.csv`: explicit image/mask pairing, dimensions and SHA-256 hashes.
- `download_verification.json`: source, download time, archive hash and integrity checks.
- Original archive retained at `../../../downloads/ETIS-LaribPolypDB.rar` relative to this directory.

All 392 files were decoded successfully and every image/mask pair has matching dimensions. Original pixels, encoding and filenames are preserved. The archive contains TIFF files, despite the older official webpage describing PNG/BMP paths. Only the archive directory layout was reorganized.

## Research boundary

This preparation performed download and file-integrity checks only. No model predictions, performance metrics, visual case selection, foreground-size analysis, training or hyperparameter selection were performed. No training or evaluation configuration was modified. Downloading the dataset does not by itself lock an evaluation protocol. Before its first model evaluation, freeze the compared methods, checkpoint hashes, seeds, preprocessing, SCZ settings, metrics and statistical analysis. This cohort is intended for the user's once-only locked external evaluation, not method development.

The original source archive cannot independently establish patient-level independence or absence from foundation-model pretraining data.

## Attribution and use

The official page restricts use to research and education and prohibits commercial use. Original images belong to Lariboisiere Hospital-APHP and ETIS laboratory; ground truth belongs to ETIS laboratory. Consult the linked official page for authoritative terms.

Required reference stated on that page:
Juan S. Silva, Aymeric Histace, Olivier Romain, Xavier Dray, Bertrand Granado. *Towards embedded detection of polyps in WCE images for early diagnosis of colorectal cancer*. International Journal of Computer Assisted Radiology and Surgery, 2014, 9(2), 283–293.
