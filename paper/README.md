# Figures and tables of the paper

The scripts read the accepted numbers in `data/` (exported from the per-image records and the suite reports) and
write vector PDF and 600 dpi PNG figures to `figures/` and LaTeX tables to `tables/`. Run them from this folder:

```bash
python scripts/make_figures.py      # lesion-size, gain and ultrasound seed figures
python scripts/make_tables.py       # most tables
python scripts/controls_table.py    # Table A3
python scripts/iou_table.py         # Table A2
python scripts/gap_analysis.py      # Table 8 (difference to Polyp-PVT by lesion size)
python scripts/pvt_five.py          # Polyp-PVT five-seed contrasts
python scripts/dedup_analysis.py    # gains with every distinct image counted once
python scripts/bca_intervals.py     # bias-corrected and accelerated intervals
python scripts/review_analyses.py   # paired analyses written to data/review/review_results.json
```

`make_method_figure.py` and `make_image_figures.py` draw Figs. 1, 2 and 4 from `data/images/`: for each case listed
in `data/figure_cases.json`, the image (`<id>_img.jpg`), the ground truth and the baseline, detail-branch and
two-pass predictions (`<id>_<kind>.png`), resized to a longest side of 720 px. These files come from the public
datasets and from the masks exported by `python run.py research_zoom_export run`; they are not redistributed, and the
step that assembles them is not part of this release. The cases were selected by the rule stated in each figure
caption from the per-image records, not by inspection.

All figures use `scripts/style.py`: 8 pt sans-serif text, TrueType fonts embedded in the PDF (`pdf.fonttype 42`),
column and text widths of 87 mm and 180 mm.
