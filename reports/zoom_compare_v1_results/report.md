# zoom_compare_v1: SCZ against generic inference alternatives (detail-branch model, five seeds)

Nothing was trained. The locked configuration t15 remains the method whatever these numbers show.

OOD = equal-weight mean of the four OOD polyp sets; runs = full inference runs per image (30 stochastic passes each).

| method | runs per image | Kvasir DSC | OOD DSC | gain over single pass [95% CI], p | t15 minus method [95% CI], p | verdict |
|---|---|---|---|---|---|---|
| single | 1.00 | 89.54 | 77.36 |  | +2.12 [+1.32, +2.91], p=0.0000 |  |
| t15 | 1.75 | 89.92 | 79.48 | +2.12 [+1.30, +2.91], p=0.0000 |  |  |
| hflip | 2.00 | 90.31 | 78.75 | +1.39 [+0.96, +1.86], p=0.0000 | +0.73 [-0.12, +1.54], p=0.0897 | t15_preferable |
| tile2 | 4.00 | 70.10 | 55.77 | -21.59 [-23.48, -19.75], p=0.0000 | +23.71 [+21.85, +25.59], p=0.0000 | t15_preferable |
| ms12 | 5.00 | 88.36 | 77.50 | +0.15 [-0.71, +1.03], p=0.7569 | +1.97 [+1.13, +2.79], p=0.0000 | t15_preferable |
| tile15 | 4.00 | 87.38 | 75.63 | -1.73 [-3.03, -0.44], p=0.0083 | +3.85 [+2.77, +4.94], p=0.0000 | t15_preferable |
| ms123 | 9.00 | 88.97 | 78.26 | +0.91 [-0.09, +1.95], p=0.0872 | +1.21 [+0.38, +2.05], p=0.0041 | t15_preferable |

Per test set, DSC (%):

| method | Kvasir | ColonDB | ClinicDB | CVC300 | BKAI |
|---|---|---|---|---|---|
| single | 89.54 | 70.71 | 83.08 | 79.72 | 75.91 |
| t15 | 89.92 | 71.95 | 84.16 | 82.13 | 79.67 |
| hflip | 90.31 | 71.67 | 83.81 | 81.96 | 77.55 |
| tile2 | 70.10 | 46.61 | 64.34 | 57.55 | 54.57 |
| ms12 | 88.36 | 68.43 | 82.89 | 82.20 | 76.48 |
| tile15 | 87.38 | 65.82 | 80.89 | 78.13 | 77.66 |
| ms123 | 88.97 | 68.99 | 83.43 | 81.70 | 78.93 |

t15 minus method on the OOD mean, per seed (pp):

- hflip: +0.94, +0.34, +0.94, +1.19, +0.24
- tile2: +26.29, +22.94, +26.27, +22.67, +20.36
- ms12: +2.19, +1.52, +1.79, +2.81, +1.55
- tile15: +4.62, +3.52, +3.72, +4.28, +3.11
- ms123: +1.71, +1.25, +0.85, +1.70, +0.56

## Definitions

- **hflip** (2 runs): mean of the single-pass map and the map of the horizontally reflected image (reflected back), on the 224 grid
- **tile2** (4 runs): uniform slicing: the original image is cut into 2 x 2 non-overlapping tiles of half the side; each tile is segmented by the same network and the maps are stitched at the original resolution
- **ms12** (5 runs): multi-scale averaging, two scales; mean of the single-pass map (resized to the original resolution) and the tile2 map
- **tile15** (4 runs): 2 x 2 overlapping tiles of two thirds of the side, anchored at the corners; maps averaged where tiles overlap
- **ms123** (9 runs): multi-scale averaging, three scales; mean of the single-pass map, the tile15 map and the tile2 map

## Disclosures

- The five polyp test sets were used in earlier stages of this project.
- CVC-300 and CVC-ColonDB share 52 exact image hashes.
- Only the detail-branch model is evaluated; the baseline is not.
- ETIS-Larib is not touched.
