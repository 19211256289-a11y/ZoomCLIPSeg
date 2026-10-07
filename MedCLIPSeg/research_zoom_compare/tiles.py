"""Tile layouts and stitching. numpy/OpenCV only, so it can be tested without a GPU."""
import cv2
import numpy as np


def tile_rects(H, W, layout):
    """Rectangles (x0, y0, w, h) of a 2 x 2 layout: 'tile2' non-overlapping halves, 'tile15' corner tiles of two thirds."""
    if layout == 'tile2':
        w, h = W//2, H//2
        return [(0, 0, w, h), (w, 0, W-w, h), (0, h, w, H-h), (w, h, W-w, H-h)]
    if layout == 'tile15':
        w, h = int(round(2*W/3)), int(round(2*H/3))
        return [(0, 0, w, h), (W-w, 0, w, h), (0, H-h, w, h), (W-w, H-h, w, h)]
    raise ValueError(layout)


def stitch(maps, rects, shape):
    """Mean of the tile maps at the original resolution; every pixel must be covered."""
    H, W = shape
    total, count = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for tile, (x0, y0, w, h) in zip(maps, rects):
        total[y0:y0+h, x0:x0+w] += cv2.resize(tile.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        count[y0:y0+h, x0:x0+w] += 1
    if not (count > 0).all():
        raise RuntimeError('Tiles do not cover the image')
    return total/count


def full_resolution(probability, shape):
    return cv2.resize(probability.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
