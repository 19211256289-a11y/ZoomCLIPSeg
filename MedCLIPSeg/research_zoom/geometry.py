"""Window rule, crops and fusion. numpy/OpenCV only, so it can be tested without a GPU."""
import math

import cv2
import numpy as np


def region(mask):
    """Foreground fraction and normalised bounding box (x0, y0, x1, y1) of a boolean mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0., None
    h, w = mask.shape
    return float(mask.mean()), (xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h)


def window_rule(fraction, box, target, zoom, allow_out):
    """Returns (kind, side): kind is 'in', 'out' or 'none'; side is the window side as a fraction of the image."""
    if box is None:
        return 'none', 1.
    side = max(math.sqrt(fraction/target), zoom['margin']*max(box[2]-box[0], box[3]-box[1]), zoom['min_window'])
    if side <= zoom['zoom_in_if_window_at_most']:
        return 'in', side
    if allow_out and fraction >= zoom['zoom_out_if_fraction_at_least']:
        side = min(math.sqrt(fraction/zoom['zoom_out_target_fraction']), zoom['max_window'])
        if side > 1:
            return 'out', side
    return 'none', 1.


def crop(raw, kind, side, box):
    """Returns the second-pass input (original pixels) and its rectangle (x0, y0, w, h)."""
    H, W = raw.shape[:2]
    if kind == 'in':
        w, h = min(W, max(2, round(side*W))), min(H, max(2, round(side*H)))
        cx, cy = (box[0]+box[2])/2*W, (box[1]+box[3])/2*H
        x0 = int(min(max(round(cx-w/2), 0), W-w))
        y0 = int(min(max(round(cy-h/2), 0), H-h))
        return raw[y0:y0+h, x0:x0+w], (x0, y0, w, h)
    if kind != 'out':
        raise ValueError(kind)
    w, h = max(W, round(side*W)), max(H, round(side*H))
    x0, y0 = (w-W)//2, (h-H)//2
    canvas = np.zeros((h, w, 3), raw.dtype)
    canvas[y0:y0+H, x0:x0+W] = raw
    return canvas, (x0, y0, w, h)


def compose(second, kind, rect, shape):
    """Final probability map at the image resolution, or None when the second pass found nothing."""
    H, W = shape
    x0, y0, w, h = rect
    patch = cv2.resize(second.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == 'in':
        full = np.zeros((H, W), np.float32)
        full[y0:y0+h, x0:x0+w] = patch
    else:
        full = np.ascontiguousarray(patch[y0:y0+H, x0:x0+W])
    return full if (full > .5).any() else None


def rescale(raw, mask, scale):
    """Synthetic scale shift of one image and its binary (0/255) mask."""
    H, W = raw.shape[:2]
    if scale < 1:
        w, h = max(1, round(W*scale)), max(1, round(H*scale))
        x0, y0 = (W-w)//2, (H-h)//2
        image, label = np.zeros_like(raw), np.zeros((H, W), np.uint8)
        image[y0:y0+h, x0:x0+w] = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
        label[y0:y0+h, x0:x0+w] = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8)*255
        return image, label
    w, h = max(2, round(W/scale)), max(2, round(H/scale))
    _, box = region(mask > 127)
    cx, cy = ((box[0]+box[2])/2*W, (box[1]+box[3])/2*H) if box else (W/2, H/2)
    x0 = int(min(max(round(cx-w/2), 0), W-w))
    y0 = int(min(max(round(cy-h/2), 0), H-h))
    return np.ascontiguousarray(raw[y0:y0+h, x0:x0+w]), np.ascontiguousarray(mask[y0:y0+h, x0:x0+w])
