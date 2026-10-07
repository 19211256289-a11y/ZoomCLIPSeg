"""Window rule and second-pass input of the probe variants. numpy/OpenCV only, so it can be tested without a GPU."""
import cv2

from research_zoom.geometry import window_rule, crop

INPUT = 224


def zoom_constants(variant, accepted):
    """The accepted constants with the three that a variant may change."""
    return dict(accepted, margin=variant['margin'], min_window=variant['min_window'],
                zoom_in_if_window_at_most=variant['trigger'])


def window_for(fraction, box, variant, accepted):
    """(kind, side) of the accepted zoom-in rule with the constants of this variant; never zooms out."""
    return window_rule(fraction, box, variant['tau'], zoom_constants(variant, accepted), False)


def second_input(raw, kind, side, box, variant):
    """Returns (window pixels, rectangle in the coordinates of the output map, shape of the output map)."""
    if kind != 'in':
        raise ValueError(kind)
    H, W = raw.shape[:2]
    if variant['pixels'] == 'original':
        window, rect = crop(raw, kind, side, box)
        return window, rect, (H, W)
    if variant['pixels'] != 'input224':
        raise ValueError(variant['pixels'])
    small = cv2.resize(raw, (INPUT, INPUT))      # the same call as the first-pass preprocessing
    window, rect = crop(small, kind, side, box)
    if variant['output'] == 'grid224':
        return window, rect, (INPUT, INPUT)
    if variant['output'] != 'original':
        raise ValueError(variant['output'])
    x0, y0, w, h = rect
    X0, Y0 = int(round(x0*W/INPUT)), int(round(y0*H/INPUT))
    X1, Y1 = int(round((x0+w)*W/INPUT)), int(round((y0+h)*H/INPUT))
    X1, Y1 = min(W, max(X1, X0+1)), min(H, max(Y1, Y0+1))
    return window, (X0, Y0, X1-X0, Y1-Y0), (H, W)
