"""Figures with real images. Run from the paper folder: python scripts/make_image_figures.py
Cases are chosen by fixed rules from the per-image records (data/figure_cases.json), not by visual inspection."""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox

sys.path.insert(0, 'scripts')
from style import *

GT_C, PR_C, WIN_C = '#35d0ff', '#ff8a3d', '#ffe14d'   # ground truth outline, prediction, zoom window / magnified region
NAMES = {'Kvasir': 'Kvasir-SEG', 'ColonDB': 'CVC-ColonDB', 'ClinicDB': 'CVC-ClinicDB', 'CVC300': 'CVC-300', 'BKAI': 'BKAI'}
ID_C, OOD_C = DETAIL, '#c0392b'
HALO = [pe.Stroke(linewidth=1.9, foreground='#1a1a19', alpha=0.5), pe.Normal()]   # dark halo under bright outlines
cases = json.load(open('data/figure_cases.json'))


def load(cid, kind):
    path = f'data/images/{cid}_{kind}' + ('.jpg' if kind == 'img' else '.png')
    if not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR if kind == 'img' else cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if kind == 'img' else image > 127


def square(image, gt, side=None, centre=None):
    """Square view: the whole image padded to a square, or a square crop around the lesion."""
    h, w = image.shape[:2]
    if side is None:
        s = max(h, w)
        y0, x0 = (h - s)//2, (w - s)//2
    else:
        s = int(side)
        cy, cx = centre
        y0, x0 = int(cy - s/2), int(cx - s/2)
    return x0, y0, s


def label(fig, ax, text, keep):
    """Annotation on an image: white bold text on a dark translucent backdrop, placed in the first corner (top-left,
    top-right, bottom-left, bottom-right) in which it does not cover the outlines (keep: mask of pixels to leave free)."""
    corners = [(0.03, 0.97, 'left', 'top'), (0.97, 0.97, 'right', 'top'), (0.03, 0.03, 'left', 'bottom'), (0.97, 0.03, 'right', 'bottom')]
    t = ax.text(0, 0, text, transform=ax.transAxes, color='white', fontsize=6.5, fontweight='bold', zorder=10,
                bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.5', fc=(0.06, 0.06, 0.06, 0.6), ec='none'))
    renderer = fig.canvas.get_renderer()
    pad = 0.28*6.5*fig.dpi/72 + 2
    best = None
    for x, y, ha, va in corners:
        t.set_position((x, y))
        t.set_ha(ha)
        t.set_va(va)
        bb = t.get_window_extent(renderer)
        bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
        (ax0, ay0), (ax1, ay1) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        xs, xe = sorted((ax0, ax1))
        ys, ye = sorted((ay0, ay1))
        region = keep[max(int(ys), 0):max(int(ye) + 1, 0), max(int(xs), 0):max(int(xe) + 1, 0)]
        overlap = int(region.sum())
        if overlap == 0:
            return
        if best is None or overlap < best[0]:
            best = (overlap, (x, y, ha, va))
    x, y, ha, va = best[1]
    t.set_position((x, y))
    t.set_ha(ha)
    t.set_va(va)


def show(fig, ax, image, gt, pred=None, view=None, text=None, fill=True, avoid=None):
    """One image cell; avoid: optional mask of further pixels the label must not cover (e.g. the region box)."""
    h, w = image.shape[:2]
    x0, y0, s = view or square(image, gt)
    ax.set_facecolor('black')
    ax.imshow(image, extent=(0, w, h, 0), interpolation='lanczos')
    keep = np.zeros((h, w), bool) if avoid is None else avoid.copy()
    if pred is not None and pred.any():
        if fill:
            overlay = np.zeros((h, w, 4))
            overlay[pred] = (*matplotlib.colors.to_rgb(PR_C), 0.38)
            ax.imshow(overlay, extent=(0, w, h, 0), interpolation='nearest')
        ax.contour(np.arange(w) + .5, np.arange(h) + .5, pred.astype(float), levels=[.5], colors=[PR_C], linewidths=0.9).set_path_effects(HALO)
        keep |= pred
    if gt is not None and gt.any():
        ax.contour(np.arange(w) + .5, np.arange(h) + .5, gt.astype(float), levels=[.5], colors=[GT_C], linewidths=1.0).set_path_effects(HALO)
        keep |= gt
    ax.set_xlim(x0, x0 + s)
    ax.set_ylim(y0 + s, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if text:
        k = max(3, int(0.03*s))
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((k, k), np.uint8)) > 0   # margin around the outlines
        label(fig, ax, text, keep)


def region_box(ax, x0, y0, s):
    """Dashed box marking the magnified region, in the window colour with a dark halo."""
    ax.add_patch(matplotlib.patches.Rectangle((x0, y0), s, s, fill=False, ec=WIN_C, lw=0.7, ls=(0, (2.5, 1.5)), path_effects=HALO, zorder=6))


def border_mask(shape, x0, y0, s, thickness=0.012):
    """Pixels of the outline of a square box (x0, y0, s), used to keep the labels off the region box."""
    h, w = shape
    t = max(2, int(thickness*max(h, w)))
    mask = np.zeros((h, w), bool)
    mask[max(y0 - t, 0):y0 + s + t, max(x0 - t, 0):x0 + s + t] = True
    mask[max(y0 + t, 0):max(y0 + s - t, 0), max(x0 + t, 0):max(x0 + s - t, 0)] = False
    return mask


def lesion_view(image, gt, pred, pad=3.2, minimum=0.22):
    """Square crop that contains the lesion and the prediction, for the magnified row."""
    h, w = image.shape[:2]
    both = gt | (pred if pred is not None else False)
    ys, xs = np.nonzero(both)
    side = max(pad*max(np.ptp(np.nonzero(gt)[0]) + 1, np.ptp(np.nonzero(gt)[1]) + 1), np.ptp(ys) + 1.15, np.ptp(xs) + 1.15, minimum*max(h, w))
    side = int(min(side, h, w))
    x0 = int(min(max(xs.mean() - side/2, 0), w - side))
    y0 = int(min(max(ys.mean() - side/2, 0), h - side))
    return x0, y0, side


def legend(fig, y, items=None, size=7):
    """Centred legend row (figure coordinates): outline swatches for ground truth, prediction and, optionally, the region box."""
    items = items or [('line', GT_C, 'ground truth'), ('fill', PR_C, 'prediction')]
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    width, height, pad, gap = 0.16/fw, 0.11/fh, 0.06/fw, 0.22/fw      # swatch, text offset and item spacing (inches)
    renderer = fig.canvas.get_renderer()
    widths = []
    for _, _, text in items:
        t = fig.text(0, 0, text, fontsize=size)
        widths.append(t.get_window_extent(renderer).width/(fig.dpi*fw))
        t.remove()
    x = 0.5 - (sum(widths) + len(items)*(width + pad) + gap*(len(items) - 1))/2
    for (kind, color, text), tw in zip(items, widths):
        if kind == 'fill':
            fig.patches.append(matplotlib.patches.Rectangle((x, y), width, height, transform=fig.transFigure, fc=(*matplotlib.colors.to_rgb(color), 0.38), ec=color, lw=1.0))
        elif kind == 'line':
            fig.patches.append(matplotlib.patches.Rectangle((x, y), width, height, transform=fig.transFigure, fc='none', ec=color, lw=1.0))
        else:
            fig.patches.append(matplotlib.patches.Rectangle((x, y), width, height, transform=fig.transFigure, fc='none', ec=color, lw=0.9, ls=(0, (2.5, 1.5)), path_effects=HALO))
        fig.text(x + width + pad, y + height/2, text, fontsize=size, va='center')
        x += width + pad + tw + gap


def grid(rows, cols, top, bottom, left=0.34, gap=0.025):
    """Figure with exactly square cells; margins in inches."""
    cell = (TWO_COL - left - gap*(cols - 1))/cols
    height = top + bottom + rows*cell + gap*(rows - 1)
    fig = plt.figure(figsize=(TWO_COL, height))
    axes = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            axes[i, j] = fig.add_axes([(left + j*(cell + gap))/TWO_COL, (bottom + (rows - 1 - i)*(cell + gap))/height, cell/TWO_COL, cell/height])
    return fig, axes


# ---------------------------------------------------------------- lesion scale and single-pass behaviour
def fig_scale_cases():
    order = ['scale:tiny_overseg', 'scale:tiny_missed', 'scale:typical', 'scale:large_underseg']
    titles = {'scale:tiny_overseg': 'Small lesion,\nover-segmented', 'scale:tiny_missed': 'Small lesion,\nmissed',
              'scale:typical': 'Typical size,\nsegmented well', 'scale:large_underseg': 'Large lesion,\nunder-segmented'}
    cols = [c for key in order for c in cases if c['why'] == key]
    n = len(cols)
    fig, axes = grid(2, n, top=0.36, bottom=0.42)
    for j, c in enumerate(cols):
        image, gt, pred = load(c['id'], 'img'), load(c['id'], 'gt'), load(c['id'], 'detail')
        view = lesion_view(image, gt, pred)
        show(fig, axes[0, j], image, gt, pred, text=f"DSC {100*c['dsc_detail']:.1f}", avoid=border_mask(gt.shape, *view))
        show(fig, axes[1, j], image, gt, pred, view=view, text=f"{c['frac']:.1f}% of image")
        region_box(axes[0, j], *view)
        axes[1, j].set_xlabel(NAMES[c['dataset']], fontsize=7, color=OOD_C, fontweight='bold', labelpad=3)
    start = 0
    for key in order:
        k = sum(c['why'] == key for c in cols)
        if not k:
            continue
        left = axes[0, start].get_position().x0
        right = axes[0, start + k - 1].get_position().x1
        fig.text((left + right)/2, axes[0, 0].get_position().y1 + 0.015, titles[key], ha='center', va='bottom', fontsize=7.5, linespacing=1.1)
        start += k
    axes[0, 0].set_ylabel('Full image', fontsize=7.5)
    axes[1, 0].set_ylabel('Magnified', fontsize=7.5)
    legend(fig, 0.02, [('line', GT_C, 'ground truth'), ('fill', PR_C, 'prediction'), ('dash', WIN_C, 'magnified region')])
    fig.savefig('figures/fig_scale_cases.pdf', bbox_inches=None); fig.savefig('figures/fig_scale_cases.png', dpi=600, bbox_inches=None); plt.close(fig)


# ---------------------------------------------------------------- qualitative comparison
def fig_qualitative():
    cols = [c for t in NAMES for q in ('qual:gain_p75', 'qual:gain_median') for c in cases if c['dataset'] == t and c['why'] == q]
    rows = [('gt', 'Image and\nground truth', None), ('baseline', 'Baseline', 'dsc_base'), ('detail', '+ Detail branch', 'dsc_detail')]
    if all(os.path.exists(f"data/images/{c['id']}_two.png") for c in cols):
        rows.append(('two', 'ZoomCLIPSeg\n(+ SCZ)', 'dsc_two'))
    n = len(cols)
    fig, axes = grid(len(rows), n, top=0.03, bottom=0.42)
    for j, c in enumerate(cols):
        image, gt = load(c['id'], 'img'), load(c['id'], 'gt')
        for i, (kind, label_, key) in enumerate(rows):
            if kind == 'gt':
                show(fig, axes[i, j], image, gt, text=f"{c['frac']:.1f}%")
            else:
                show(fig, axes[i, j], image, gt, load(c['id'], kind), text=f'{100*c[key]:.1f}')
            if j == 0:
                axes[i, j].set_ylabel(label_, fontsize=7, linespacing=1.1)
    for k, t in enumerate(NAMES):
        left, right = axes[-1, 2*k].get_position().x0, axes[-1, 2*k + 1].get_position().x1
        fig.text((left + right)/2, axes[-1, 0].get_position().y0 - 0.02, NAMES[t], ha='center', va='top', fontsize=7.5,
                 fontweight='bold', color=ID_C if t == 'Kvasir' else OOD_C)
    legend(fig, 0.02)
    fig.savefig('figures/fig_qualitative.pdf', bbox_inches=None); fig.savefig('figures/fig_qualitative.png', dpi=600, bbox_inches=None); plt.close(fig)


# ---------------------------------------------------------------- failure cases of SCZ (one column)
def fig_failures():
    spec = [('fail:largest_loss', 'BKAI', 'Largest loss from\nthe second pass'), ('fail:gain_p05', 'BKAI', '5th percentile\nof the SCZ gain'),
            ('scale:tiny_missed', 'ColonDB', 'Small lesion missed:\nno second pass'), ('scale:large_underseg', 'ColonDB', 'Large lesion:\nno second pass')]
    cols = [(next(c for c in cases if c['why'] == why and c['dataset'] == data), title) for why, data, title in spec]
    left, gap, top, bottom = 0.2, 0.045, 0.3, 0.52
    cell = (ONE_COL-left-3*gap)/4
    height = top+bottom+2*cell+gap
    fig = plt.figure(figsize=(ONE_COL, height))
    for j, (c, title) in enumerate(cols):
        image, gt = load(c['id'], 'img'), load(c['id'], 'gt')
        for i, (kind, key) in enumerate((('detail', 'dsc_detail'), ('two', 'dsc_two'))):
            ax = fig.add_axes([(left+j*(cell+gap))/ONE_COL, (bottom+(1-i)*(cell+gap))/height, cell/ONE_COL, cell/height])
            show(fig, ax, image, gt, load(c['id'], kind), text=f'DSC {100*c[key]:.1f}')
            if j == 0:
                ax.set_ylabel(('Single pass', 'With SCZ')[i], fontsize=7, labelpad=2)
            if i == 0:
                ax.set_title(title, fontsize=6.3, linespacing=1.1, pad=3)
            else:
                ax.set_xlabel(f"{NAMES[c['dataset']]}\n{c['frac']:.1f}% of image", fontsize=6.3, color=MUTED, labelpad=2.5, linespacing=1.15)
    legend(fig, 0.035)
    fig.savefig('figures/fig_failures.pdf', bbox_inches=None, pad_inches=0.01); fig.savefig('figures/fig_failures.png', dpi=600, bbox_inches=None, pad_inches=0.01); plt.close(fig)


if __name__ == '__main__':
    fig_scale_cases()
    fig_qualitative()
    fig_failures()
