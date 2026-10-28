"""Method overview (Figure 2). Run from the paper folder: python scripts/make_method_figure.py

Panel (a) is a schematic of the network. Panel (b) uses a real test image (BKAI, seed 666): the first prediction is the
exported single-pass mask of the detail-branch model and the window is computed from it with the rule of the paper.

The drawing is laid out in inches (one data unit = one inch at the printed width of 7.08 in), so that the point sizes
given to matplotlib are the point sizes on the printed page.
"""
import math
import sys

import cv2
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, 'scripts')
from style import *

NAME = 'ZoomCLIPSeg'
# ---- palette: ink, muted and light lines from style.py; one blue (detail branch) and one orange (SCZ) accent
FROZEN_F, FROZEN_E = '#ecebe7', '#a7a59f'          # frozen modules: light neutral fill
TRAIN_F, TRAIN_E = '#ffffff', MUTED                # modules trained as in MedCLIPSeg: white
DET_F, DET_BG = '#e3eefb', '#f5f8fd'               # 12 % and 4 % tints of DETAIL
ZOOM_F, ZOOM_BG = '#fdeae0', '#fef7f3'             # 12 % and 4 % tints of ZOOM
GROUP_F, GROUP_E = '#f8f7f5', '#d6d4ce'            # backbone group box
GT_C, PR_C, WIN_C = '#35d0ff', '#ff8a3d', '#ffe14d'   # ground truth, prediction, zoom window (bright, for dark images)
LW_BOX, LW_ARROW = 0.7, 0.9
HEAD = '-|>,head_length=0.5,head_width=0.22'
CASE, TAU = 'BKAI_e101f07b34db', 0.15
W, H = TWO_COL, 5.80
F_MOD, F_SUB, F_SMALL, F_PANEL, F_LEG = 7, 6, 5.8, 8.5, 6

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')
ax.set_autoscale_on(False)


# ---------------------------------------------------------------------------------------------- drawing helpers
def text_width(s, size, weight='normal'):
    """Width of a string in inches at the printed size (used to lay out the legends)."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    width = t.get_window_extent(fig.canvas.get_renderer()).width/fig.dpi
    t.remove()
    return width


def box(x, y, w, h, face, edge, lw=LW_BOX, r=0.05, z=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=face, ec=edge, lw=lw, ls=ls, zorder=z))


def module(x, y, w, h, title, sub=None, kind='trained', color=INK):
    """A module box: title (7 pt bold) and an optional second line (6 pt, muted)."""
    face, edge = dict(frozen=(FROZEN_F, FROZEN_E), trained=(TRAIN_F, TRAIN_E), ours=(DET_F, DETAIL))[kind]
    box(x, y, w, h, face, edge)
    cx, cy = x + w/2, y + h/2
    if sub:
        ax.text(cx, cy + 0.058, title, ha='center', va='center', fontsize=F_MOD, fontweight='bold', color=color, zorder=4)
        ax.text(cx, cy - 0.062, sub, ha='center', va='center', fontsize=F_SUB, color=MUTED, zorder=4)
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=F_MOD, fontweight='bold', color=color, zorder=4, linespacing=1.15)
    return dict(l=(x, cy), r=(x + w, cy), t=(cx, y + h), b=(cx, y), cx=cx, cy=cy, x=x, y=y, w=w, h=h)


def slab(cx, cy, s, face, edge, lines=(), n=3, off=0.028, color=INK, z=3):
    """A feature map as a stack of n sheets, centred on (cx, cy) with side s (front sheet); lines of text inside."""
    x, y = cx - s/2, cy - s/2
    for k in reversed(range(n)):
        ax.add_patch(Rectangle((x + k*off, y + k*off), s, s, fc=face, ec=edge, lw=LW_BOX, zorder=z - 0.01*k))
    if len(lines) == 1:
        ax.text(cx, cy, lines[0], ha='center', va='center', fontsize=F_SUB, color=color, zorder=z + 1)
    elif len(lines) == 2:
        ax.text(cx, cy + 0.048, lines[0], ha='center', va='center', fontsize=F_SUB, color=color, zorder=z + 1)
        ax.text(cx, cy - 0.052, lines[1], ha='center', va='center', fontsize=F_SMALL, color=MUTED if color == INK else color, zorder=z + 1, alpha=0.9 if color != INK else 1)
    top = y + s + (n - 1)*off
    return dict(l=(x, cy), r=(x + s + (n - 1)*off, cy), t=(cx, y + s), tt=(cx, top), b=(cx, y), cx=cx, cy=cy, s=s)


def node(cx, cy, symbol, r=0.085, color=INK):
    """Circular operator: '+' sum, 'x' inner product, 'C' concatenation."""
    ax.add_patch(Circle((cx, cy), r, fc='white', ec=color, lw=LW_BOX, zorder=5))
    if symbol == '+':
        ax.plot([cx - r*0.5, cx + r*0.5], [cy, cy], color=color, lw=LW_BOX, zorder=6)
        ax.plot([cx, cx], [cy - r*0.5, cy + r*0.5], color=color, lw=LW_BOX, zorder=6)
    elif symbol == 'x':
        d = r*0.38
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color=color, lw=LW_BOX, zorder=6)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color=color, lw=LW_BOX, zorder=6)
    else:
        ax.text(cx, cy - 0.003, symbol, ha='center', va='center', fontsize=6, color=color, zorder=6, fontweight='bold')
    return dict(l=(cx - r, cy), r=(cx + r, cy), t=(cx, cy + r), b=(cx, cy - r))


def route(points, color=MUTED, lw=LW_ARROW, z=3, ls='-', head=True):
    """Orthogonal polyline with one arrow head on the last segment."""
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_joinstyle='miter', solid_capstyle='butt')
    a, b = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=HEAD if head else '-', mutation_scale=7, lw=lw, color=color, zorder=z, ls=ls,
                                 shrinkA=0, shrinkB=0, joinstyle='miter', capstyle='butt'))


def badge(cx, cy, number, color=ZOOM, r=0.082):
    ax.add_patch(Circle((cx, cy), r, fc=color, ec='none', zorder=8))
    ax.text(cx, cy - 0.004, str(number), ha='center', va='center', fontsize=6.2, color='white', fontweight='bold', zorder=9)


def panel_title(x, y, letter, title):
    ax.text(x, y, f'({letter})', fontweight='bold', fontsize=F_PANEL, va='center', ha='left')
    ax.text(x + 0.21, y, title, fontsize=F_PANEL, va='center', ha='left')


def picture(image, x, y, w, h, edge=MUTED, lw=0.6, z=4):
    ax.imshow(image, extent=(x, x + w, y, y + h), zorder=z, interpolation='lanczos', aspect='auto')
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=edge, lw=lw, zorder=z + 2))


def window_rect(x, y, w, h, z=7, lw=1.0):
    """Dashed zoom window with a dark halo, so that it reads on bright and on dark tissue."""
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=WIN_C, lw=lw, ls=(0, (3, 1.6)), zorder=z,
                           path_effects=[pe.Stroke(linewidth=lw + 1.4, foreground='#1a1a19', alpha=0.55), pe.Normal()]))


# ---------------------------------------------------------------------------------------------- real example
raw = cv2.cvtColor(cv2.imread(f'data/images/{CASE}_img.jpg'), cv2.COLOR_BGR2RGB)
gt = cv2.imread(f'data/images/{CASE}_gt.png', cv2.IMREAD_GRAYSCALE) > 127
first = cv2.imread(f'data/images/{CASE}_detail.png', cv2.IMREAD_GRAYSCALE) > 127
final = cv2.imread(f'data/images/{CASE}_two.png', cv2.IMREAD_GRAYSCALE) > 127
IH, IW = gt.shape
ys, xs = np.nonzero(first)
area = float(first.mean())
bbox = (xs.min()/IW, ys.min()/IH, (xs.max() + 1)/IW, (ys.max() + 1)/IH)
longer = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
side = max(math.sqrt(area/TAU), 1.25*longer, 0.2)
assert side <= 0.8
ww, wh = round(side*IW), round(side*IH)
wx = int(min(max(round((bbox[0] + bbox[2])/2*IW - ww/2), 0), IW - ww))
wy = int(min(max(round((bbox[1] + bbox[3])/2*IH - wh/2), 0), IH - wh))
print(f'example {CASE}: lesion {100*gt.mean():.2f}% of image, first prediction a={100*area:.2f}%, b={longer:.3f}, s={side:.3f}, '
      f'lesion share of window {100*gt[wy:wy+wh, wx:wx+ww].mean():.2f}%')


def overlay(x, y, w, h, view=None, prediction=None, truth=True, window=False, edge=MUTED, lw=0.6):
    """The example image (or a window of it) with outlines; view = (x0, y0, w, h) in pixels."""
    vx, vy, vw, vh = view or (0, 0, IW, IH)
    picture(raw[vy:vy+vh, vx:vx+vw], x, y, w, h, edge=edge, lw=lw)
    sx, sy = w/vw, h/vh
    gx = x + (np.arange(vw) + .5)*sx
    gy = y + h - (np.arange(vh) + .5)*sy
    if prediction is not None:
        part = prediction[vy:vy+vh, vx:vx+vw]
        rgba = np.zeros((vh, vw, 4))
        rgba[part] = (*matplotlib.colors.to_rgb(PR_C), 0.38)
        ax.imshow(rgba, extent=(x, x + w, y, y + h), zorder=5, interpolation='nearest', aspect='auto')
        ax.contour(gx, gy, part.astype(float), levels=[.5], colors=[PR_C], linewidths=0.8, zorder=5.5)
    if truth:
        ax.contour(gx, gy, gt[vy:vy+vh, vx:vx+vw].astype(float), levels=[.5], colors=[GT_C], linewidths=0.8, zorder=5.6)
    if window:
        window_rect(x + (wx - vx)*sx, y + h - (wy - vy + wh)*sy, ww*sx, wh*sy)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)


KV = cv2.cvtColor(cv2.imread('data/images/Kvasir_cju8b0jr0r2o_img.jpg'), cv2.COLOR_BGR2RGB)
KV_MASK = cv2.imread('data/images/Kvasir_cju8b0jr0r2o_detail.png', cv2.IMREAD_GRAYSCALE) > 127


def thumb(x, y, s, edge=MUTED, mask=False, lw=0.6):
    """Input illustration: a Kvasir-SEG test image, centre-cropped to a square (with its prediction if mask)."""
    h, w = KV.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m)//2, (w - m)//2
    picture(KV[y0:y0+m, x0:x0+m], x, y, s, s, edge=edge, lw=lw)
    if mask:
        part = KV_MASK[y0:y0+m, x0:x0+m]
        rgba = np.zeros((m, m, 4))
        rgba[part] = (*matplotlib.colors.to_rgb(PR_C), 0.38)
        ax.imshow(rgba, extent=(x, x + s, y, y + s), zorder=5, interpolation='nearest', aspect='auto')
        ax.contour(x + (np.arange(m) + .5)*s/m, y + s - (np.arange(m) + .5)*s/m, part.astype(float), levels=[.5], colors=[PR_C], linewidths=0.8, zorder=5.5)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)


# ============================================================================================== (a) network
# vertical grid (inches from the bottom of the drawing)
Y_TITLE_A = H - 0.12
YT, YI = 5.00, 4.14                # text row and image row of the backbone
MH = 0.38                          # module height
YM = (YT + YI)/2                   # middle row: inner product and z_base
BB_TOP, BB_BOT = YT + MH/2 + 0.30, YI - MH/2 - 0.16    # backbone group box
GAP_Y = (BB_BOT + 3.66)/2          # the horizontal segment of the F route and the fusion node run here
DB_TOP, DB_BOT = 3.66, 2.55        # detail-branch group box
YD = 3.09                          # detail feature-map row
X0, X1 = 1.12, 5.76                # group boxes
XF = 6.10                          # fusion column

panel_title(0.04, Y_TITLE_A, 'a', f'{NAME} network')

# group boxes
box(X0, BB_BOT, X1 - X0, BB_TOP - BB_BOT, GROUP_F, GROUP_E, r=0.08, z=0.5)
ax.text(X0 + 0.10, BB_TOP - 0.07, 'MedCLIPSeg backbone', fontsize=6.5, color=MUTED, fontweight='bold', va='top', ha='left')
box(X0, DB_BOT, X1 - X0, DB_TOP - DB_BOT, DET_BG, DETAIL, r=0.08, z=0.5)
ax.text(X0 + 0.10, DB_TOP - 0.07, 'Detail branch (ours)', fontsize=6.5, color=DETAIL, fontweight='bold', va='top', ha='left')

# inputs
box(0.10, YT - 0.19, 0.88, 0.38, '#ffffff', MUTED, r=0.10)
ax.text(0.54, YT + 0.07, '"a polyp in a', ha='center', va='center', fontsize=F_SUB, style='italic', zorder=4)
ax.text(0.54, YT - 0.06, 'colonoscopy image"', ha='center', va='center', fontsize=F_SUB, style='italic', zorder=4)
ax.text(0.54, YT - 0.26, 'class-level sentence', ha='center', va='top', fontsize=F_SUB, color=MUTED)
T224 = 0.50
thumb(0.54 - T224/2, YI - T224/2, T224)
ax.text(0.54, YI - T224/2 - 0.06, 'image, 224 × 224', ha='center', va='top', fontsize=F_SUB, color=MUTED)
T448 = 0.70
thumb(0.54 - T448/2, YD - T448/2, T448, edge=DETAIL, lw=0.8)
ax.text(0.54, YD - T448/2 - 0.06, 'same image, 448 × 448', ha='center', va='top', fontsize=F_SUB, color=DETAIL)

# backbone modules (left to right): encoders, adapters, head / decoder, features, inner product, logits
EX, EW = 1.26, 0.92
AX_, AW = 2.37, 0.68
HX, HW = 3.24, 0.70
text_enc = module(EX, YT - MH/2, EW, MH, 'Text encoder', 'BiomedBERT', 'frozen')
img_enc = module(EX, YI - MH/2, EW, MH, 'Image encoder', 'UniMed-CLIP ViT-B/16', 'frozen')
adapters = module(AX_, YI - MH/2, AW, YT - YI + MH, 'Cross-modal\nadapters', None, 'trained')
head = module(HX, YT - MH/2, HW, MH, 'Mask head', 'query $q$', 'trained')
dec = module(HX, YI - MH/2, HW, MH, 'Upscaling', 'decoder', 'trained')
SF = 0.26
feat = slab(4.26, YI, SF, '#ffffff', TRAIN_E, lines=('$F$', '56²'))
XPROD = 4.74
prod = node(XPROD, YM, 'x')
SZ = 0.44
zbase = slab(5.32, YM, SZ, '#ffffff', TRAIN_E, lines=('$z_{\\mathrm{base}}$', '224²'))

route([(0.98, YT), text_enc['l']])
route([(0.54 + T224/2, YI), img_enc['l']])
route([text_enc['r'], (AX_, YT)])
route([img_enc['r'], (AX_, YI)])
route([(AX_ + AW, YT), head['l']])
route([(AX_ + AW, YI), dec['l']])
route([dec['r'], feat['l']])
route([head['r'], (XPROD, YT), prod['t']])
route([feat['r'], (XPROD, YI), prod['b']])
route([prod['r'], zbase['l']])

# detail branch: three encoder stages, concatenation with F, decoder with skip connections, 1x1 fusion
S1, S2, S3 = 0.44, 0.34, 0.26
e1 = slab(1.26 + S1/2, YD, S1, DET_F, DETAIL, lines=('224²', '16 ch'), color=DETAIL)
e2 = slab(2.02 + S2/2, YD, S2, DET_F, DETAIL, lines=('112²', '32 ch'), color=DETAIL)
e3 = slab(2.66 + S3/2, YD, S3, DET_F, DETAIL, lines=('56²', '64 ch'), color=DETAIL)
XC = 3.30
cat = node(XC, YD, 'C', color=DETAIL)
d1 = slab(3.64 + S3/2, YD, S3, DET_F, DETAIL, lines=('56²',), color=DETAIL)
d2 = slab(4.20 + S2/2, YD, S2, DET_F, DETAIL, lines=('112²',), color=DETAIL)
d3 = slab(4.84 + S1/2, YD, S1, DET_F, DETAIL, lines=('224²',), color=DETAIL)
conv = module(5.48, YD - 0.10, 0.20, 0.20, '1×1', None, 'ours', color=DETAIL)
route([(0.54 + T448/2, YD), e1['l']], color=DETAIL)
route([e1['r'], e2['l']], color=DETAIL)
route([e2['r'], e3['l']], color=DETAIL)
route([e3['r'], cat['l']], color=DETAIL)
route([cat['r'], d1['l']], color=DETAIL)
route([d1['r'], d2['l']], color=DETAIL)
route([d2['r'], d3['l']], color=DETAIL)
route([d3['r'], conv['l']], color=DETAIL)
ax.text(5.58, YD - 0.14, 'zero-init.', ha='center', va='top', fontsize=F_SMALL, color=DETAIL)
ax.text((e2['cx'] + e3['cx'])/2 + 0.06, DB_TOP - 0.07, 'encoder', ha='center', va='top', fontsize=F_SUB, color=DETAIL, alpha=0.85)
ax.text((d1['cx'] + d2['cx'])/2 + 0.14, DB_TOP - 0.07, 'decoder', ha='center', va='top', fontsize=F_SUB, color=DETAIL, alpha=0.85)
# decoder features of the backbone enter the detail decoder (routed through the gap between the two groups)
route([feat['b'], (feat['cx'], GAP_Y), (XC, GAP_Y), cat['t']], color=DETAIL)
# skip connections, below the feature maps (dashed shaft, solid head)
for a, b, level in ((e2, d2, YD - S2/2 - 0.14), (e1, d3, YD - S1/2 - 0.20)):
    ax.plot([a['cx'], a['cx'], b['cx'], b['cx']], [a['b'][1], level, level, b['b'][1] - 0.05], color=DETAIL, lw=0.7, ls=(0, (2.4, 1.6)), zorder=2.5, solid_joinstyle='miter')
    ax.add_patch(FancyArrowPatch((b['cx'], b['b'][1] - 0.05), b['b'], arrowstyle=HEAD, mutation_scale=7, lw=0.7, color=DETAIL, zorder=2.5, shrinkA=0, shrinkB=0))

# fusion and output
plus = node(XF, GAP_Y, '+')
route([zbase['r'], (XF, YM), plus['t']])
route([conv['r'], (XF, YD), plus['b']], color=DETAIL)
ax.text(XF - 0.05, (YD + GAP_Y)/2 + 0.02, '$\\Delta z$', fontsize=F_MOD, color=DETAIL, ha='right', va='center')
ax.text(XF + 0.07, YM - 0.24, '$z=z_{\\mathrm{base}}+\\Delta z$', fontsize=6.5, color=INK, ha='left', va='center')
TP = 0.56
PX = XF + 0.26
thumb(PX, GAP_Y - TP/2, TP, mask=True)
route([plus['r'], (PX, GAP_Y)])
ax.text(PX + TP/2, GAP_Y - TP/2 - 0.06, 'prediction', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(PX + TP/2, GAP_Y - TP/2 - 0.17, 'sigmoid, mean of\n30 stochastic passes', ha='center', va='top', fontsize=F_SMALL, color=MUTED, linespacing=1.1)

# legend of panel (a): swatches of the three module kinds, the feature-map symbol and the two operators
LY = DB_BOT - 0.18
items = [('frozen', FROZEN_F, FROZEN_E), ('trained', TRAIN_F, TRAIN_E), (f'added by {NAME} (trained)', DET_F, DETAIL),
         ('feature map (side length)', None, None), ('inner product', 'x', None), ('concatenation', 'C', None)]
widths = [text_width(label, F_LEG) for label, *_ in items]
step = 0.24
x = W/2 - (sum(widths) + 0.28*len(items) + step*(len(items) - 1))/2
for (label, face, edge), width in zip(items, widths):
    if edge is not None:
        ax.add_patch(FancyBboxPatch((x, LY - 0.06), 0.22, 0.12, boxstyle='round,pad=0,rounding_size=0.02', fc=face, ec=edge, lw=LW_BOX))
    elif face is None:
        for k in (1, 0):
            ax.add_patch(Rectangle((x + 0.03 + k*0.03, LY - 0.06 + k*0.03), 0.13, 0.13, fc='white', ec=TRAIN_E, lw=LW_BOX))
    else:
        node(x + 0.11, LY, face, r=0.07)
    ax.text(x + 0.28, LY, label, fontsize=F_LEG, va='center')
    x += 0.28 + width + step

# ============================================================================================== (b) SCZ
Y_TITLE_B = 2.00
IMW = 1.04
IMH = IMW*IH/IW
TY = 0.84                           # bottom of the image row
MY = TY + IMH/2
BY = TY + IMH + 0.13                # badges
panel_title(0.04, Y_TITLE_B, 'b', 'Scale-canonical zoom (SCZ) at inference')

X1_, XN1, X2_, XR, X3_, XN2, X4_ = 0.06, 1.19, 1.68, 2.84, 4.28, 5.41, 5.90
NW, NH, RW = 0.40, 0.30, 1.32

overlay(X1_, TY, IMW, IMH, truth=False)
ax.text(X1_ + IMW/2, TY - 0.07, 'input image', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(X1_ + IMW/2, TY - 0.19, f'lesion: {100*gt.mean():.1f} % of the image', ha='center', va='top', fontsize=F_SMALL, color=MUTED)

net1 = module(XN1, MY - NH/2, NW, NH, 'network', '(a)', 'frozen')
route([(X1_ + IMW, MY), net1['l']])
route([net1['r'], (X2_, MY)])
badge(XN1 + NW/2, BY, 1)
ax.text(XN1 + NW/2, TY - 0.07, 'first pass', ha='center', va='top', fontsize=6.2, fontweight='bold')

overlay(X2_, TY, IMW, IMH, prediction=first, window=True)
ax.text(X2_ + IMW/2, TY - 0.07, 'first prediction', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(X2_ + IMW/2, TY - 0.19, f'a = {100*area:.1f} % of the image, b = {longer:.2f}', ha='center', va='top', fontsize=F_SMALL, color=MUTED)

box(XR, TY, RW, IMH, ZOOM_F, ZOOM, r=0.06)
ax.text(XR + RW/2, TY + IMH - 0.13, 'window rule', ha='center', va='center', fontsize=6.5, fontweight='bold', zorder=4)
ax.text(XR + RW/2, MY + 0.02, '$s=\\max\\left(\\sqrt{a/\\tau},\\ 1.25\\,b,\\ 0.2\\right)$', ha='center', va='center', fontsize=6.3, zorder=4)
ax.text(XR + RW/2, TY + 0.15, 'zoom in only if $s\\leq 0.8$', ha='center', va='center', fontsize=F_SUB, zorder=4)
ax.text(XR + RW/2, TY - 0.07, f'here $s$ = {side:.2f}', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(XR + RW/2, TY - 0.19, '$\\tau$ = 0.15 (target lesion share)', ha='center', va='top', fontsize=F_SMALL, color=MUTED)
badge(XR + RW/2, BY, 2)
route([(X2_ + IMW, MY), (XR, MY)])
route([(XR + RW, MY), (X3_, MY)])

overlay(X3_, TY, IMW, IMH, view=(wx, wy, ww, wh), edge='none')
window_rect(X3_, TY, IMW, IMH, z=7, lw=1.0)
ax.text(X3_ + IMW/2, TY - 0.07, 'magnified window', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(X3_ + IMW/2, TY - 0.19, f'{1/side:.1f}× magnification', ha='center', va='top', fontsize=F_SMALL, color=MUTED)

net2 = module(XN2, MY - NH/2, NW, NH, 'network', '(a)', 'frozen')
route([(X3_ + IMW, MY), net2['l']])
badge(XN2 + NW/2, BY, 3)
ax.text(XN2 + NW/2, TY - 0.07, 'second pass', ha='center', va='top', fontsize=6.2, fontweight='bold')

overlay(X4_, TY, IMW, IMH, prediction=final, window=True)
route([net2['r'], (X4_, MY)])
badge(X4_ + IMW/2, BY, 4)
ax.text(X4_ + IMW/2, TY - 0.07, 'final prediction', ha='center', va='top', fontsize=6.2, fontweight='bold')
ax.text(X4_ + IMW/2, TY - 0.19, 'DSC 18.8 % → 72.7 %', ha='center', va='top', fontsize=F_SMALL, color=MUTED)

# legend of panel (b), centred under the pipeline
LY = TY - 0.46
labels = ['ground truth', 'prediction', 'zoom window']
widths = [text_width(s, F_LEG) for s in labels]
x = W/2 - (sum(widths) + 3*0.28 + 2*0.24)/2
ax.plot([x, x + 0.22], [LY, LY], color=GT_C, lw=1.2)
ax.text(x + 0.28, LY, labels[0], fontsize=F_LEG, va='center')
x += 0.28 + widths[0] + 0.24
ax.add_patch(Rectangle((x, LY - 0.055), 0.22, 0.11, fc=(*matplotlib.colors.to_rgb(PR_C), 0.38), ec=PR_C, lw=0.8))
ax.text(x + 0.28, LY, labels[1], fontsize=F_LEG, va='center')
x += 0.28 + widths[1] + 0.24
window_rect(x, LY - 0.055, 0.22, 0.11, z=7, lw=1.0)
ax.text(x + 0.28, LY, labels[2], fontsize=F_LEG, va='center')

ax.set_xlim(0, W)
ax.set_ylim(LY - 0.10, H)
save(fig, 'fig_method')
