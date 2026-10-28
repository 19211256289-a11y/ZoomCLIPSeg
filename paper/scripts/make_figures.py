"""Result figures from the accepted per-image records. Run from the paper folder: python scripts/make_figures.py"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, 'scripts')
from style import *

SETS = ['Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI']
LABEL = {'Kvasir': 'Kvasir-SEG\n(in-domain)', 'ColonDB': 'CVC-ColonDB', 'ClinicDB': 'CVC-ClinicDB', 'CVC300': 'CVC-300', 'BKAI': 'BKAI'}
OOD = SETS[1:]
four = json.load(open('data/test_four_arms_results.json'))['results']
zoom = json.load(open('data/test_zoom_results.json'))
zrows = list(csv.DictReader(open('data/test_zoom_cases.csv')))
rng = np.random.default_rng(20261005)


def ci(values, draws=10000):
    values = np.asarray(values)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return np.quantile(boot, .025), np.quantile(boot, .975)


# ---------------------------------------------------------------- Fig: paired gains with intervals
def fig_gains():
    # drawn at the width at which it is included (0.8 of the text width), so that the 8 pt type prints at 8 pt
    fig, axes = plt.subplots(1, 2, figsize=(0.8*TWO_COL, 2.0), sharey=True, sharex=True, layout='constrained')
    specs = [('a', 'Detail branch vs. baseline', DETAIL,
              {t: four[t]['contrasts']['detail - baseline']['DSC']['all'] for t in SETS}),
             ('b', 'SCZ vs. single-pass inference', ZOOM,
              {t: zoom['targets'][t]['contrast']['DSC']['all'] for t in SETS})]
    for ax, (letter, title, color, data) in zip(axes, specs):
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for i, t in enumerate(SETS):
            y = len(SETS) - 1 - i + (0.0 if i else 0.0)
            d = data[t]
            lo, hi = d['bootstrap_ci95_pp']
            ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle='round', zorder=3)
            ax.scatter(d['per_seed_mean_pp'], [y - 0.26] * len(d['per_seed_mean_pp']), s=9, marker='|', color=MUTED, lw=0.9, zorder=3)
            ax.scatter([d['mean_pp']], [y], s=30, color=color, edgecolor='white', lw=0.8, zorder=4)
            ax.text(7.55, y, f"{d['mean_pp']:+.2f}".replace('-', '\u2212'), va='center', ha='right', fontsize=7.5, color=INK)
        ax.set_yticks(range(len(SETS)))
        ax.set_yticklabels([LABEL[t] for t in SETS][::-1])
        ax.set_ylim(-0.6, len(SETS) - 0.4)
        ax.set_xlim(-2.1, 7.6)
        ax.set_xticks([-2, 0, 2, 4, 6])
        ax.set_xlabel('Paired DSC difference (percentage points)')
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        panel(ax, letter, title)
    fig.get_layout_engine().set(wspace=0.03, w_pad=0.02, h_pad=0.02)
    save(fig, 'fig_gains', exact=True)


# ---------------------------------------------------------------- Fig: cumulative effect per test set
def fig_progressive():
    arms = lambda t, a: four[t]['arms'][a]['DSC_percent']
    get = lambda v: v['mean'] if isinstance(v, dict) else v
    five = json.load(open('data/main_five_seed.json'))    # five-seed means written by make_tables.py
    cases = list(csv.DictReader(open('data/review/baseline_zoom_cases.csv', encoding='utf-8-sig')))   # baseline with SCZ, five seeds
    bscz = {t: float(np.mean([np.mean([100*float(r['DSC']) for r in cases if r['dataset'] == t and r['config'] == 't15' and r['seed'] == s])
                              for s in ('666', '667', '668', '669', '670')])) for t in SETS}
    rows = []
    for k, t in enumerate(SETS):
        rows.append((LABEL[t], five['baseline'][k], five['detail'][k], five['zoom'][k], bscz[t]))
    macro = tuple(np.mean([r[k] for r in rows[1:]]) for k in (1, 2, 3, 4))
    rows.append(('OOD mean', *macro))
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.45), layout='constrained')
    for i, (name, b, d, z, bz) in enumerate(rows):
        y = len(rows) - 1 - i - (0.35 if i == len(rows) - 1 else 0)
        ax.plot([b, bz], [y+.14, y+.14], color=GRID, lw=1.8, solid_capstyle='round', zorder=1)   # upper track: baseline, without and with SCZ
        ax.plot([d, z], [y-.14, y-.14], color=GRID, lw=1.8, solid_capstyle='round', zorder=1)     # lower track: detail-branch model
        ax.scatter([b], [y+.14], s=20, marker='o', color=BASE, edgecolor='white', lw=0.7, zorder=3)
        ax.scatter([d], [y-.14], s=20, marker='s', color=DETAIL, edgecolor='white', lw=0.7, zorder=3)
        ax.scatter([bz], [y+.14], s=22, marker='^', color=BASE, edgecolor='white', lw=0.7, zorder=3)
        ax.scatter([z], [y-.14], s=24, marker='D', color=ZOOM, edgecolor='white', lw=0.7, zorder=4)
        ax.text(max(b, d, z, bz) + 0.6, y, f'{z-b:+.2f}'.replace('-', '\u2212'), va='center', ha='left', fontsize=7.5,
                fontweight='bold' if i == len(rows) - 1 else 'normal')
        rows[i] = (name, y)
    ax.set_yticks([y for _, y in rows])
    ax.set_yticklabels([n for n, _ in rows])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(68, 93.5)
    ax.set_ylim(-0.95, len(rows) - 0.45)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('DSC (%)')
    handles = [plt.Line2D([], [], ls='', marker='o', color=BASE, ms=5, label='Baseline'),
               plt.Line2D([], [], ls='', marker='^', color=BASE, ms=5, label='Baseline + SCZ'),
               plt.Line2D([], [], ls='', marker='s', color=DETAIL, ms=5, label='Detail branch, single pass'),
               plt.Line2D([], [], ls='', marker='D', color=ZOOM, ms=5, label='ZoomCLIPSeg')]
    fig.legend(handles=handles, loc='outside upper center', ncol=2, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    save(fig, 'fig_progressive', exact=True)


# ---------------------------------------------------------------- Fig: lesion size
def per_image():
    """Seed-averaged single-pass and two-pass DSC of every OOD image."""
    table = {}
    for r in zrows:
        if r['dataset'] not in OOD or float(r['gt_pixels']) == 0:
            continue
        key = (r['dataset'], r['name'])
        e = table.setdefault(key, dict(frac=float(r['gt_fraction']), single=[], t15=[], zoomed=[]))
        e[r['config']].append(float(r['DSC']))
        if r['config'] == 't15':
            e['zoomed'].append(r['kind'] == 'in')
    frac = np.array([e['frac'] for e in table.values()])
    one = np.array([100 * np.mean(e['single']) for e in table.values()])
    two = np.array([100 * np.mean(e['t15']) for e in table.values()])
    return frac, one, two


def fig_size():
    frac, one, two = per_image()
    edges = [0, .01, .03, .059, .15, 1.01]
    names = ['<1', '1–3', '3–5.9', '5.9–15', '>15']
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.45), gridspec_kw=dict(width_ratios=[1, 1.15]), layout='constrained')
    ax = axes[0]
    x = np.arange(len(names))
    for k, (values, color, marker, label) in enumerate([(one, SINGLE, 's', 'Single pass'), (two, TWO, 'D', 'With SCZ')]):
        means, los, his = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = values[(frac >= lo) & (frac < hi)]
            a, b = ci(v)
            means.append(v.mean()), los.append(a), his.append(b)
        off = (-0.13, 0.13)[k]
        ax.vlines(x + off, los, his, color=color, lw=1.3, zorder=2)
        ax.scatter(x + off, means, s=24, marker=marker, color=color, edgecolor='white', lw=0.7, zorder=3, label=label)
    counts = [int(((frac >= lo) & (frac < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(n={c})' for n, c in zip(names, counts)])
    ax.set_xlabel('Lesion area (% of image)')
    ax.set_ylabel('DSC (%)')
    ax.set_ylim(10, 95)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', handletextpad=0.2)
    panel(ax, 'a', 'Accuracy by lesion size')
    ax = axes[1]
    delta = two - one
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.scatter(100 * frac, delta, s=7, color=TWO, alpha=0.32, lw=0, zorder=2, rasterized=True)
    order = np.argsort(frac)
    win = 61
    xs = np.array([np.exp(np.mean(np.log(100 * frac[order][i:i + win]))) for i in range(0, len(order) - win + 1, 6)])
    ys = np.array([delta[order][i:i + win].mean() for i in range(0, len(order) - win + 1, 6)])
    ax.plot(xs, ys, color=INK, lw=1.4, zorder=4, label=f'running mean ({win} images)')
    ax.set_xscale('log')
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(['0.3', '1', '3', '10', '30'])
    ax.minorticks_off()
    ax.set_xlim(0.12, 75)
    ax.set_ylim(-62, 92)
    ax.set_xlabel('Lesion area (% of image, log scale)')
    ax.set_ylabel('SCZ − single-pass DSC (pp)')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', handlelength=1.4)
    panel(ax, 'b', 'Per-image change')
    fig.get_layout_engine().set(wspace=0.06, w_pad=0.02, h_pad=0.02)
    save(fig, 'fig_size', exact=True)
    return dict(n=len(frac), counts=counts)


if __name__ == '__main__':
    fig_gains()
    fig_progressive()
    print(fig_size())


# ---------------------------------------------------------------- breast ultrasound: per-seed DSC
def fig_busi():
    rows = list(csv.DictReader(open('data/busi_cases.csv', encoding='utf-8-sig')))
    zb = json.load(open('data/zoom_busi_results.json'))['arms']['detail']['tcal']['targets']
    sets, names = ['BUSI', 'BUSBRA', 'BUSUC', 'BUID', 'UDIAT'], ['BUSI\n(in-domain)', 'BUS-BRA', 'BUS-UC', 'BUID', 'UDIAT']

    def dsc(r):
        g, p, t = int(r['gt_pixels']), int(r['pred_pixels']), int(r['tp_pixels'])
        return 1. if g+p == 0 else 2*t/(g+p)

    def seeds(arm, target):
        return [100*np.mean([dsc(r) for r in rows if r['dataset'] == target and r['arm'] == arm and r['seed'] == s]) for s in ('666', '667', '668')]

    series = [('Baseline', BASE, {t: seeds('baseline', t) for t in sets}),
              ('+ Detail branch', DETAIL, {t: seeds('detail', t) for t in sets})]
    series.append(('ZoomCLIPSeg (+ SCZ)', ZOOM, {t: [v+d for v, d in zip(series[1][2][t], zb[t]['contrast']['DSC']['all']['per_seed_mean_pp'])] for t in sets}))
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4), layout='constrained')
    width = 0.24
    for k, (label, color, values) in enumerate(series):
        for i, t in enumerate(sets):
            x = i+(k-1)*(width+0.03)
            mean = np.mean(values[t])
            ax.plot([x, x], [min(values[t]), max(values[t])], color=color, lw=3.2, alpha=0.25, zorder=2, solid_capstyle='round')
            ax.plot([x-width/2, x+width/2], [mean, mean], color=color, lw=1.4, zorder=3, solid_capstyle='butt')
            ax.scatter([x]*3, values[t], s=12, color=color, ec='white', lw=0.5, zorder=4, label=label if i == 0 else None)
    ax.axvline(0.5, color=GRID, lw=0.8)
    ax.set_xticks(range(len(sets)), names)
    ax.set_ylim(44, 88)
    ax.set_yticks(range(50, 81, 10))
    ax.set_ylabel('DSC (%)')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    ax.text(2.5, 47.2, 'out-of-distribution', ha='center', va='center', fontsize=7, color=MUTED)
    fig.legend(loc='outside upper center', ncol=3, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    save(fig, 'fig_busi', exact=True)


fig_busi()
