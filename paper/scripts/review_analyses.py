"""Additional analyses requested in review, computed from accepted per-image records only (no new model runs).
Run from the paper folder: python scripts/review_analyses.py   ->  data/review/review_results.json

A. Polyps, five seeds: ZoomCLIPSeg against (i) the baseline with SCZ and (ii) the single-pass baseline.
B. Breast ultrasound: is the gain of the detail branch a detection effect? Empty predictions, image-level
   sensitivity/specificity, and the gain restricted to images on which both models predict a lesion.
C. ETIS-Larib: what happens to lesions of 3-5.9% of the image under SCZ.
"""
import csv
import json
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(20261006)
SETS = ['Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI']
OOD = SETS[1:]
S5 = ['666', '667', '668', '669', '670']
S3 = S5[:3]


def read(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def stats(diff, draws=10000, perms=100000):
    """Mean, 95% bootstrap interval over images and two-sided sign-flip permutation p of per-image differences (pp)."""
    diff = np.asarray(diff, float)
    boot = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(1)
    signs = rng.choice([-1., 1.], (perms, len(diff)))
    null = np.abs((signs*diff).mean(1))
    p = (1+np.sum(null >= abs(diff.mean())-1e-12))/(perms+1)
    return dict(mean_pp=float(diff.mean()), ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p=float(p), n=len(diff))


def macro_stats(per_set, draws=10000, perms=20000):
    """Equal-weight mean over datasets; images are resampled / sign-flipped within each dataset."""
    means = [np.mean(v) for v in per_set.values()]
    obs = float(np.mean(means))
    boot = np.mean([np.asarray(v)[rng.integers(0, len(v), (draws, len(v)))].mean(1) for v in per_set.values()], 0)
    null = np.mean([(rng.choice([-1., 1.], (perms, len(v)))*np.asarray(v)).mean(1) for v in per_set.values()], 0)
    p = (1+np.sum(np.abs(null) >= abs(obs)-1e-12))/(perms+1)
    return dict(mean_pp=obs, ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p=float(p))


result = {}

# ------------------------------------------------------------------ A. polyps, five seeds
table = defaultdict(dict)      # (arm, config) -> (dataset, seed, name) -> DSC
for r in read('data/review/baseline_zoom_cases.csv'):
    table[('baseline', r['config'])][(r['dataset'], r['seed'], r['name'])] = 100*float(r['DSC'])
for path in ('data/test_zoom_cases.csv', 'data/zoom_new_seed_cases.csv'):
    for r in read(path):
        table[('detail', r['config'])][(r['dataset'], r['seed'], r['name'])] = 100*float(r['DSC'])


def seed_mean(arm, config, dataset, seed):
    return float(np.mean([v for (d, s, _), v in table[(arm, config)].items() if d == dataset and s == seed]))


def image_diff(a, b, dataset):
    """Per-image difference a - b, averaged over the five seeds."""
    names = sorted({n for (d, s, n) in table[a] if d == dataset})
    return [np.mean([table[a][(dataset, s, n)]-table[b][(dataset, s, n)] for s in S5]) for n in names]


means = {f'{arm}/{cfg}': {t: float(np.mean([seed_mean(arm, cfg, t, s) for s in S5])) for t in SETS}
         for arm, cfg in (('baseline', 'single'), ('baseline', 't15'), ('detail', 'single'), ('detail', 't15'))}
for k, v in means.items():
    v['OOD'] = float(np.mean([v[t] for t in OOD]))
contrasts = {}
for label, a, b in (('full_vs_baseline_with_scz', ('detail', 't15'), ('baseline', 't15')),
                    ('full_vs_baseline', ('detail', 't15'), ('baseline', 'single')),
                    ('scz_on_baseline', ('baseline', 't15'), ('baseline', 'single')),
                    ('scz_on_detail', ('detail', 't15'), ('detail', 'single'))):
    per_set = {t: image_diff(a, b, t) for t in SETS}
    per_seed = [float(np.mean([seed_mean(*a, t, s)-seed_mean(*b, t, s) for t in OOD])) for s in S5]
    contrasts[label] = dict(sets={t: stats(per_set[t]) for t in SETS}, ood_macro=macro_stats({t: per_set[t] for t in OOD}),
                            ood_macro_per_seed_pp=per_seed, seeds_positive=int(sum(v > 0 for v in per_seed)))
result['polyps_five_seeds'] = dict(means=means, contrasts=contrasts)

# ------------------------------------------------------------------ B. breast ultrasound: detection or delineation?
busi = [r for r in read('data/busi_cases.csv') if r['dataset'] != 'Val']
BOOD = ['BUSBRA', 'BUSUC', 'BUID', 'UDIAT']


def dsc(r):
    g, p, t = int(r['gt_pixels']), int(r['pred_pixels']), int(r['tp_pixels'])
    return 100. if g+p == 0 else 200*t/(g+p)


rows = {(r['dataset'], r['arm'], r['seed'], r['name']): r for r in busi}
ultra = dict(per_seed=[])
for s in S3:
    e = dict(seed=s)
    for arm in ('baseline', 'detail'):
        ood = [r for r in busi if r['arm'] == arm and r['seed'] == s and r['dataset'] in BOOD]
        assert all(int(r['gt_pixels']) > 0 for r in ood)
        test = [r for r in busi if r['arm'] == arm and r['seed'] == s and r['dataset'] == 'BUSI']
        pos, neg = [r for r in test if int(r['gt_pixels']) > 0], [r for r in test if int(r['gt_pixels']) == 0]
        e[arm] = dict(ood_n=len(ood), ood_empty=sum(int(r['pred_pixels']) == 0 for r in ood),
                      ood_no_overlap=sum(int(r['tp_pixels']) == 0 for r in ood),
                      ood_pooled_dsc=float(np.mean([dsc(r) for r in ood])),
                      busi_sensitivity=float(np.mean([int(r['pred_pixels']) > 0 for r in pos])), busi_pos=len(pos),
                      busi_hit_rate=float(np.mean([int(r['tp_pixels']) > 0 for r in pos])),
                      busi_specificity=float(np.mean([int(r['pred_pixels']) == 0 for r in neg])), busi_neg=len(neg))
    # paired decomposition on the pooled OOD images of this seed
    names = [(d, n) for (d, a, sd, n) in rows if a == 'baseline' and sd == s and d in BOOD]
    diff = np.array([dsc(rows[(d, 'detail', s, n)])-dsc(rows[(d, 'baseline', s, n)]) for d, n in names])
    b_empty = np.array([int(rows[(d, 'baseline', s, n)]['pred_pixels']) == 0 for d, n in names])
    d_empty = np.array([int(rows[(d, 'detail', s, n)]['pred_pixels']) == 0 for d, n in names])
    both = ~b_empty & ~d_empty
    e['pooled_gain_pp'] = float(diff.mean())
    e['gain_both_nonempty_pp'] = float(diff[both].mean())
    e['n_both_nonempty'] = int(both.sum())
    e['share_of_gain_from_baseline_empty'] = float(diff[b_empty].sum()/diff.sum())
    e['share_of_gain_from_both_nonempty'] = float(diff[both].sum()/diff.sum())
    e['both_nonempty_stats'] = stats(diff[both], perms=20000)
    ultra['per_seed'].append(e)
# seed-averaged per-image gain on images where no model of either arm returns an empty mask
keep, gains = defaultdict(list), []
for d in BOOD:
    for n in sorted({n for (dd, a, sd, n) in rows if dd == d}):
        rs = [rows[(d, a, s, n)] for a in ('baseline', 'detail') for s in S3]
        if all(int(r['pred_pixels']) > 0 for r in rs):
            keep[d].append(np.mean([dsc(rows[(d, 'detail', s, n)])-dsc(rows[(d, 'baseline', s, n)]) for s in S3]))
ultra['never_empty'] = dict(n={d: len(v) for d, v in keep.items()}, sets={d: stats(v, perms=20000) for d, v in keep.items()},
                            ood_macro=macro_stats(keep), pooled=stats([x for v in keep.values() for x in v], perms=20000))
result['ultrasound_detection'] = ultra

# ------------------------------------------------------------------ C. ETIS, lesions of 3-5.9% of the image
etis = read('data/review/etis_cases.csv')
E = {(r['arm'], r['config'], r['seed'], r['name']): r for r in etis}
names = sorted({r['name'] for r in etis})
frac = {n: float(E[('detail', 'single', '666', n)]['gt_fraction']) for n in names}
strata = {}
for label, lo, hi in (('<1%', 0, .01), ('1-3%', .01, .03), ('3-5.9%', .03, .0589897), ('>5.9%', .0589897, 2)):
    sel = [n for n in names if lo <= frac[n] < hi]
    block = dict(n=len(sel))
    for arm in ('baseline', 'detail'):
        one = np.array([[100*float(E[(arm, 'single', s, n)]['DSC']) for s in S5] for n in sel])
        two = np.array([[100*float(E[(arm, 't15', s, n)]['DSC']) for s in S5] for n in sel])
        zoomed = np.array([[E[(arm, 't15', s, n)]['kind'] == 'in' for s in S5] for n in sel])
        prec1 = np.array([[float(E[(arm, 'single', s, n)]['foreground_precision'] or 0) for s in S5] for n in sel])
        prec2 = np.array([[float(E[(arm, 't15', s, n)]['foreground_precision'] or 0) for s in S5] for n in sel])
        rec1 = np.array([[float(E[(arm, 'single', s, n)]['foreground_recall'] or 0) for s in S5] for n in sel])
        rec2 = np.array([[float(E[(arm, 't15', s, n)]['foreground_recall'] or 0) for s in S5] for n in sel])
        area1 = np.array([[int(E[(arm, 'single', s, n)]['pred_pixels'])/max(1, int(E[(arm, 'single', s, n)]['gt_pixels'])) for s in S5] for n in sel])
        d = two-one
        img = d.mean(1)
        order = np.sort(img)
        block[arm] = dict(single=float(one.mean()), scz=float(two.mean()), stats=stats(img, perms=20000),
                          second_pass_share=float(zoomed.mean()),
                          cells_worse_10=int((d < -10).sum()), cells_better_10=int((d > 10).sum()), cells=int(d.size),
                          images_worse=int((img < -1).sum()), images_better=int((img > 1).sum()),
                          loss_from_worst3_images_pp=float(order[:3].sum()/len(img)), mean_without_worst3_pp=float(order[3:].mean()),
                          precision=[float(prec1[zoomed].mean()), float(prec2[zoomed].mean())],
                          recall=[float(rec1[zoomed].mean()), float(rec2[zoomed].mean())],
                          first_pass_area_ratio_median=float(np.median(area1)),
                          first_pass_dsc_of_worsened_cells=float(one[d < -10].mean()) if (d < -10).any() else None,
                          first_pass_area_ratio_of_worsened_cells=float(np.median(area1[d < -10])) if (d < -10).any() else None)
    strata[label] = block
result['etis_strata'] = strata

json.dump(result, open('data/review/review_results.json', 'w'), indent=1)
print(json.dumps(result, indent=1))
