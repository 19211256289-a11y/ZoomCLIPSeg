"""Five-seed comparison of the Polyp-PVT reference (polyp_pvt_kvasir_v1: seeds 666-668; pvt_seeds_v1: 669, 670) with
our baseline and full method, from accepted per-image records. Run from the paper folder: python scripts/pvt_five.py
-> data/review/pvt_five_seed.json"""
import csv
import json

import numpy as np

SEEDS = ['666', '667', '668', '669', '670']
SETS = ['Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI', 'ETIS']
OOD = SETS[1:5]
read = lambda path: list(csv.DictReader(open(path, encoding='utf-8-sig')))
table = {'pvt': {}, 'baseline': {}, 'full': {}}
for r in read('data/review/pvt_cases.csv'):
    if r['model'] == 'pvt':
        table['pvt'][(str(r['seed']), r['dataset'], r['name'])] = 100*float(r['DSC'])
for path in ('data/test_zoom_cases.csv', 'data/zoom_new_seed_cases.csv'):
    for r in read(path):
        if r['config'] == 't15':
            table['full'][(r['seed'], r['dataset'], r['name'])] = 100*float(r['DSC'])
for r in read('data/review/baseline_zoom_cases.csv'):
    if r['config'] == 'single':
        table['baseline'][(r['seed'], r['dataset'], r['name'])] = 100*float(r['DSC'])
for r in read('data/review/etis_cases.csv'):
    if (r['arm'], r['config']) == ('baseline', 'single'):
        table['baseline'][(r['seed'], 'ETIS', r['name'])] = 100*float(r['DSC'])
    if (r['arm'], r['config']) == ('detail', 't15'):
        table['full'][(r['seed'], 'ETIS', r['name'])] = 100*float(r['DSC'])
names = {t: sorted(n for (s, tt, n) in table['pvt'] if s == SEEDS[0] and tt == t) for t in SETS}
for model in table:
    for s in SEEDS:
        for t in SETS:
            assert sorted(n for (ss, tt, n) in table[model] if ss == s and tt == t) == names[t], (model, s, t)
msd = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), per_seed=[float(x) for x in v])
summary = {}
for model in table:
    per = {t: [float(np.mean([table[model][(s, t, n)] for n in names[t]])) for s in SEEDS] for t in SETS}
    summary[model] = dict(sets={t: dict(n=len(names[t]), DSC=msd(per[t])) for t in SETS},
                          OOD_DSC=msd([float(np.mean([per[t][i] for t in OOD])) for i in range(len(SEEDS))]))
rng = np.random.default_rng(20261018)


def stats(d, draws=10000, perms=20000):
    d = np.asarray(d, float)
    boot = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    null = np.abs((rng.choice([-1., 1.], (perms, len(d)))*d).mean(1))
    return dict(mean_pp=float(d.mean()), ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(null >= abs(d.mean())-1e-12))/(perms+1)))


def macro(per, draws=10000, perms=20000):
    values = [np.asarray(v, float) for v in per.values()]
    observed = float(np.mean([v.mean() for v in values]))
    boot = np.mean([v[rng.integers(0, len(v), (draws, len(v)))].mean(1) for v in values], 0)
    null = np.mean([(rng.choice([-1., 1.], (perms, len(v)))*v).mean(1) for v in values], 0)
    return dict(mean_pp=observed, ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(np.abs(null) >= abs(observed)-1e-12))/(perms+1)))


contrasts = {}
for other in ('baseline', 'full'):
    per = {t: [float(np.mean([table['pvt'][(s, t, n)]-table[other][(s, t, n)] for s in SEEDS])) for n in names[t]] for t in SETS}
    contrasts['pvt_minus_'+other] = dict(
        sets={t: stats(per[t]) for t in SETS}, ood_macro=macro({t: per[t] for t in OOD}),
        ood_macro_per_seed_pp=[float(np.mean([np.mean([table['pvt'][(s, t, n)]-table[other][(s, t, n)] for n in names[t]]) for t in OOD]))
                               for s in SEEDS],
        etis_per_seed_pp=[float(np.mean([table['pvt'][(s, 'ETIS', n)]-table[other][(s, 'ETIS', n)] for n in names['ETIS']])) for s in SEEDS])
json.dump(dict(seeds=SEEDS, summary=summary, contrasts=contrasts), open('data/review/pvt_five_seed.json', 'w'), indent=1)
for k, c in contrasts.items():
    print(k, 'Kvasir %+.2f' % c['sets']['Kvasir']['mean_pp'], 'OOD %+.2f [%.2f, %.2f]' % (c['ood_macro']['mean_pp'], *c['ood_macro']['ci95_pp']),
          'ETIS %+.2f [%.2f, %.2f]' % (c['sets']['ETIS']['mean_pp'], *c['sets']['ETIS']['ci95_pp']))
