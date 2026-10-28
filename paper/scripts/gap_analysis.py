"""Gap between the fixed-sentence models and the dedicated network by lesion size, from accepted per-image records
(no new model runs). Run from the paper folder: python scripts/gap_analysis.py  ->  data/review/gap_by_size.json

Seeds 666-670. For every lesion-size stratum of the pooled OOD polyp images and
of ETIS-Larib: seed-averaged DSC of the baseline, of the full method and of Polyp-PVT, and paired differences with a
bootstrap interval over images and a sign-flip permutation test."""
import csv
import json

import numpy as np

SEEDS = ['666', '667', '668', '669', '670']
OOD = ['ColonDB', 'ClinicDB', 'CVC300', 'BKAI']
SMALL = 0.058989688326573426
BINS = [(0., .01, '<1%'), (.01, .03, '1-3%'), (.03, SMALL, '3-5.9%'), (SMALL, .15, '5.9-15%'), (.15, 1.01, '>=15%')]


def read(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


table = {}      # model -> (dataset, name) -> seed -> DSC in percent
fraction = {}   # (dataset, name) -> lesion area as a fraction of the image


def add(model, row):
    if row['seed'] in SEEDS:
        table.setdefault(model, {}).setdefault((row['dataset'], row['name']), {})[row['seed']] = 100*float(row['DSC'])
        fraction[(row['dataset'], row['name'])] = float(row['gt_fraction'])


for r in read('data/review/baseline_zoom_cases.csv'):
    if r['config'] == 'single':
        add('baseline', r)
    if r['config'] == 't15':
        add('baseline_scz', r)
for path in ('data/test_zoom_cases.csv', 'data/zoom_new_seed_cases.csv'):
    for r in read(path):
        if r['config'] == 't15':
            add('full', r)
for r in read('data/review/etis_cases.csv'):
    if (r['arm'], r['config']) == ('baseline', 'single'):
        add('baseline', r)
    if (r['arm'], r['config']) == ('baseline', 't15'):
        add('baseline_scz', r)
    if (r['arm'], r['config']) == ('detail', 't15'):
        add('full', r)
for r in read('data/review/pvt_cases.csv'):
    if r['model'] == 'pvt':
        add('pvt', dict(r, seed=str(r['seed'])))

keys = sorted(table['pvt'])
for model in ('baseline', 'baseline_scz', 'full', 'pvt'):
    assert sorted(table[model]) == keys and all(len(v) == len(SEEDS) for v in table[model].values()), model
value = {m: {k: float(np.mean([table[m][k][s] for s in SEEDS])) for k in keys} for m in table}


def stats(diff, rng, draws=10000, perms=20000):
    diff = np.asarray(diff, float)
    boot = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(1)
    null = np.abs((rng.choice([-1., 1.], (perms, len(diff)))*diff).mean(1))
    return dict(mean_pp=float(diff.mean()), ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(null >= abs(diff.mean())-1e-12))/(perms+1)))


result = {}
for cohort, datasets in (('OOD pooled', OOD), ('ETIS', ['ETIS'])):
    rows = []
    for index, (low, high, label) in enumerate(BINS+[(0., 1.01, 'all')]):
        chosen = [k for k in keys if k[0] in datasets and low <= fraction[k] < high and fraction[k] > 0]
        rng = np.random.default_rng([20261007, len(result), index])
        get = lambda model: np.array([value[model][k] for k in chosen])
        rows.append(dict(stratum=label, n=len(chosen), baseline=float(get('baseline').mean()), full=float(get('full').mean()),
                         pvt=float(get('pvt').mean()),
                         pvt_minus_baseline=stats(get('pvt')-get('baseline'), rng),
                         pvt_minus_full=stats(get('pvt')-get('full'), rng),
                         full_minus_baseline=stats(get('full')-get('baseline'), rng),
                         # drawn after the three contrasts above, so that their intervals are unchanged
                         baseline_scz=float(get('baseline_scz').mean()),
                         pvt_minus_baseline_scz=stats(get('pvt')-get('baseline_scz'), rng)))
    result[cohort] = rows
json.dump(dict(seeds=SEEDS, note='seed-averaged per-image DSC (%); pooled images, not the equal-weight mean of the sets',
               strata=result), open('data/review/gap_by_size.json', 'w'), indent=1)
for cohort, rows in result.items():
    print(cohort)
    for r in rows:
        a, b = r['pvt_minus_baseline'], r['pvt_minus_full']
        print(f"  {r['stratum']:8s} n={r['n']:3d} baseline {r['baseline']:6.2f} full {r['full']:6.2f} pvt {r['pvt']:6.2f} | gap of baseline "
              f"{a['mean_pp']:+6.2f} [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}] | gap of full {b['mean_pp']:+6.2f} "
              f"[{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}] p={b['p']:.4f} | full-baseline {r['full_minus_baseline']['mean_pp']:+.2f} "
              f"[{r['full_minus_baseline']['ci95_pp'][0]:+.2f}, {r['full_minus_baseline']['ci95_pp'][1]:+.2f}]")
