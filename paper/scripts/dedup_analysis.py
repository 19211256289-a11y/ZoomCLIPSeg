"""Sensitivity of the main polyp contrasts to duplicated test images (no new model runs).
Run from the paper folder: python scripts/dedup_analysis.py -> data/review/dedup_results.json
CVC-ColonDB is reduced to one image per exact file hash, and its images that also occur in CVC-300 are removed, so
that every distinct image is counted once in the equal-weight mean of the four OOD sets. Five seeds."""
import csv
import json

import numpy as np

SEEDS = ['666', '667', '668', '669', '670']
OOD = ['ColonDB', 'ClinicDB', 'CVC300', 'BKAI']
groups = json.load(open('data/review/image_hash_groups.json'))       # dataset -> hash -> image names (sealed test cohorts)
shared = set(groups['ColonDB']) & set(groups['CVC300'])
keep = {t: {names[0] for names in groups[t].values()} for t in OOD}
keep['ColonDB'] = {names[0] for digest, names in groups['ColonDB'].items() if digest not in shared}
read = lambda path: list(csv.DictReader(open(path, encoding='utf-8-sig')))
table = {}
for r in read('data/review/baseline_zoom_cases.csv'):
    table.setdefault(('baseline', r['config']), {})[(r['dataset'], r['seed'], r['name'])] = 100*float(r['DSC'])
for path in ('data/test_zoom_cases.csv', 'data/zoom_new_seed_cases.csv'):
    for r in read(path):
        table.setdefault(('detail', r['config']), {})[(r['dataset'], r['seed'], r['name'])] = 100*float(r['DSC'])
names = {t: sorted({k[2] for k in table[('baseline', 'single')] if k[0] == t}) for t in OOD}
stem = lambda name: name.rsplit('.', 1)[0]            # result rows are keyed by the mask file, whose extension can differ
keep = {t: {n for n in names[t] if stem(n) in {stem(k) for k in keep[t]}} for t in OOD}
assert all(len(keep[t]) == len({stem(n) for n in keep[t]}) for t in OOD) and len(keep['BKAI']) == 100 and len(keep['ClinicDB']) == 61


def stats(a, b, subset, rng, draws=10000, perms=20000):
    diff = {t: np.array([np.mean([table[a][(t, s, n)]-table[b][(t, s, n)] for s in SEEDS]) for n in names[t] if subset is None or n in subset[t]])
            for t in OOD}
    obs = float(np.mean([v.mean() for v in diff.values()]))
    boot = np.mean([v[rng.integers(0, len(v), (draws, len(v)))].mean(1) for v in diff.values()], 0)
    null = np.mean([(rng.choice([-1., 1.], (perms, len(v)))*v).mean(1) for v in diff.values()], 0)
    return dict(mean_pp=obs, ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p=float((1+np.sum(np.abs(null) >= abs(obs)-1e-12))/(perms+1)), n={t: len(v) for t, v in diff.items()})


CONTRASTS = dict(full_vs_baseline=(('detail', 't15'), ('baseline', 'single')), full_vs_baseline_with_scz=(('detail', 't15'), ('baseline', 't15')),
                 scz_on_detail=(('detail', 't15'), ('detail', 'single')), scz_on_baseline=(('baseline', 't15'), ('baseline', 'single')),
                 detail_single_pass=(('detail', 'single'), ('baseline', 'single')))
result = dict(shared_between_colondb_and_cvc300=len(shared), colondb_distinct=len(groups['ColonDB']), colondb_kept=len(keep['ColonDB']), contrasts={})
for index, (name, (a, b)) in enumerate(CONTRASTS.items()):
    result['contrasts'][name] = dict(all_images=stats(a, b, None, np.random.default_rng([20261008, index, 0])),
                                     deduplicated=stats(a, b, keep, np.random.default_rng([20261008, index, 1])))
json.dump(result, open('data/review/dedup_results.json', 'w'), indent=1)
print(result['shared_between_colondb_and_cvc300'], result['colondb_distinct'], result['colondb_kept'])
for name, r in result['contrasts'].items():
    f = lambda s: f"{s['mean_pp']:+.2f} [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}] p={s['p']:.4f}"
    print(f"{name:28s} all {f(r['all_images'])} | dedup {f(r['deduplicated'])} n={r['deduplicated']['n']}")
