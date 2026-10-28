"""BCa bootstrap intervals for the main paired contrasts (Table 2, five seeds; Table 4, three seeds), from the per-image
records. Statistic: mean over images of the seed-averaged DSC difference (per set) and the equal-weight mean over the
four OOD sets (macro). Resampling is within each set; the jackknife for the acceleration leaves one image out.
Writes data/review/bca_intervals.json."""
import csv, json, sys
import numpy as np
from scipy.stats import norm

S = ['Kvasir', 'ColonDB', 'ClinicDB', 'CVC300', 'BKAI']
OOD = S[1:]
def load(f):
    return list(csv.DictReader(open(f, encoding='utf-8-sig')))

def table(rows, seeds, cfg_key, cfg):
    d = {}
    for r in rows:
        if r.get(cfg_key, 'single') == cfg and r['seed'] in seeds:
            d[(r['dataset'], r['seed'], r['name'])] = 100*float(r['DSC'])
    return d

def seed_avg_diff(a, b, seeds):
    names = {t: sorted({n for (tt, s, n) in a if tt == t and s == seeds[0]}) for t in S}
    return {t: np.array([np.mean([a[(t, s, n)]-b[(t, s, n)] for s in seeds]) for n in names[t]]) for t in S}

def bca(samples, stat, rng, draws=10000):
    """samples: dict set -> array of per-image differences; stat(dict) -> float."""
    obs = stat(samples)
    boot = np.empty(draws)
    for i in range(draws):
        boot[i] = stat({t: v[rng.integers(0, len(v), len(v))] for t, v in samples.items()})
    z0 = norm.ppf(np.clip(np.mean(boot < obs), 1e-6, 1-1e-6))
    jack = []
    for t, v in samples.items():
        for k in range(len(v)):
            jack.append(stat({**samples, t: np.delete(v, k)}))
    jack = np.array(jack); jm = jack.mean()
    a = np.sum((jm-jack)**3)/(6*np.sum((jm-jack)**2)**1.5)
    def q(alpha):
        z = norm.ppf(alpha)
        return norm.cdf(z0+(z0+z)/(1-a*(z0+z)))
    lo, hi = np.quantile(boot, q(.025)), np.quantile(boot, q(.975))
    plo, phi = np.quantile(boot, .025), np.quantile(boot, .975)
    return dict(mean=float(obs), bca=[float(lo), float(hi)], percentile=[float(plo), float(phi)], z0=float(z0), a=float(a))

def contrasts(diff, rng):
    out = {}
    for t in S:
        out[t] = bca({t: diff[t]}, lambda d: float(np.mean(d[t])), rng)
    out['OOD'] = bca({t: diff[t] for t in OOD}, lambda d: float(np.mean([np.mean(d[t]) for t in OOD])), rng)
    return out

rng = np.random.default_rng(20261011)
five = ['666', '667', '668', '669', '670']; three = five[:3]
tz = load('data/test_zoom_cases.csv')+load('data/zoom_new_seed_cases.csv')
fa = load('data/test_four_arms_cases.csv')+load('data/kvasir_new_seed_cases.csv')
bz = load('data/review/baseline_zoom_cases.csv')
zoom5 = table(tz, five, 'config', 't15'); single5 = table(tz, five, 'config', 'single')
base5 = table([r for r in fa if r['arm'] == 'baseline'], five, 'arm', 'baseline')
basescz5 = table(bz, five, 'config', 't15')
result = {
    'table2_zoomclipseg_minus_baseline': contrasts(seed_avg_diff(zoom5, base5, five), rng),
    'table2_zoomclipseg_minus_baseline_scz': contrasts(seed_avg_diff(zoom5, basescz5, five), rng),
    'table4_scz_minus_single_three_seeds': contrasts(seed_avg_diff(table(tz, three, 'config', 't15'), table(tz, three, 'config', 'single'), three), rng),
    'five_seed_scz_minus_single': contrasts(seed_avg_diff(zoom5, single5, five), rng),
}
json.dump(result, open('data/review/bca_intervals.json', 'w'), indent=1)
for k, v in result.items():
    print(k)
    for t, r in v.items():
        print(f"  {t:9s} {r['mean']:+.2f}  percentile [{r['percentile'][0]:+.2f}, {r['percentile'][1]:+.2f}]  BCa [{r['bca'][0]:+.2f}, {r['bca'][1]:+.2f}]  z0={r['z0']:+.3f} a={r['a']:+.3f}")
