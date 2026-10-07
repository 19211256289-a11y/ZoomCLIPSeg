"""Statistics and report of the ETIS-Larib evaluation. numpy only."""
import csv
import json

import numpy as np

from .common import NAME, OUTPUT, REPORT, load_protocol, write_json, now, read

STRATA = [('<1%', 0, .01), ('1-3%', .01, .03), ('3-5.9%', .03, .058989688326573426), ('5.9-15%', .058989688326573426, .15), ('>15%', .15, 1.01)]


def load(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('DSC', 'NSD', 'gt_fraction', 'window'):
            row[key] = float(row[key])
        for key in ('gt_pixels', 'pred_pixels', 'tp_pixels', 'fallback'):
            row[key] = int(row[key])
    return rows


def paired(a, b, rng, statistics):
    """a, b: arrays (seeds, images). Difference a - b in percentage points."""
    per_seed = 100*(a-b).mean(1)
    image = 100*(a-b).mean(0)
    n = len(image)
    boot = image[rng.integers(0, n, (statistics['bootstrap_draws'], n))].mean(1)
    observed = abs(image.mean())
    count, draws, block = 0, statistics['permutation_draws'], 10000
    for start in range(0, draws, block):
        signs = rng.choice((-1., 1.), size=(min(block, draws-start), n))
        count += int((np.abs((signs*image).mean(1)) >= observed-1e-12).sum())
    return dict(mean_pp=float(per_seed.mean()), per_seed_pp=[float(v) for v in per_seed], seeds_positive=int((per_seed > 0).sum()),
                ci95_pp=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], permutation_p=float((count+1)/(draws+1)),
                improved=int((image > 1e-9).sum()), worsened=int((image < -1e-9).sum()), n=n)


def write_report():
    protocol = load_protocol()
    statistics, seeds, arms, config = protocol['statistics'], protocol['seeds'], protocol['arms'], protocol['primary_config']
    rng = np.random.default_rng(statistics['rng_seed'])
    data, names, meta = {}, None, {}
    for arm in arms:
        for seed in seeds:
            folder = OUTPUT/'evaluations'/f'{arm}_seed{seed}'/'ETIS'
            if not read(folder/'manifest.json').get('complete'):
                raise ValueError(f'Evaluation not complete: {arm}_seed{seed}')
            rows = load(folder/'cases.csv')
            for name_ in ('single', config):
                part = sorted((r for r in rows if r['config'] == name_), key=lambda r: r['name'])
                if names is None:
                    names = [r['name'] for r in part]
                    fraction = np.array([r['gt_fraction'] for r in part])
                if [r['name'] for r in part] != names:
                    raise ValueError('Case lists differ between runs')
                data[arm, seed, name_] = part
    if len(names) != protocol['dataset']['n']:
        raise ValueError('Unexpected number of cases')
    array = lambda arm, cfg, key='DSC': np.array([[r[key] for r in data[arm, s, cfg]] for s in seeds], float)
    summary = {}
    for arm in arms:
        for cfg in ('single', config):
            dsc, nsd = array(arm, cfg), array(arm, cfg, 'NSD')
            summary[f'{arm}/{cfg}'] = dict(
                DSC=float(100*dsc.mean()), DSC_per_seed=[float(v) for v in 100*dsc.mean(1)], DSC_sd=float(np.std(100*dsc.mean(1), ddof=1)),
                NSD=float(100*nsd.mean()), small_DSC=float(100*dsc[:, fraction <= protocol['small_target_threshold']].mean()),
                failures_per_seed=[int(v) for v in (dsc < .5).sum(1)],
                empty_per_seed=[int(sum(r['pred_pixels'] == 0 for r in data[arm, s, cfg])) for s in seeds])
        summary[f'{arm}/second_pass_share'] = float(np.mean([[r['kind'] != 'none' for r in data[arm, s, config]] for s in seeds]))
        summary[f'{arm}/second_pass_empty_per_seed'] = [int(sum(r['fallback'] for r in data[arm, s, config])) for s in seeds]
    contrasts = {
        'two_pass_on_detail': paired(array('detail', config), array('detail', 'single'), rng, statistics),
        'two_pass_on_baseline': paired(array('baseline', config), array('baseline', 'single'), rng, statistics),
        'detail_minus_baseline_single': paired(array('detail', 'single'), array('baseline', 'single'), rng, statistics),
        'detail_minus_baseline_two_pass': paired(array('detail', config), array('baseline', config), rng, statistics),
        'full_method_minus_baseline_single': paired(array('detail', config), array('baseline', 'single'), rng, statistics)}
    strata = []
    for label, low, high in STRATA:
        keep = (fraction > low) & (fraction <= high) if low else (fraction <= high)
        if keep.any():
            strata.append(dict(stratum=label, n=int(keep.sum()),
                               **{f'{arm}_{cfg}': float(100*array(arm, cfg)[:, keep].mean()) for arm in arms for cfg in ('single', config)},
                               detail_second_pass_share=float(np.mean([[r['kind'] != 'none' for r, k in zip(data['detail', s, config], keep) if k] for s in seeds]))))
    gate, primary = protocol['gate'], contrasts['two_pass_on_detail']
    checks = dict(g1=primary['mean_pp'] >= gate['g1_dsc_gain_at_least_pp'], g2=primary['seeds_positive'] == len(seeds))
    plan = read(OUTPUT/'evaluation_plan.json')
    result = dict(name=NAME, created_at=now(), plan_digest=plan['digest'], seeds=seeds, n=len(names), verdict='pass' if all(checks.values()) else 'fail',
                  checks=checks, summary=summary, contrasts=contrasts, strata=strata, lesion_fraction_median=float(np.median(fraction)),
                  overlap_with_known_data=plan['overlap_with_known_data'])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/'results.json', result)
    f = lambda v: f'{v:.2f}'
    g = lambda v: f'{v:+.2f}'
    lines = [f'# {NAME}: one-time evaluation on ETIS-Larib', '', protocol['purpose'].strip(), '',
             f"{len(names)} images, every image with a polyp; median lesion area {100*result['lesion_fraction_median']:.2f}% of the image. "
             f"Seeds {', '.join(map(str, seeds))}. Fixed sentence: \"{protocol['prompt']}\".", '',
             f"**Verdict: {result['verdict']}** (detail arm, two-pass minus single-pass): {g(primary['mean_pp'])} pp "
             f"(per seed {', '.join(g(v) for v in primary['per_seed_pp'])})", '',
             '## DSC (%)', '', '| arm | inference | mean | SD over seeds | per seed | NSD | small-lesion DSC | failures per seed |', '|---|---|---|---|---|---|---|---|']
    for arm in arms:
        for cfg, label in (('single', 'single pass'), (config, 'two-pass')):
            s = summary[f'{arm}/{cfg}']
            lines.append(f"| {arm} | {label} | {f(s['DSC'])} | {f(s['DSC_sd'])} | {', '.join(f(v) for v in s['DSC_per_seed'])} | {f(s['NSD'])} | "
                         f"{f(s['small_DSC'])} | {', '.join(map(str, s['failures_per_seed']))} |")
    lines += ['', '## Paired differences (pp)', '', '| contrast | mean | per seed | seeds > 0 | 95% CI over images | permutation p | images better / worse |', '|---|---|---|---|---|---|---|']
    for key, c in contrasts.items():
        lines.append(f"| {key} | {g(c['mean_pp'])} | {', '.join(g(v) for v in c['per_seed_pp'])} | {c['seeds_positive']}/{len(seeds)} | "
                     f"{g(c['ci95_pp'][0])} to {g(c['ci95_pp'][1])} | {c['permutation_p']:.4f} | {c['improved']} / {c['worsened']} |")
    lines += ['', '## By lesion area (share of the image)', '', '| area | n | baseline single | baseline two-pass | detail single | detail two-pass | second pass (detail) |', '|---|---|---|---|---|---|---|']
    for s in strata:
        lines.append(f"| {s['stratum']} | {s['n']} | {f(s['baseline_single'])} | {f(s['baseline_'+config])} | {f(s['detail_single'])} | "
                     f"{f(s['detail_'+config])} | {100*s['detail_second_pass_share']:.0f}% |")
    lines += ['', '## Gate (fixed in advance)', '',
              f"- g1 two-pass gain of the detail arm {g(primary['mean_pp'])} pp (need >= {gate['g1_dsc_gain_at_least_pp']:+.1f}): {checks['g1']}",
              f"- g2 positive for every seed ({primary['seeds_positive']}/{len(seeds)}): {checks['g2']}", '',
              f"Second pass applied to {100*summary['detail/second_pass_share']:.0f}% of the images (detail arm), {100*summary['baseline/second_pass_share']:.0f}% (baseline).",
              'Exact duplicates of images used before: '+(json.dumps(result['overlap_with_known_data']) if result['overlap_with_known_data'] else 'none found.'),
              'Disclosures: '+' '.join(protocol['disclosures'])]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return str(REPORT/'report.md')
